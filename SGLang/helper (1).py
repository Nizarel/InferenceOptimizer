"""
Lab 1: KV Cache - LLM Inference Fundamentals
=============================================

This module provides helper functions for understanding LLM inference:
1. Tokenization and Detokenization
2. Auto-regressive Decoding
3. Scaled Dot-Product Attention
4. KV Cache Optimization
"""

# ============================================================
# 1. Imports
# ============================================================

import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

# Suppress transformers library warnings
logging.set_verbosity_error()


# ============================================================
# 2. Configuration
# ============================================================

# Default model path - can be overridden when creating TinyLLM instance
DEFAULT_MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


# ============================================================
# 3. Helper Functions
# ============================================================

# ------------------------------
# 3.1 Sampling Function
# ------------------------------

def sample(logits: Tensor, temperature: float = 0.0) -> Tensor:
    """Sample the next token from logits.
    
    Sampling strategy determines the diversity and determinism of generated text:
    - temperature = 0: Greedy Decoding, always select the highest probability token
    - temperature > 0: Temperature sampling, higher temperature = more random output
    
    How temperature works:
    - Divide logits by temperature before applying softmax
    - temperature < 1: Sharper probability distribution, high-prob tokens more likely
    - temperature > 1: Flatter probability distribution, low-prob tokens have a chance
    
    Args:
        logits: Raw model output logits, shape (batch_size, vocab_size)
        temperature: Temperature parameter controlling sampling randomness
            - 0: Greedy decoding (deterministic)
            - 0.1-0.7: Lower randomness, suitable for factual tasks
            - 0.7-1.0: Medium randomness, suitable for creative writing
            - >1.0: High randomness, may produce incoherent output
    
    Returns:
        next_id: Sampled token ID, shape (batch_size, 1)
    """
    if temperature > 0:
        # Temperature sampling: apply softmax after dividing logits by temperature
        probs = torch.softmax(logits / temperature, dim=-1)
        # Randomly sample one token from the probability distribution
        next_id = torch.multinomial(probs, num_samples=1)
    else:
        # Greedy decoding: directly select the token with highest logit
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
    return next_id


# ------------------------------
# 3.2 Attention Implementation
# ------------------------------

def _attention_impl(q: Tensor, k: Tensor, v: Tensor, scale: float, mask: Tensor) -> Tensor:
    """Low-level implementation of attention mechanism.
    
    This is the core computation of Scaled Dot-Product Attention:
    
    1. Compute attention scores: scores = Q @ K^T * scale
       - Q @ K^T computes similarity between each query and all keys
       - scale prevents dot product from being too large
    
    2. Apply causal mask: scores[mask=False] = -inf
       - Set future position scores to negative infinity
       - After softmax, these positions have zero weight
    
    3. Compute attention weights: probs = softmax(scores)
       - Convert scores to probability distribution
       - Each row sums to 1
    
    4. Weighted sum: output = probs @ V
       - Weighted average of values using attention weights
    
    Args:
        q: Query tensor, shape (B, H, Lq, D)
        k: Key tensor, shape (B, H, Lk, D)
        v: Value tensor, shape (B, H, Lv, D)
        scale: Scale factor, typically 1/sqrt(D)
        mask: Causal mask, shape (1, 1, Lq, Lk)
    
    Returns:
        Attention output, shape (B, H, Lq, D)
    """
    # Step 1: Compute attention scores
    # Q @ K^T: (B, H, Lq, D) @ (B, H, D, Lk) -> (B, H, Lq, Lk)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Step 2: Apply causal mask
    # Set mask=False positions (future positions) to -inf
    scores = scores.masked_fill(~mask, float("-inf"))
    
    # Step 3: Softmax to get attention weights
    # -inf becomes 0 after softmax
    probs = torch.softmax(scores, dim=-1)
    
    # Step 4: Weighted sum
    # probs @ V: (B, H, Lq, Lk) @ (B, H, Lv, D) -> (B, H, Lq, D)
    return torch.matmul(probs, v)


def simple_causal_attention(query: Tensor, key: Tensor, value: Tensor, **kwargs) -> Tensor:
    """Simple causal attention implementation (Scaled Dot-Product Attention with Causal Mask).
    
    This is an educational attention implementation used to replace PyTorch's
    F.scaled_dot_product_attention. Through monkey-patching, we can observe
    the effect of custom attention implementation.
    
    Core attention formula:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Where:
    - Q (Query): Query vector, represents "what I'm looking for"
    - K (Key): Key vector, represents "what I have"
    - V (Value): Value vector, represents "what I want to return"
    - d_k: Dimension of key vector, used for scaling to prevent gradient vanishing
    
    Args:
        query:  Query tensor, shape (B, Hq, Lq, D)
                B=batch_size, Hq=num query heads, Lq=query seq length, D=head dim
        key:    Key tensor, shape (B, Hk, Lk, D)
                Hk=num key heads (may be smaller than Hq in GQA)
        value:  Value tensor, shape (B, Hv, Lv, D)
        **kwargs: Other parameters (for compatibility)
    
    Returns:
        Attention output, shape (B, Hq, Lq, D)
    """
    # Get head dimension D for computing scale factor
    Dh = query.shape[-1]
    # Scale factor = 1/sqrt(D), prevents dot product from being too large
    scale = 1.0 / (Dh**0.5)

    # ========== Grouped Query Attention (GQA) Handling ==========
    # GQA is an optimization technique: multiple query heads share the same key/value heads
    # This reduces KV Cache memory usage
    # Example: if query has 32 heads and key/value only has 8 heads
    #          then gqa_group_size = 32 / 8 = 4, every 4 query heads share 1 kv head
    gqa_group_size = query.shape[1] // key.shape[1]
    # Repeat key and value gqa_group_size times to match query head count
    key = key.repeat_interleave(gqa_group_size, dim=1)
    value = value.repeat_interleave(gqa_group_size, dim=1)

    # Convert to float32 for better numerical stability (avoid float16 precision issues)
    qf, kf, vf = query.float(), key.float(), value.float()

    # ========== Build Causal Mask ==========
    # Causal mask ensures each position can only see itself and previous positions, not the future
    # This is the core constraint of auto-regressive generation
    Tq, Tk = qf.shape[-2], kf.shape[-2]  # Get query and key sequence lengths
    # Create lower triangular matrix as mask:
    # [[1, 0, 0],
    #  [1, 1, 0],
    #  [1, 1, 1]]
    mask = torch.ones((Tq, Tk), device=qf.device, dtype=torch.bool).tril()
    # Expand dimensions to match attention scores shape (B, H, Lq, Lk)
    mask = mask[None, None, :, :]  # (1, 1, Lq, Lk)

    # Call the underlying attention implementation
    out = _attention_impl(qf, kf, vf, scale, mask)

    # Convert back to original dtype (usually float16)
    return out.to(dtype=query.dtype)


# Store original PyTorch attention for monkey-patching
_orig_sdp = F.scaled_dot_product_attention


def enable_custom_attention():
    """Enable custom scaled dot-product attention implementation (monkey-patch)."""
    F.scaled_dot_product_attention = simple_causal_attention


def disable_custom_attention():
    """Restore original PyTorch scaled dot-product attention."""
    F.scaled_dot_product_attention = _orig_sdp


# ------------------------------
# 3.3 TinyLLM Model Wrapper Class
# ------------------------------

class TinyLLM:
    """Lightweight LLM wrapper class.
    
    Encapsulates common HuggingFace model operations with a unified interface:
    - Model loading and initialization
    - Text generation (using HuggingFace's generate method)
    - Tokenization and detokenization
    - Raw forward pass (with/without KV Cache)
    
    This class is designed to help learners focus on understanding the core
    concepts of LLM inference, rather than being distracted by HuggingFace API details.
    """
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        """Initialize TinyLLM.
        
        Args:
            model_path: HuggingFace model ID or local path
                        e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        """
        self.model_path = model_path

        # Initialize tokenizer
        # Tokenizer converts text to token IDs and vice versa
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Initialize model
        # dtype=torch.float16: Use half-precision floats to reduce VRAM usage
        # device_map="auto": Automatically distribute model across available GPU/CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16, device_map="auto"
        )
        # Set to evaluation mode (disables dropout and other training-time randomness)
        self.model.eval()

    @torch.inference_mode()
    def generate_std(self, input_text: str, max_new_tokens: int, temperature: float = 0.0) -> str:
        """Generate text using HuggingFace's standard generate method.
        
        This is the simplest generation approach. HuggingFace internally implements:
        - Automatic KV Cache management
        - Various sampling strategies (greedy, temperature, top-k, top-p, etc.)
        - Stop condition detection (EOS token)
        
        Args:
            input_text: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0=greedy, >0=random sampling)
        
        Returns:
            Complete generated text (including input prompt)
        """
        # Encode text to token IDs and move to model device
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(
            self.model.device
        )
        # Ensure temperature is non-negative
        temperature = max(0.0, temperature)
        # Call HuggingFace's generate method
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,  # Enable sampling when temperature > 0
            temperature=temperature if temperature > 0.0 else None,
        )
        # Decode generated token IDs back to text
        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        return generated_text

    def tokenize(self, text: str) -> List[int]:
        """Convert text to a list of token IDs (tokenization).
        
        Tokenization is the first step in LLM text processing:
        - Text -> subword tokens -> token IDs
        
        Example:
        "DeepLearning.AI is" -> ["Deep", "Learning", ".AI", " is"] -> [33464, 47467, 88778, 374]
        
        Args:
            text: Input text
        
        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token ID list back to text (detokenization).
        
        This is the reverse of tokenization:
        - token IDs -> subword tokens -> text
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.inference_mode()
    def forward_raw(self, token_ids: List[int]) -> Tensor:
        """Raw forward pass (without KV Cache).
        
        This is the most basic forward pass approach:
        - Each call recomputes attention for all tokens
        - Computational complexity: O(n^2), where n is sequence length
        - Good for understanding principles, but inefficient
        
        In auto-regressive generation without KV Cache:
        - Generate token 1: compute attention for 1 token
        - Generate token 2: recompute attention for 2 tokens
        - Generate token n: recompute attention for n tokens
        - Total computation: 1 + 2 + 3 + ... + n = O(n^2)
        
        Args:
            token_ids: List of input token IDs
        
        Returns:
            Logits for next token, shape (1, vocab_size)
        """
        # Convert token ID list to tensor, add batch dimension
        input_ids = torch.tensor([token_ids], device=self.model.device)
        # Forward pass
        outputs: CausalLMOutputWithPast = self.model(input_ids=input_ids)
        # Extract logits for the last position (for predicting next token)
        return outputs.logits[:, -1, :]

    @torch.inference_mode()
    def forward_raw_with_kv_cache(
        self, token_ids: List[int], past_key_values: Optional[Cache] = None
    ) -> Tuple[Tensor, Cache]:
        """Forward pass with KV Cache.
        
        KV Cache is the core optimization technique for LLM inference:
        - Cache previously computed Key and Value tensors
        - New tokens only need to compute their own K, V, then concatenate with cache
        - Reduces computational complexity from O(n^2) to O(n)
        
        How it works:
        1. Prefill stage: Process all input tokens, build initial KV Cache
        2. Decode stage: Process only 1 new token at a time, reuse cached KV
        
        Memory usage:
        - KV Cache size = 2 * num_layers * num_heads * head_dim * seq_len * batch_size
        - For long sequences, KV Cache may consume significant VRAM
        
        Args:
            token_ids: List of input token IDs
                      - Prefill stage: all input tokens
                      - Decode stage: only the latest 1 token
            past_key_values: Previously cached KV pairs
                            - None: first call (Prefill)
                            - Cache object: subsequent calls (Decode)
        
        Returns:
            Tuple of:
            - next_token_logits: Logits for next token
            - past_key_values: Updated KV Cache
        """
        # Convert token IDs to tensor
        input_ids = torch.tensor([token_ids], device=self.model.device)
        # Forward pass with KV Cache enabled
        outputs: CausalLMOutputWithPast = self.model(
            input_ids=input_ids,
            use_cache=True,  # Enable KV Cache
            past_key_values=past_key_values  # Pass in previous cache
        )
        # Extract logits for next token
        next_token_logits = outputs.logits[:, -1, :]
        # Get updated KV Cache
        return next_token_logits, outputs.past_key_values


# ------------------------------
# 3.4 Auto-regressive Decoding Functions
# ------------------------------

def auto_regressive_decode(
    llm: TinyLLM, input_text: str, max_new_tokens: int, temperature: float = 0.0
) -> str:
    """Naive implementation of auto-regressive decoding (without KV Cache).
    
    Auto-regressive decoding is the core process of LLM text generation:
    1. Tokenize input text to get token sequence
    2. Loop to generate new tokens:
       a. Feed all current tokens into the model
       b. Get logits for the last position
       c. Sample next token from logits
       d. Append new token to sequence
    3. Detokenize final token sequence to get text
    
    Problem: Each new token generation recomputes attention for all tokens
    - Total computation for n tokens: O(n^2)
    - This is why KV Cache optimization is needed
    
    Args:
        llm: TinyLLM instance
        input_text: Input prompt text
        max_new_tokens: Number of new tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Complete generated text
    """
    # Step 1: Tokenize
    input_ids = llm.tokenize(input_text)

    # Step 2: Auto-regressive loop
    for _ in range(max_new_tokens):
        # Process full sequence each time (inefficient!)
        next_token_logits = llm.forward_raw(input_ids)
        # Sample next token
        next_token_id = sample(next_token_logits, temperature=temperature)
        # Append to sequence
        input_ids.append(next_token_id.item())

    # Step 3: Detokenize
    return llm.detokenize(input_ids)


def auto_regressive_decode_with_kv_cache(
    llm: TinyLLM, input_text: str, max_new_tokens: int, temperature: float = 0.0
) -> str:
    """Auto-regressive decoding with KV Cache optimization.
    
    This is the core optimization technique for LLM inference!
    
    Workflow:
    
    1. Prefill Stage:
       - Process all input tokens at once
       - Compute and cache K, V for all input tokens
       - Generate first new token
    
    2. Decode Stage:
       - Process only the latest 1 token each time
       - Compute new token's K, V, concatenate with cache
       - Use complete KV to compute attention
       - Generate next token
    
    Why KV Cache works:
    - Attention computation: Attention(Q, K, V) = softmax(Q @ K^T) @ V
    - For already generated tokens, their K and V don't change
    - Only new tokens need to compute new K and V
    - Cache old K, V to avoid redundant computation
    
    Args:
        llm: TinyLLM instance
        input_text: Input prompt text
        max_new_tokens: Number of new tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Complete generated text
    """
    # Step 1: Tokenize
    input_ids = llm.tokenize(input_text)

    # ========== Prefill Stage ==========
    # Process all input tokens at once, build initial KV Cache
    next_token_logits, past_key_values = llm.forward_raw_with_kv_cache(input_ids)
    next_token_id = sample(next_token_logits, temperature=temperature)
    input_ids.append(next_token_id.item())

    # ========== Decode Stage ==========
    # Process only the latest 1 token each time, reuse cached KV
    for _ in range(max_new_tokens - 1):
        # Note: Only pass the last token (input_ids[-1:])
        # past_key_values contains K, V for all previous tokens
        next_token_logits, past_key_values = llm.forward_raw_with_kv_cache(
            input_ids[-1:],  # Only pass the latest 1 token!
            past_key_values=past_key_values  # Reuse cache
        )
        next_token_id = sample(next_token_logits, temperature=temperature)
        input_ids.append(next_token_id.item())

    # Step 3: Detokenize
    return llm.detokenize(input_ids)


# ------------------------------
# 3.5 Demo/Utility Functions
# ------------------------------

def demo_tokenization(llm: TinyLLM, input_text: str) -> List[int]:
    """Demonstrate tokenization process."""
    token_ids = llm.tokenize(input_text)
    print("Tokenization Progress:")
    for token_id in token_ids:
        token = llm.tokenizer.convert_ids_to_tokens(token_id)
        token_display = token.replace("\u0120", " ")  # Ġ = BPE space prefix → readable space
        print(f"Token ID: {token_id:>6}: \"{token_display}\"")
    return token_ids


def demo_generation_comparison(
    llm: TinyLLM, input_text: str, max_new_tokens: int = 16
) -> None:
    """Compare different generation methods and their performance."""
    print("=" * 60)
    print("Generation Method Comparison")
    print("=" * 60)
    
    # 1. Standard generation (baseline)
    output_std = llm.generate_std(input_text, max_new_tokens, temperature=0.0)
    print(f'[Standard] "{output_std}"')
    print("-" * 60)
    
    # 2. Auto-regressive without KV Cache
    tic = time.time()
    output_no_cache = auto_regressive_decode(llm, input_text, max_new_tokens, temperature=0.0)
    time_no_cache = time.time() - tic
    print(f'[No KV Cache] "{output_no_cache}"')
    print(f"Time: {time_no_cache:.4f}s")
    print("-" * 60)
    
    # 3. Auto-regressive with KV Cache
    tic = time.time()
    output_with_cache = auto_regressive_decode_with_kv_cache(llm, input_text, max_new_tokens, temperature=0.0)
    time_with_cache = time.time() - tic
    print(f'[With KV Cache] "{output_with_cache}"')
    print(f"Time: {time_with_cache:.4f}s")
    print("-" * 60)
    
    # Summary
    speedup = time_no_cache / time_with_cache if time_with_cache > 0 else float('inf')
    print(f"Speedup: {speedup:.1f}x faster with KV Cache")
    print("=" * 60)

