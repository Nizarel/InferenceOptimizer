from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Cache
from transformers.modeling_outputs import CausalLMOutputWithPast


MODEL_CACHE = Path(__file__).resolve().parent.parent / ".hf-cache"
os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
os.environ.setdefault("HF_XET_CACHE", str(MODEL_CACHE / "xet"))


class TinyLLM:
    """Memory-optimized wrapper for causal language models.
    
    This class provides:
    - Efficient model loading with automatic dtype selection
    - Memory-aware tokenization and generation
    - Support for both standard generation and manual forward passes
    - KV cache support for optimized inference
    
    Memory optimizations implemented:
    - Uses bfloat16 weights (50% memory reduction vs float32)
    - Explicit placement of the entire model on one CUDA device
    - inference_mode() context for all forward passes (disables autograd)
    - Explicit memory cleanup in critical paths
    """
    
    def __init__(
        self,
        model_path: str,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize the TinyLLM with a pre-trained model.
        
        Args:
            model_path: HuggingFace model ID or local path
            device: CUDA device used for the entire model
        
        Memory notes:
        - bfloat16 reduces memory by ~50% compared to float32
        - Explicit device mapping prevents silent CPU offload
        - eval() mode disables dropout and other training-only layers
        """
        if not torch.cuda.is_available():
            raise RuntimeError("This lesson requires a CUDA-capable GPU.")

        target_device = torch.device(device)
        if target_device.type != "cuda":
            raise ValueError("TinyLLM only supports CUDA devices.")
        if target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())

        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        self.model_path: str = model_path

        # Initialize tokenizer (lightweight, negligible memory)
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=MODEL_CACHE,
        )

        # Load model with memory optimizations:
        # 1. dtype=torch.bfloat16 - Uses 16-bit weights (2 bytes vs 4 bytes per param)
        #    For 1.5B model: ~3GB instead of ~6GB
        # 2. device_map - Keeps every layer on the selected GPU
        # 3. low_cpu_mem_usage=True - Loads model in parts to reduce peak RAM
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=MODEL_CACHE,
            dtype=torch.bfloat16,
            device_map={"": str(target_device)},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        
        # Set to evaluation mode
        # This disables dropout, batch norm updates, etc.
        # Saves memory and ensures deterministic behavior
        self.model.eval()
        
        # Store device for convenience
        self.device: torch.device = next(self.model.parameters()).device
        parameter_devices = {parameter.device.type for parameter in self.model.parameters()}
        if parameter_devices != {"cuda"}:
            raise RuntimeError(
                f"Expected all model parameters on CUDA, found {parameter_devices}."
            )
        
        print(f"Model loaded on {self.device}")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    @torch.inference_mode()  # More efficient than no_grad() - disables view tracking
    def generate_std(
        self,
        input_text: str,
        max_new_tokens: int,
        temperature: float
    ) -> str:
        """Generate text using HuggingFace's optimized generate method.
        
        Args:
            input_text: The input prompt text
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0=greedy, >0=sampling)
            
        Returns:
            The generated text including the input prompt
            
        Memory notes:
        - Uses HuggingFace's built-in KV cache automatically
        - inference_mode() prevents gradient tracking (saves memory)
        - We explicitly move input to the correct device to avoid copies
        """
        # Tokenize and move to model's device in one step
        # return_tensors="pt" creates PyTorch tensors
        # .to(self.device) moves inputs to the GPU without unnecessary copies
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
        ).to(self.device)

        # Generate with memory-efficient settings
        output_ids: torch.Tensor = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,  # Only sample if temperature > 0
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache (critical for speed and memory)
        )

        # Decode output
        # skip_special_tokens removes <eos>, <pad>, etc.
        output_text: str = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        
        return output_text

    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
            
        Memory notes:
        - Returns Python list instead of tensor to save memory
        - Only convert to tensor when actually needed for model input
        """
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.inference_mode()
    def forward_raw(self, input_ids: List[int]) -> torch.Tensor:
        """Raw forward pass through the model without KV cache.
        
        Args:
            input_ids: List of token IDs
            
        Returns:
            Logits for the next token (shape: [1, vocab_size])
            
        Memory notes:
        - Creates temporary tensor, gets logits, then cleans up
        - Only returns the logits we need (last position)
        - Uses inference_mode() to prevent gradient tracking
        """
        # Convert list to tensor efficiently
        # unsqueeze(0) adds batch dimension: [seq_len] -> [1, seq_len]
        input_tensor: torch.Tensor = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        # Forward pass
        # We only need logits, not past_key_values or other outputs
        outputs: CausalLMOutputWithPast = self.model(
            input_tensor,
            use_cache=False,
        )
        
        # Extract logits for the last position only
        # Shape: [1, seq_len, vocab_size] -> [1, vocab_size]
        # This is more memory-efficient than keeping all positions
        next_token_logits: torch.Tensor = outputs.logits[:, -1, :]
        
        # Cleanup
        del input_tensor, outputs
        
        return next_token_logits

    @torch.inference_mode()
    def forward_raw_with_kv_cache(
        self,
        input_ids: List[int],
        past_key_values: Optional[Cache] = None
    ) -> Tuple[torch.Tensor, Cache]:
        """Forward pass with KV cache support for efficient generation.
        
        Args:
            input_ids: List of token IDs (only new tokens if using cache)
            past_key_values: Cached key-value pairs from previous forward passes
            
        Returns:
            Tuple of (logits for next token, updated cache)
            
        Memory notes:
        - KV cache stores attention states, avoiding recomputation
        - Cache grows linearly with sequence length
        - For long sequences, cache can be larger than model weights!
        - The cache follows the model's bfloat16 dtype to reduce memory by 50%
        """
        # Convert to tensor
        input_tensor: torch.Tensor = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        # Forward pass with cache
        # past_key_values contains cached attention from previous tokens
        # This avoids recomputing attention for tokens we've already processed
        outputs: CausalLMOutputWithPast = self.model(
            input_tensor,
            past_key_values=past_key_values,
            use_cache=True  # Critical: tells model to return updated cache
        )
        
        # Extract results
        next_token_logits: torch.Tensor = outputs.logits[:, -1, :]
        new_cache = outputs.past_key_values
        if new_cache is None:
            raise RuntimeError("The model did not return a KV cache.")
        
        # Cleanup input tensor (cache is returned, so keep it)
        del input_tensor, outputs
        
        return next_token_logits, new_cache
