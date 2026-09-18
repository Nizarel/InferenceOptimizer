from __future__ import annotations

import torch
from torch import Tensor
from tiny_llm import TinyLLM


def sample(logits: Tensor, temperature: float = 0.0) -> Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def auto_regressive_decode(
    tiny_llm: TinyLLM,
    text: str,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> str:
    token_ids = tiny_llm.tokenize(text)
    for _ in range(max_new_tokens):
        logits = tiny_llm.forward_raw(token_ids)
        next_token = sample(logits, temperature)
        token_ids.append(next_token.item())
    return tiny_llm.detokenize(token_ids)


def auto_regressive_decode_with_kv_cache(
    tiny_llm: TinyLLM,
    text: str,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> str:
    token_ids = tiny_llm.tokenize(text)
    if max_new_tokens <= 0:
        return tiny_llm.detokenize(token_ids)

    logits, past_key_values = tiny_llm.forward_raw_with_kv_cache(token_ids)
    next_token = sample(logits, temperature)
    token_ids.append(next_token.item())

    for _ in range(max_new_tokens - 1):
        logits, past_key_values = tiny_llm.forward_raw_with_kv_cache(
            [next_token.item()],
            past_key_values,
        )
        next_token = sample(logits, temperature)
        token_ids.append(next_token.item())

    return tiny_llm.detokenize(token_ids)


def demo_tokenization(tiny_llm: TinyLLM, text: str) -> list[int]:
    token_ids = tiny_llm.tokenizer.encode(text, add_special_tokens=True)
    tokens = tiny_llm.tokenizer.convert_ids_to_tokens(token_ids)
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)
    return token_ids