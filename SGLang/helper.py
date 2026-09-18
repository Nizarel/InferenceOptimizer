from __future__ import annotations

import os
from pathlib import Path


MODEL_CACHE = Path(__file__).resolve().parent.parent / ".hf-cache"
os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
os.environ.setdefault("HF_XET_CACHE", str(MODEL_CACHE / "xet"))

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


class TinyLLM:
    def __init__(self, model_path: str):
        if not torch.cuda.is_available():
            raise RuntimeError("This lesson requires a CUDA-capable GPU.")

        self.device = torch.device("cuda")
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=MODEL_CACHE,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=MODEL_CACHE,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()

    def generate_std(
        self,
        text: str,
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        return _generate(self, text, max_new_tokens, temperature, use_cache=True)


def sample(logits: Tensor, temperature: float = 0.0) -> Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)


@torch.inference_mode()
def _generate(
    tiny_llm: TinyLLM,
    text: str,
    max_new_tokens: int,
    temperature: float,
    *,
    use_cache: bool,
) -> str:
    inputs = tiny_llm.tokenizer(text, return_tensors="pt").to(tiny_llm.device)
    generated = tiny_llm.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        use_cache=use_cache,
        pad_token_id=tiny_llm.tokenizer.eos_token_id,
    )
    return tiny_llm.tokenizer.decode(generated[0], skip_special_tokens=True)


def auto_regressive_decode(
    tiny_llm: TinyLLM,
    text: str,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> str:
    inputs = tiny_llm.tokenizer(text, return_tensors="pt").to(tiny_llm.device)
    token_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = tiny_llm.model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            next_token = sample(outputs.logits[:, -1, :], temperature)
            token_ids = torch.cat((token_ids, next_token[:, None]), dim=-1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(next_token[:, None])),
                dim=-1,
            )

    return tiny_llm.tokenizer.decode(token_ids[0], skip_special_tokens=True)


def auto_regressive_decode_with_kv_cache(
    tiny_llm: TinyLLM,
    text: str,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> str:
    inputs = tiny_llm.tokenizer(text, return_tensors="pt").to(tiny_llm.device)
    token_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    with torch.inference_mode():
        outputs = tiny_llm.model(**inputs, use_cache=True)
        next_token = sample(outputs.logits[:, -1, :], temperature)
        token_ids = torch.cat((token_ids, next_token[:, None]), dim=-1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(next_token[:, None])),
            dim=-1,
        )
        past_key_values = outputs.past_key_values

        for _ in range(max_new_tokens - 1):
            outputs = tiny_llm.model(
                input_ids=next_token[:, None],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = sample(outputs.logits[:, -1, :], temperature)
            token_ids = torch.cat((token_ids, next_token[:, None]), dim=-1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(next_token[:, None])),
                dim=-1,
            )

    return tiny_llm.tokenizer.decode(token_ids[0], skip_special_tokens=True)


def demo_tokenization(tiny_llm: TinyLLM, text: str) -> list[int]:
    token_ids = tiny_llm.tokenizer.encode(text, add_special_tokens=True)
    tokens = tiny_llm.tokenizer.convert_ids_to_tokens(token_ids)
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)
    return token_ids