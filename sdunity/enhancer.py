import os
import re
from typing import Iterable, List, Set

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    pipeline,
    set_seed,
)

# Default model name can be overridden with SDUNITY_GPT2_MODEL env var
_DEFAULT_MODEL = os.getenv("SDUNITY_GPT2_MODEL", "gpt2-medium")
_POSITIVE_WORDS = os.getenv("SDUNITY_GPT2_POSITIVE_WORDS")
_TOP_K = int(os.getenv("SDUNITY_GPT2_TOP_K", "100"))

_allowed_tokens: Set[int] | None = None
_logits_processor: LogitsProcessor | None = None
_tokenizer = None

_pipeline = None
_loaded_model = None


class _PositiveBiasLogitsProcessor(LogitsProcessor):
    """Bias generation towards a limited vocabulary."""

    def __init__(self, allowed_tokens: Iterable[int], eos_token_id: int, penalty: float = -10000.0):
        self.allowed_tokens: List[int] = list(set(allowed_tokens))
        self.eos_token_id = eos_token_id
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, self.penalty)
        mask[:, self.allowed_tokens] = 0.0
        mask[0, input_ids[0].long()] = self.penalty
        mask[0, self.eos_token_id] = 0.0
        return scores + mask


def _safe_str(x: str) -> str:
    """Collapse spaces and strip punctuation similar to Fooocus."""
    for _ in range(16):
        x = x.replace("  ", " ")
    return x.strip("., \r\n")


def _cleanup_prompt(text: str) -> str:
    text = re.sub(" +", " ", text)
    text = re.sub(",+", ",", text)
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    return ", ".join(tokens)

def _load(model_name: str | None = None):
    """Load the GPT-2 model for prompt enhancement."""
    global _pipeline, _loaded_model, _allowed_tokens, _logits_processor
    name = model_name or _DEFAULT_MODEL
    if _pipeline is None or _loaded_model != name:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name)
        _pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        global _tokenizer
        _tokenizer = tokenizer
        _loaded_model = name

        _allowed_tokens = None
        _logits_processor = None

        if _POSITIVE_WORDS and os.path.isfile(_POSITIVE_WORDS):
            words = [w.strip().lower() for w in open(_POSITIVE_WORDS, encoding="utf-8").read().splitlines() if w.strip()]
            tokens: Set[int] = set()
            for w in words:
                ids = tokenizer.encode(" " + w, add_special_tokens=False)
                tokens.update(ids)
            if tokens:
                _allowed_tokens = tokens
                _logits_processor = _PositiveBiasLogitsProcessor(tokens, tokenizer.eos_token_id)


def enhance(prompt: str, max_tokens: int = 50, seed: int | None = None) -> str:
    """Return a comma separated list of tags generated from the prompt."""
    if not prompt:
        return prompt

    _load()

    if seed is not None:
        set_seed(int(seed))

    cleaned = _safe_str(prompt)
    text = (
        "Take a short image prompt and expand it into a detailed list of descriptive tags.\n"
        "Only return a comma-separated list of tags in Danbooru style. Do not use full sentences.\n\n"
        f"{cleaned}\nTags:"
    )

    kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": True,
        "top_k": _TOP_K,
    }

    if _logits_processor is not None:
        kwargs["logits_processor"] = LogitsProcessorList([_logits_processor])

    result = _pipeline(text, **kwargs)
    generated = result[0]["generated_text"]
    if "Tags:" in generated:
        enhanced = generated.split("Tags:", 1)[-1]
    else:
        enhanced = generated

    enhanced = re.sub(r"(?i)tag[s]?:", "", enhanced)
    enhanced = enhanced.replace("\n", ",")
    items = [t.strip() for t in enhanced.split(",") if t.strip()]
    tags = []
    seen = set()
    for item in items:
        if item not in seen:
            tags.append(item)
            seen.add(item)

    final_text = ", ".join(tags) if tags else prompt
    final_text = _cleanup_prompt(final_text)

    if _tokenizer is not None:
        ids = _tokenizer.encode(final_text, add_special_tokens=False)
        if len(ids) > 75:
            ids = ids[:75]
            final_text = _tokenizer.decode(ids)
            final_text = _cleanup_prompt(final_text)

    return final_text
