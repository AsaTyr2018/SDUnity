import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Default model name can be overridden with SDUNITY_GPT2_MODEL env var
_DEFAULT_MODEL = os.getenv("SDUNITY_GPT2_MODEL", "gpt2-medium")

_pipeline = None
_loaded_model = None

def _load(model_name: str | None = None):
    """Load the GPT-2 model for prompt enhancement."""
    global _pipeline, _loaded_model
    name = model_name or _DEFAULT_MODEL
    if _pipeline is None or _loaded_model != name:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name)
        _pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        _loaded_model = name


def enhance(prompt: str, max_tokens: int = 50) -> str:
    """Return an enhanced version of the given prompt using GPT-2."""
    if not prompt:
        return prompt
    _load()
    text = f"Improve this Stable Diffusion prompt: {prompt}\nEnhanced:"
    result = _pipeline(text, max_new_tokens=max_tokens, do_sample=False)
    generated = result[0]["generated_text"]
    if "Enhanced:" in generated:
        enhanced = generated.split("Enhanced:", 1)[-1].strip()
    else:
        enhanced = generated.strip()
    return enhanced or prompt
