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
        _pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # Slight sampling to avoid repetitive outputs
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
        )
        _loaded_model = name


def enhance(prompt: str, max_tokens: int = 50) -> str:
    """Return a comma separated list of tags generated from the prompt."""
    if not prompt:
        return prompt
    _load()
    text = (
        "Take a short image prompt and expand it into a detailed list of descriptive tags.\n"
        "Only return a comma-separated list of tags in Danbooru style. Do not use full sentences.\n\n"
        f"{prompt}\nTags:"
    )
    result = _pipeline(text, max_new_tokens=max_tokens)
    generated = result[0]["generated_text"]
    if "Tags:" in generated:
        enhanced = generated.split("Tags:", 1)[-1].strip()
    else:
        enhanced = generated.strip()
    seen = set()
    tags = []
    for t in enhanced.split(","):
        tok = t.strip()
        if tok and tok not in seen:
            tags.append(tok)
            seen.add(tok)
    return ", ".join(tags) if tags else prompt
