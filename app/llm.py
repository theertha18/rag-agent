"""
The only file that knows which LLM provider is being used.

main.py calls generate(prompt) and gets text back. To switch providers, set
the LLM_PROVIDER environment variable ("gemini" or "anthropic") -- no other
code changes needed. Later, the agent loop and tool calling will also go
through this layer.

Environment variables:
  LLM_PROVIDER      gemini (default) | anthropic
  GEMINI_API_KEY    needed when provider is gemini
  GEMINI_MODEL      optional, defaults below -- check AI Studio for a model
                    that is currently available on your free tier
  ANTHROPIC_API_KEY needed when provider is anthropic
  ANTHROPIC_MODEL   optional, defaults below
"""

import os

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


def generate(prompt: str, max_tokens: int = 500) -> str:
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini":
        return _generate_gemini(prompt)
    if provider == "anthropic":
        return _generate_anthropic(prompt, max_tokens)

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Use 'gemini' or 'anthropic'."
    )


def _generate_gemini(prompt: str) -> str:
    # Imported inside the function so you only need the package for the
    # provider you actually use.
    from google import genai

    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or ""


def _generate_anthropic(prompt: str, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )
