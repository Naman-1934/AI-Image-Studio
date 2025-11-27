# safety.py

from typing import List, Tuple

# Expanded unsafe keywords for stricter safety filtering
UNSAFE_KEYWORDS = [
    "nude", "nudity", "naked", "genitals", "penis", "vagina", "sex", "sexual",
    "porn", "porno", "pornography", "erotic", "erotica", "fetish",
    "boobs", "breasts", "nipples", "ass", "butt",
    "kill", "murder", "blood", "gore", "violent", "abuse",
    "self-harm", "suicide", "child", "child abuse",
    "terrorist", "hate", "racist", "extremist"
]

DEFAULT_NEGATIVE = [
    "low quality", "blurry", "pixelated", "deformed",
    "disfigured", "duplicate", "watermark",
    "text", "logo", "nsfw"
]


def check_prompt_safety(prompt: str) -> Tuple[bool, str]:
    """
    Checks the prompt for unsafe or disallowed keywords.

    Returns:
        (allowed: bool, reason: str)
    """
    lower = prompt.lower()

    for word in UNSAFE_KEYWORDS:
        if word in lower:
            return False, f"The prompt contains prohibited content: '{word}'"

    return True, ""


def build_prompt(base_prompt: str, style: str) -> str:
    """
    Adds style-specific and quality-improving keywords for better output.
    """
    style = style.lower().strip()

    if style == "photorealistic":
        suffix = (
            "ultra realistic, 4k, high resolution, professional photography, "
            "highly detailed, sharp focus"
        )
    elif style == "artistic":
        suffix = (
            "digital art, concept art, highly detailed, artstation, cinematic lighting"
        )
    elif style == "cartoon":
        suffix = (
            "cartoon style, 2d illustration, bold lines, flat colors, clean outline"
        )
    else:
        suffix = "highly detailed, 4k, sharp focus"

    return f"{base_prompt}, {suffix}"


def build_negative_prompt(user_negative: str) -> str:
    """
    Appends user-provided negative prompt to default negative prompts.
    """
    parts: List[str] = DEFAULT_NEGATIVE.copy()
    if user_negative.strip():
        parts.append(user_negative.strip())
    return ", ".join(parts)
