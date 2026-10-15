"""MiddleMatch helpers: the model must name the person between two anchors inside \\boxed{};
scored by case-insensitive exact match."""

import re


def extract_boxed_text(text: str) -> str:
    """Return the content of the last \\boxed{...}; fall back to "the answer/name is X", else the text."""
    if not text:
        return ""

    pattern = r'\\boxed{((?:[^{}]|{[^{}]*})*)}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    fallback_pattern = r'(?i)the (?:answer|name) is\s*([^\n\.]*)'
    fallback_matches = re.findall(fallback_pattern, text)
    if fallback_matches:
        return fallback_matches[-1].strip()

    return text


def exact_match(prediction, ground_truth):
    return str(prediction).strip().lower() == str(ground_truth).strip().lower()
