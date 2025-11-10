"""Text normalization helpers."""

from __future__ import annotations

import re
from typing import Optional

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_text(
    text: Optional[str],
    *,
    min_length: int = 10,
    language: Optional[str] = None,
    collapse_whitespace: bool = True,
) -> Optional[str]:
    """Normalize ``text`` and return a cleaned string or ``None``.

    Parameters
    ----------
    text:
        The input text to normalize.
    min_length:
        Minimum length of the normalized text. Shorter samples are discarded.
    language:
        Optional language hint for future extensions. Currently unused but kept
        for API compatibility.
    """

    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\r", "\n")
    text = CONTROL_RE.sub(" ", text)
    if collapse_whitespace:
        text = WHITESPACE_RE.sub(" ", text)
        text = MULTI_NEWLINE_RE.sub("\n\n", text)
        text = re.sub(r" *\n *", "\n", text)
    text = text.strip()
    if not text or len(text) < min_length:
        return None
    # Ensure round-trip through UTF-8 to drop invalid bytes.
    try:
        text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    except Exception:
        return None
    return text
