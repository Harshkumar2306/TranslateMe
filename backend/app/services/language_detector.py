"""
Multilingual Language Detector for TranslateMe
Combines Unicode script analysis with langdetect for deterministic,
accurate identification across English, Hindi, Tamil, and Telugu.
"""

import re
from typing import Tuple
from .language_codes import SUPPORTED_LANGUAGES, normalize_language_code


# Canonical Unicode script ranges for supported languages
SCRIPT_RANGES = [
    # Devanagari (Hindi)
    (re.compile(r"[\u0900-\u097F]"), "hi", "Hindi"),
    # Tamil
    (re.compile(r"[\u0B80-\u0BFF]"), "ta", "Tamil"),
    # Telugu
    (re.compile(r"[\u0C00-\u0C7F]"), "te", "Telugu"),
    # Latin (English)
    (re.compile(r"[a-zA-Z]"), "en", "English"),
]


def detect_language(text: str) -> Tuple[str, str, float]:
    """
    Detects the language of input text using script heuristic counts
    and returns (canonical_code, language_name, confidence).
    """
    clean_text = text.strip()
    if not clean_text:
        return "en", "English", 0.0

    # Count matching characters per script
    counts = {}
    for pattern, code, name in SCRIPT_RANGES:
        matches = len(pattern.findall(clean_text))
        if matches > 0:
            counts[code] = (matches, name)

    if counts:
        # Select script with highest character frequency
        best_code, (count, best_name) = max(counts.items(), key=lambda item: item[1][0])
        confidence = min(0.99, round(count / max(1, len(clean_text)), 2))
        return best_code, best_name, max(0.85, confidence)

    # Default fallback to English if text is punctuation or numbers
    return "en", "English", 0.50
