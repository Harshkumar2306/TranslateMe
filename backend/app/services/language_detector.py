"""
Multilingual Language Detector with Script Detection and NLLB Code Mapping
Combines Unicode script heuristics with langdetect for high accuracy across Indic and global languages.
"""

import re
from typing import Optional, Tuple
import langdetect
from langdetect import DetectorFactory

# Enforce deterministic results from langdetect
DetectorFactory.seed = 0

from .language_codes import NLLB_LANGUAGE_CATALOG

# ISO 639-1 / 639-2 to NLLB code mapping
ISO_TO_NLLB: dict[str, str] = {
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
    "kn": "kan_Knda",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "sa": "san_Deva",
    "ne": "nep_Deva",
    "si": "sin_Sinh",
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "zh": "zho_Hans",
    "ar": "ara_Arab",
    "nl": "nld_Latn",
    "tr": "tur_Latn",
    "pl": "pol_Polx",
    "sv": "swe_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ms": "msa_Latn",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "uk": "ukr_Cyrl",
    "fa": "pes_Arab",
    "tl": "fil_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "no": "nor_Latn",
    "sw": "swa_Latn",
    "af": "afr_Latn",
    "ka": "kat_Geor",
    "hy": "hye_Armn",
    "my": "mya_Mymr",
    "km": "khm_Khmr",
    "lo": "lao_Laoo",
}

# Unicode Script ranges for high-confidence script identification
SCRIPT_RANGES = [
    (r'[\u0900-\u097F]', "hin_Deva", "Devanagari"),     # Hindi / Sanskrit / Marathi
    (r'[\u0B80-\u0BFF]', "tam_Taml", "Tamil"),          # Tamil
    (r'[\u0C00-\u0C7F]', "tel_Telu", "Telugu"),         # Telugu
    (r'[\u0D00-\u0D7F]', "mal_Mlym", "Malayalam"),      # Malayalam
    (r'[\u0C80-\u0CFF]', "kan_Knda", "Kannada"),        # Kannada
    (r'[\u0980-\u09FF]', "ben_Beng", "Bengali"),        # Bengali / Assamese
    (r'[\u0A80-\u0AFF]', "guj_Gujr", "Gujarati"),       # Gujarati
    (r'[\u0A00-\u0A7F]', "pan_Guru", "Gurmukhi"),       # Punjabi
    (r'[\u0B00-\u0B7F]', "ory_Orya", "Odia"),           # Odia
    (r'[\u0600-\u06FF]', "ara_Arab", "Arabic"),         # Arabic / Urdu / Persian
    (r'[\u3040-\u30FF\u31F0-\u31FF]', "jpn_Jpan", "Japanese"), # Hiragana / Katakana
    (r'[\uAC00-\uD7AF\u1100-\u11FF]', "kor_Hang", "Korean"),   # Hangul
    (r'[\u4E00-\u9FFF]', "zho_Hans", "Chinese"),        # CJK Unified Ideographs
    (r'[\u0400-\u04FF]', "rus_Cyrl", "Cyrillic"),       # Cyrillic
    (r'[\u0E00-\u0E7F]', "tha_Thai", "Thai"),           # Thai
    (r'[\u0590-\u05FF]', "heb_Hebr", "Hebrew"),         # Hebrew
    (r'[\u0370-\u03FF]', "ell_Grek", "Greek"),          # Greek
]


def detect_language(text: str) -> Tuple[Optional[str], Optional[str], float]:
    """
    Detects language and returns (nllb_code, language_name, confidence).
    """
    cleaned = text.strip()
    if not cleaned:
        return None, None, 0.0

    # 1. Unicode Script Heuristics for distinct scripts (Tamil, Telugu, Malayalam, etc.)
    for pattern, nllb_code, script_name in SCRIPT_RANGES:
        matches = len(re.findall(pattern, cleaned))
        if matches >= 3 or (matches >= 1 and len(cleaned) < 10):
            # If Devanagari, let langdetect refine between Hindi, Marathi, Nepali if possible
            if nllb_code == "hin_Deva":
                try:
                    iso = langdetect.detect(cleaned)
                    if iso in ("mr", "ne", "hi"):
                        mapped = ISO_TO_NLLB.get(iso, "hin_Deva")
                        return mapped, NLLB_LANGUAGE_CATALOG.get(mapped, {}).get("name", "Hindi"), 0.98
                except Exception:
                    pass
            # If Arabic script, refine between Arabic, Urdu, Persian
            elif nllb_code == "ara_Arab":
                try:
                    iso = langdetect.detect(cleaned)
                    if iso in ("ur", "fa", "ar"):
                        mapped = ISO_TO_NLLB.get(iso, "ara_Arab")
                        return mapped, NLLB_LANGUAGE_CATALOG.get(mapped, {}).get("name", "Arabic"), 0.95
                except Exception:
                    pass

            lang_meta = NLLB_LANGUAGE_CATALOG.get(nllb_code, {})
            return nllb_code, lang_meta.get("name", script_name), 0.99

    # 2. General statistical language detection using langdetect
    try:
        iso = langdetect.detect(cleaned)
        probabilities = langdetect.detect_langs(cleaned)
        conf = probabilities[0].prob if probabilities else 0.85

        if iso in ISO_TO_NLLB:
            nllb_code = ISO_TO_NLLB[iso]
            lang_name = NLLB_LANGUAGE_CATALOG.get(nllb_code, {}).get("name", iso)
            return nllb_code, lang_name, round(conf, 2)
    except Exception:
        pass

    # Default fallback to English if text is purely ASCII
    if all(ord(c) < 128 for c in cleaned):
        return "eng_Latn", "English", 0.80

    return None, None, 0.0
