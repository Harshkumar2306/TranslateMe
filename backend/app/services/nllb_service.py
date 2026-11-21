"""
Multilingual Neural Machine Translation Engine using NLLB-200 Transformer
Model: facebook/nllb-200-distilled-600M (Meta AI / Research)
Includes Intelligent Truecasing, Proper Name Recognition, and Anti-Hallucination Decoding.
"""

import re
import time
import logging
from typing import Optional, List, Tuple
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from backend.app.core.device import get_device
from backend.app.core.config import settings
from .language_codes import is_valid_language, get_language_name
from .language_detector import detect_language

logger = logging.getLogger(__name__)


# Common question prefixes for punctuation inference
QUESTION_STARTERS = {
    'what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom',
    'is', 'are', 'am', 'was', 'were', 'can', 'could', 'would', 'should',
    'do', 'does', 'did', 'will', 'shall', 'have', 'has', 'had',
    'kya', 'kaise', 'kahan', 'kab', 'kyun', 'kaun', 'kisne', 'kisko'
}

# Common places & proper nouns to preserve casing
COMMON_PROPER_NOUNS = [
    'india', 'delhi', 'new delhi', 'mumbai', 'bangalore', 'bengaluru', 'chennai',
    'kolkata', 'hyderabad', 'pune', 'ahmedabad', 'jaipur', 'lucknow', 'chandigarh',
    'america', 'london', 'paris', 'tokyo', 'germany', 'france', 'japan', 'china'
]


def preprocess_translation_input(text: str, src_lang: str) -> Tuple[str, bool, bool]:
    """
    Intelligently cleans, truecases, and formats input before tokenization.
    Resolves lowercase proper names (e.g., 'harsh kumar' -> 'Harsh Kumar'),
    capitalizes pronouns ('i' -> 'I'), and infers sentence boundary punctuation
    to prevent the Transformer from hallucinating open-ended dialogue continuations.
    
    Returns:
        (preprocessed_text, had_trailing_punctuation, was_question)
    """
    clean = text.strip()
    if not clean:
        return clean, False, False

    had_trailing_punct = clean[-1] in ('.', '!', '?', ':', ';', '।', '؟', '۔')
    first_word = clean.split()[0].lower() if clean.split() else ''
    is_question = first_word in QUESTION_STARTERS or clean.endswith('?')

    if src_lang in ('eng_Latn', 'fra_Latn', 'spa_Latn', 'deu_Latn', 'ita_Latn', 'por_Latn'):
        # 1. Capitalize standalone pronoun 'i', 'i\'m', 'i\'ve', 'i\'d', 'i\'ll'
        clean = re.sub(r'\bi\b', 'I', clean)
        clean = re.sub(r'\bi\'m\b', "I'm", clean, flags=re.IGNORECASE)
        clean = re.sub(r'\bi\'ve\b', "I've", clean, flags=re.IGNORECASE)
        clean = re.sub(r'\bi\'d\b', "I'd", clean, flags=re.IGNORECASE)
        clean = re.sub(r'\bi\'ll\b', "I'll", clean, flags=re.IGNORECASE)

        # 2. Capitalize the first letter of sentences
        clean = clean[0].upper() + clean[1:] if len(clean) > 1 else clean.upper()

        # 3. Proper Name Casing after introductions:
        # e.g., 'I am harsh kumar' -> 'I am Harsh Kumar'
        #       'My name is harsh' -> 'My name is Harsh'
        #       'This is harsh kumar' -> 'This is Harsh Kumar'
        def titlecase_names(match):
            intro = match.group(1)
            name_phrase = match.group(2)
            punct = match.group(3)
            # Capitalize each word in name phrase
            capitalized_name = ' '.join(w.capitalize() for w in name_phrase.split())
            return f"{intro} {capitalized_name}{punct}"

        clean = re.sub(
            r'^(I am|My name is|This is|Name is|Call me)\s+([a-zA-Z\s]+?)([.?!]?)$',
            titlecase_names,
            clean,
            flags=re.IGNORECASE
        )

        # If it's just two words and no punctuation (e.g. 'harsh kumar'), title-case it as a name
        words = clean.split()
        if len(words) in (1, 2, 3) and not had_trailing_punct and all(w.isalpha() for w in words):
            # Check if likely a personal name
            if words[0].lower() not in QUESTION_STARTERS and words[0].lower() not in ('the', 'a', 'an', 'this', 'that', 'it', 'they', 'we', 'you'):
                clean = ' '.join(w.capitalize() for w in words)

        # 4. Standardize known proper nouns & places
        for place in COMMON_PROPER_NOUNS:
            clean = re.sub(rf'\b{place}\b', place.title(), clean, flags=re.IGNORECASE)

        # 5. Sentence boundary closure:
        # Transformers trained on sentence-aligned corpora require closed clause boundaries.
        # Open clauses trigger continuation hallucinations.
        if not had_trailing_punct:
            clean = clean + ('?' if is_question else '.')

    return clean, had_trailing_punct, is_question


def postprocess_translation_output(output: str, had_trailing_punct: bool, is_question: bool) -> str:
    """
    Cleans generated translation tokens:
    - Strips unprompted quotation marks and stray novelistic tags.
    - Matches ending punctuation appropriately.
    """
    res = output.strip()

    # Remove dialogue quote wraps: e.g., "..." or '...'
    res = re.sub(r'^[\"\'«“]+|[\"\'»”]+$', '', res).strip()

    # Strip novelistic speaker attribution hallucinations if any (e.g., '... ने कहा, "..."')
    if 'ने कहा,' in res or 'ने कहा कि' in res:
        parts = re.split(r'ने कहा[,\s]+', res)
        if len(parts) > 1 and len(parts[1].strip()) > 0:
            res = parts[1].strip().strip('\"\'«“»”')

    # Remove trailing punctuation if original had no punctuation and it's a short 1-3 word phrase
    if not had_trailing_punct and len(res.split()) <= 3 and res.endswith(('।', '.', '!', '?')):
        res = res[:-1].strip()

    return res.strip()


class NLLBTranslationEngine:
    _instance: Optional["NLLBTranslationEngine"] = None

    def __init__(self):
        self.model_name = settings.MODEL_NAME
        self.device = get_device()
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    @classmethod
    def get_instance(cls) -> "NLLBTranslationEngine":
        if cls._instance is None:
            cls._instance = NLLBTranslationEngine()
        return cls._instance

    def load_model(self):
        """Loads tokenizer and model weights once into memory."""
        if self.is_loaded:
            return

        logger.info(f"Loading NLLB-200 model ({self.model_name}) onto {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info("NLLB-200 Transformer successfully loaded and ready for inference!")
        except Exception as e:
            logger.error(f"Failed to load NLLB model: {e}")
            raise e

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        num_beams: int = 4,
        return_alternatives: bool = False,
        max_length: int = 256
    ) -> dict:
        """
        Executes genuine Transformer translation with intelligent Truecasing,
        anti-hallucination penalties, and beam search decoding.
        """
        if not self.is_loaded:
            self.load_model()

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Input text cannot be empty.")

        detected_lang_code = None
        detected_lang_name = None

        # Handle Auto Detection
        if source_lang.lower() == "auto":
            code, name, _ = detect_language(clean_text)
            if not code or not is_valid_language(code):
                raise ValueError("Could not automatically identify the source language with high confidence. Please select it manually.")
            source_lang = code
            detected_lang_code = code
            detected_lang_name = name

        # Validate language codes
        if not is_valid_language(source_lang):
            raise ValueError(f"Unsupported source language code: '{source_lang}'")
        if not is_valid_language(target_lang):
            raise ValueError(f"Unsupported target language code: '{target_lang}'")

        # Get Target forced BOS token ID
        target_token_id = self.tokenizer.convert_tokens_to_ids(target_lang)
        if target_token_id is None or target_token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Target language code '{target_lang}' token not found in model vocabulary.")

        # Preprocess input (Truecasing, Name Entity capitalization, Sentence boundary closure)
        preprocessed_text, had_punct, is_question = preprocess_translation_input(clean_text, source_lang)

        # Configure tokenizer source language
        self.tokenizer.src_lang = source_lang
        inputs = self.tokenizer(
            preprocessed_text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.MAX_INPUT_LENGTH
        ).to(self.device)

        num_return = min(num_beams, 3) if return_alternatives and num_beams > 1 else 1

        # Real inference with optimized generation hyperparameters:
        # - length_penalty=0.8: prefers concise, faithful translations over run-on hallucinations
        # - repetition_penalty=1.15: penalizes loops
        # - no_repeat_ngram_size=3: prevents 3-gram repetitions
        # - early_stopping=True: terminates promptly at EOS
        start_time = time.perf_counter()
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=target_token_id,
                num_beams=max(num_beams, num_return),
                num_return_sequences=num_return,
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
                length_penalty=0.8,
                early_stopping=True,
                max_length=max_length
            )
        elapsed_sec = time.perf_counter() - start_time

        # Decode output tokens
        decoded_outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        primary_translation = postprocess_translation_output(decoded_outputs[0], had_punct, is_question)
        alternatives = [
            postprocess_translation_output(alt, had_punct, is_question)
            for alt in decoded_outputs[1:]
            if postprocess_translation_output(alt, had_punct, is_question) != primary_translation
        ]

        return {
            "original_text": clean_text,
            "translated_text": primary_translation,
            "source_language": source_lang,
            "source_language_name": get_language_name(source_lang),
            "target_language": target_lang,
            "target_language_name": get_language_name(target_lang),
            "detected_language": detected_lang_code,
            "inference_time": round(elapsed_sec, 3),
            "model_name": "NLLB-200 (facebook/nllb-200-distilled-600M)",
            "alternatives": alternatives
        }


translation_engine = NLLBTranslationEngine.get_instance()
