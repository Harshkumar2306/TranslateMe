"""
Multilingual Neural Machine Translation Engine using NLLB-200 Transformer
Model: facebook/nllb-200-distilled-600M (Meta AI / Research)
"""

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
        num_beams: int = 2,
        return_alternatives: bool = False,
        max_length: int = 256
    ) -> dict:
        """
        Executes genuine Transformer translation between any supported language pair.
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

        # Configure tokenizer source language
        self.tokenizer.src_lang = source_lang
        inputs = self.tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.MAX_INPUT_LENGTH
        ).to(self.device)

        num_return = min(num_beams, 3) if return_alternatives and num_beams > 1 else 1

        # Real inference with precise latency tracking
        start_time = time.perf_counter()
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=target_token_id,
                num_beams=max(num_beams, num_return),
                num_return_sequences=num_return,
                max_length=max_length
            )
        elapsed_sec = time.perf_counter() - start_time

        # Decode output tokens
        decoded_outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        primary_translation = decoded_outputs[0].strip()
        alternatives = [alt.strip() for alt in decoded_outputs[1:] if alt.strip() != primary_translation]

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
