"""
Primary Multilingual Translation Service using Custom PyTorch Transformer
Trained strictly from scratch on parallel corpora with BPE subword tokenization.
Supports English (en), Hindi (hi), Tamil (ta), and Telugu (te).
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any, Tuple
import torch

from ml_lab.custom_transformer import Seq2SeqTransformer
from ml_lab.tokenizer import MultilingualSubwordTokenizer, is_supported_language, get_control_token
from backend.app.core.device import get_device

logger = logging.getLogger(__name__)

# Transparent mapping between legacy code aliases and canonical ISO-639 codes
CODE_MAP = {
    "eng_Latn": "en",
    "hin_Deva": "hi",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "en": "en",
    "hi": "hi",
    "ta": "ta",
    "te": "te"
}

REVERSE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu"
}


class CustomMultilingualEngine:
    def __init__(
        self,
        checkpoint_path: str = "ml_lab/checkpoints/custom_multilingual_transformer.pt",
        tokenizer_path: str = "ml_lab/tokenizer/multilingual_tokenizer.json"
    ):
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.model: Optional[Seq2SeqTransformer] = None
        self.tokenizer: Optional[MultilingualSubwordTokenizer] = None
        self.config: Dict[str, Any] = {}
        self.is_loaded = False
        self.device = get_device()

    def load_model(self) -> bool:
        """Loads trained weights and subword tokenizer into custom Seq2SeqTransformer."""
        if not os.path.exists(self.checkpoint_path):
            logger.warning(f"Checkpoint not found at: {self.checkpoint_path}")
            return False

        try:
            logger.info(f"Loading custom Multilingual Transformer from {self.checkpoint_path} onto {self.device}...")
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

            tok_path = ckpt.get("tokenizer_path", self.tokenizer_path)
            if not os.path.exists(tok_path):
                tok_path = self.tokenizer_path
            self.tokenizer = MultilingualSubwordTokenizer.load(tok_path)

            self.config = ckpt.get("config", {})
            self.model = Seq2SeqTransformer(
                src_vocab_size=self.config.get("vocab_size", self.tokenizer.vocab_size),
                tgt_vocab_size=self.config.get("vocab_size", self.tokenizer.vocab_size),
                d_model=self.config.get("d_model", 256),
                n_heads=self.config.get("n_heads", 4),
                num_encoder_layers=self.config.get("num_layers", 3),
                num_decoder_layers=self.config.get("num_layers", 3),
                d_ff=self.config.get("d_ff", 1024),
                dropout=self.config.get("dropout", 0.1),
                pad_idx=self.tokenizer.pad_id,
                share_embeddings=True
            ).to(self.device)

            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            self.is_loaded = True
            logger.info("Custom Multilingual Transformer loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load custom Multilingual Transformer: {e}", exc_info=True)
            self.is_loaded = False
            return False

    def normalize_code(self, code: str) -> str:
        """Normalizes lang code to 'en', 'hi', 'ta', 'te'."""
        clean = code.strip()
        if clean in CODE_MAP:
            return CODE_MAP[clean]
        return clean.lower()

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        num_beams: int = 2,
        return_alternatives: bool = False,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Translates text using the custom trained Transformer.
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            success = self.load_model()
            if not success or self.model is None or self.tokenizer is None:
                raise RuntimeError("Custom Multilingual Transformer could not be loaded. Please ensure model checkpoint exists.")

        src_code = self.normalize_code(source_lang)
        tgt_code = self.normalize_code(target_lang)

        # Validate language support
        if not is_supported_language(src_code):
            raise ValueError(f"Source language '{source_lang}' is not supported by your custom Transformer. Supported: English (en), Hindi (hi), Tamil (ta), Telugu (te).")
        if not is_supported_language(tgt_code):
            raise ValueError(f"Target language '{target_lang}' is not supported by your custom Transformer. Supported: English (en), Hindi (hi), Tamil (ta), Telugu (te).")

        start_time = time.time()
        text_clean = text.strip()
        if not text_clean:
            return {
                "original_text": text,
                "translated_text": "",
                "source_language": src_code,
                "target_language": tgt_code,
                "source_language_name": REVERSE_MAP.get(src_code, src_code),
                "target_language_name": REVERSE_MAP.get(tgt_code, tgt_code),
                "detected_language": None,
                "inference_time": 0.0,
                "model_name": "TranslateMe Custom Multilingual Transformer (From Scratch)",
                "alternatives": []
            }

        # Encode source sequence
        src_token_ids = self.tokenizer.encode(text_clean, add_special_tokens=True)
        src_tensor = torch.tensor([src_token_ids], dtype=torch.long, device=self.device)
        tgt_control_id = self.tokenizer.get_control_token_id(tgt_code)

        cross_attention_matrix = None
        alternatives = []

        with torch.no_grad():
            if num_beams > 1:
                beams = self.model.beam_search_decode(
                    src_tensor,
                    beam_size=num_beams,
                    max_len=64,
                    sos_idx=self.tokenizer.sos_id,
                    eos_idx=self.tokenizer.eos_id,
                    prefix_tokens=[tgt_control_id]
                )
                best_seq = beams[0][0][0].tolist()
                translated_text = self.tokenizer.decode(best_seq)

                if return_alternatives and len(beams) > 1:
                    for b_seq, _ in beams[1:]:
                        alt_text = self.tokenizer.decode(b_seq[0].tolist())
                        if alt_text and alt_text != translated_text and alt_text not in alternatives:
                            alternatives.append(alt_text)
            else:
                decode_res = self.model.greedy_decode(
                    src_tensor,
                    sos_idx=self.tokenizer.sos_id,
                    eos_idx=self.tokenizer.eos_id,
                    prefix_tokens=[tgt_control_id],
                    max_len=64,
                    return_attention=return_attention,
                    repetition_penalty=1.2
                )
                if return_attention:
                    out, attn = decode_res
                    if attn is not None:
                        # cross attention shape: (1, n_heads, tgt_len, src_len)
                        cross_attention_matrix = attn[0].mean(dim=0).cpu().tolist()
                else:
                    out = decode_res
                translated_text = self.tokenizer.decode(out[0].tolist())

        elapsed = round(time.time() - start_time, 4)

        result = {
            "original_text": text,
            "translated_text": translated_text,
            "source_language": src_code,
            "target_language": tgt_code,
            "source_language_name": REVERSE_MAP.get(src_code, src_code),
            "target_language_name": REVERSE_MAP.get(tgt_code, tgt_code),
            "detected_language": None,
            "inference_time": elapsed,
            "model_name": "TranslateMe Custom Multilingual Transformer (From Scratch)",
            "alternatives": alternatives
        }

        if return_attention:
            result["cross_attention_matrix"] = cross_attention_matrix
            result["source_subwords"] = [self.tokenizer.tokenizer.id_to_token(i) for i in src_token_ids]
            if 'out' in locals():
                result["target_subwords"] = [self.tokenizer.tokenizer.id_to_token(i) for i in out[0].tolist()]

        return result


custom_multilingual_engine = CustomMultilingualEngine()
