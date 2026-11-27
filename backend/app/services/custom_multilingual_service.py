"""
Primary Multilingual Translation Service using Custom PyTorch Transformer
Trained strictly from scratch on parallel corpora with BPE subword tokenization.
Supports English (en), Hindi (hi), Tamil (ta), and Telugu (te).
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any, Union
import torch

from ml_lab.custom_transformer import Seq2SeqTransformer
from ml_lab.custom_transformer.modern_transformer import ModernSeq2SeqTransformer, ModernTransformerConfig
from ml_lab.tokenizer import MultilingualSubwordTokenizer, is_supported_language
from backend.app.core.config import settings
from backend.app.core.device import get_device
from backend.app.services.language_detector import detect_language

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
    "te": "te",
}

REVERSE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
}


class CustomMultilingualEngine:
    def __init__(
        self,
        checkpoint_path: str = settings.CHECKPOINT_PATH,
        tokenizer_path: str = settings.TOKENIZER_PATH,
    ):
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.model: Optional[Union[Seq2SeqTransformer, ModernSeq2SeqTransformer]] = None
        self.tokenizer: Optional[MultilingualSubwordTokenizer] = None
        self.config: Dict[str, Any] = {}
        self.checkpoint_meta: Dict[str, Any] = {}
        self.is_loaded = False
        self.device = get_device()
        self._script_masks: Dict[str, torch.Tensor] = {}

    def load_model(self) -> bool:
        """Loads trained weights and subword tokenizer into custom Seq2SeqTransformer."""
        if not os.path.exists(self.checkpoint_path):
            logger.warning(f"Checkpoint not found at: {self.checkpoint_path}")
            return False

        try:
            logger.info(f"Loading custom Multilingual Transformer from {self.checkpoint_path} onto {self.device}...")
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

            tok_path = ckpt.get("tokenizer_path", self.tokenizer_path)
            if not tok_path or not os.path.exists(tok_path):
                tok_path = self.tokenizer_path
            self.tokenizer = MultilingualSubwordTokenizer.load(tok_path)

            self.config = dict(ckpt.get("config", {}))
            state_dict = ckpt.get("model_state_dict", {})
            n_enc = self.config.get("num_encoder_layers", self.config.get("num_layers", 6))
            n_dec = self.config.get("num_decoder_layers", self.config.get("num_layers", 6))

            is_modern = (
                self.config.get("use_rope", False)
                or any("rope" in k for k in state_dict.keys())
                or self.config.get("model_type") == "modern"
            )
            if is_modern:
                modern_cfg = ModernTransformerConfig(
                    vocab_size=self.config.get("vocab_size", self.tokenizer.vocab_size),
                    d_model=self.config.get("d_model", 768),
                    n_heads=self.config.get("n_heads", 12),
                    num_encoder_layers=n_enc,
                    num_decoder_layers=n_dec,
                    d_ff=self.config.get("d_ff", 2048),
                    max_len=self.config.get("max_len", 1024),
                    dropout=self.config.get("dropout", 0.1),
                    pad_idx=self.tokenizer.pad_id,
                    bos_idx=getattr(self.tokenizer, "bos_id", self.tokenizer.sos_id),
                    eos_idx=self.tokenizer.eos_id,
                    use_rope=self.config.get("use_rope", True),
                    use_qk_norm=self.config.get("use_qk_norm", True),
                    share_embeddings=self.config.get("share_embeddings", True),
                    use_layerscale=self.config.get("use_layerscale", True),
                    layerscale_init=self.config.get("layerscale_init", 0.1),
                    soft_cap=self.config.get("soft_cap", 30.0),
                )
                self.model = ModernSeq2SeqTransformer(modern_cfg).to(self.device)
                self.model.load_state_dict(state_dict)
                if modern_cfg.share_embeddings:
                    self.model.lm_head.weight = self.model.shared_embedding.weight
            else:
                self.model = Seq2SeqTransformer(
                    src_vocab_size=self.config.get("src_vocab_size", self.config.get("vocab_size", self.tokenizer.vocab_size)),
                    tgt_vocab_size=self.config.get("tgt_vocab_size", self.config.get("vocab_size", self.tokenizer.vocab_size)),
                    d_model=self.config.get("d_model", 384),
                    n_heads=self.config.get("n_heads", 6),
                    num_encoder_layers=n_enc,
                    num_decoder_layers=n_dec,
                    d_ff=self.config.get("d_ff", 1536),
                    dropout=self.config.get("dropout", 0.1),
                    pad_idx=self.tokenizer.pad_id,
                    share_embeddings=self.config.get("share_embeddings", True),
                    pre_ln=self.config.get("pre_ln", False),
                ).to(self.device)
                self.model.load_state_dict(state_dict)

            self.model.eval()
            self._script_masks = {}

            # Extract real checkpoint metadata & training trajectories
            dtypes = sorted({str(t.dtype).replace("torch.", "") for t in state_dict.values()})
            ckpt_size_mb = round(os.path.getsize(self.checkpoint_path) / (1024 * 1024), 2)
            legacy_hist = ckpt.get("history") if isinstance(ckpt.get("history"), dict) else {}
            train_hist = ckpt.get("train_loss_history") or legacy_hist.get("train_loss", [])
            val_hist = ckpt.get("val_loss_history") or legacy_hist.get("val_loss", [])
            lr_hist = ckpt.get("lr_history") or legacy_hist.get("lr", [])
            raw_epoch = ckpt.get("epoch")

            self.checkpoint_meta = {
                "is_modern": is_modern,
                "epoch": raw_epoch,
                "epochs_completed": (raw_epoch + 1) if (is_modern and isinstance(raw_epoch, int)) else raw_epoch,
                "global_step": ckpt.get("global_step"),
                "best_val_loss": float(ckpt["best_val_loss"]) if ckpt.get("best_val_loss") is not None else None,
                "train_loss_history": [float(x) for x in train_hist],
                "val_loss_history": [float(x) for x in val_hist],
                "lr_history": [float(x) for x in lr_hist],
                "timestamp": ckpt.get("timestamp"),
                "checkpoint_size_mb": ckpt_size_mb,
                "weight_dtype": ", ".join(dtypes) if dtypes else "float32",
                "tokenizer_vocab_size": self.tokenizer.vocab_size,
            }

            self.is_loaded = True
            logger.info("Custom Multilingual Transformer loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load custom Multilingual Transformer: {e}", exc_info=True)
            self.is_loaded = False
            return False

    def get_model_parameter_count(self) -> int:
        if self.model is None:
            return 0
        raw = self.model.count_parameters()
        if isinstance(raw, dict):
            return int(raw.get("trainable_parameters", raw.get("total_parameters", 0)))
        return int(raw)

    def get_display_model_name(self) -> str:
        if isinstance(self.model, ModernSeq2SeqTransformer):
            n_params = self.get_model_parameter_count()
            return f"TranslateMe Custom Multilingual Transformer (Modern {n_params / 1e6:.2f}M)"
        return "TranslateMe Custom Multilingual Transformer (From Scratch)"

    def _decode_token_display(self, token_id: int) -> str:
        """Formats a single subword token ID for human-readable attention heatmaps."""
        if self.tokenizer is None:
            return str(token_id)
        tok_str = self.tokenizer.tokenizer.id_to_token(token_id) or str(token_id)
        if token_id in (self.tokenizer.pad_id, self.tokenizer.sos_id, self.tokenizer.eos_id, self.tokenizer.unk_id) or tok_str.startswith("<2"):
            return tok_str
        decoded = self.tokenizer.decode([token_id], skip_special_tokens=False)
        if not decoded or "\ufffd" in decoded:
            return tok_str.replace("Ġ", " ")
        return decoded

    def get_script_mask(self, target_lang: str) -> Optional[torch.Tensor]:
        """
        Builds a boolean tensor of shape (vocab_size,) where True indicates
        tokens that belong to foreign alphabets and MUST be masked out.
        Prevents script contamination (e.g. Tamil characters appearing in Hindi).
        """
        if self.tokenizer is None or self.model is None:
            return None

        if target_lang in self._script_masks:
            return self._script_masks[target_lang]

        script_map = {"hi": "Devanagari", "ta": "Tamil", "te": "Telugu", "en": "Latin"}
        target_script = script_map.get(target_lang)
        if not target_script:
            return None

        vocab_size = self.tokenizer.vocab_size
        disallowed = torch.zeros(vocab_size, dtype=torch.bool, device=self.device)

        for i in range(vocab_size):
            if i in (self.tokenizer.pad_id, self.tokenizer.sos_id, self.tokenizer.eos_id, self.tokenizer.unk_id):
                continue
            tok_str = self.tokenizer.tokenizer.id_to_token(i) or ""
            if tok_str.startswith("<2"):
                continue

            decoded_str = self.tokenizer.decode([i]) or tok_str
            for ch in decoded_str:
                cp = ord(ch)
                if 0x0900 <= cp <= 0x097F and target_script != "Devanagari":
                    disallowed[i] = True
                    break
                elif 0x0B80 <= cp <= 0x0BFF and target_script != "Tamil":
                    disallowed[i] = True
                    break
                elif 0x0C00 <= cp <= 0x0C7F and target_script != "Telugu":
                    disallowed[i] = True
                    break

        self._script_masks[target_lang] = disallowed
        return disallowed

    def normalize_code(self, code: str) -> str:
        """Normalizes lang code to 'en', 'hi', 'ta', 'te', or 'auto'."""
        clean = code.strip()
        if clean.lower() == "auto":
            return "auto"
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
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        """
        Translates text using the custom trained Transformer.
        Supports 'auto' source detection via script analysis.
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            success = self.load_model()
            if not success or self.model is None or self.tokenizer is None:
                raise RuntimeError("Custom Multilingual Transformer could not be loaded. Please ensure model checkpoint exists.")

        src_code = self.normalize_code(source_lang)
        tgt_code = self.normalize_code(target_lang)
        detected_language: Optional[str] = None

        if src_code == "auto":
            det_code, _, _ = detect_language(text)
            src_code = CODE_MAP.get(det_code, "en")
            detected_language = src_code

        # Validate language support
        if not is_supported_language(src_code):
            raise ValueError(f"Source language '{source_lang}' is not supported by your custom Transformer. Supported: English (en), Hindi (hi), Tamil (ta), Telugu (te).")
        if not is_supported_language(tgt_code):
            raise ValueError(f"Target language '{target_lang}' is not supported by your custom Transformer. Supported: English (en), Hindi (hi), Tamil (ta), Telugu (te).")

        start_time = time.time()
        text_clean = text.strip()
        model_display_name = self.get_display_model_name()

        if not text_clean:
            return {
                "original_text": text,
                "translated_text": "",
                "source_language": src_code,
                "target_language": tgt_code,
                "source_language_name": REVERSE_MAP.get(src_code, src_code),
                "target_language_name": REVERSE_MAP.get(tgt_code, tgt_code),
                "detected_language": detected_language,
                "inference_time": 0.0,
                "model_name": model_display_name,
                "alternatives": [],
            }

        is_modern = isinstance(self.model, ModernSeq2SeqTransformer)
        ctrl_id = self.tokenizer.get_control_token_id(tgt_code)
        raw_ids = self.tokenizer.encode(text_clean, add_special_tokens=False)

        if is_modern:
            src_token_ids = [self.tokenizer.sos_id] + raw_ids + [self.tokenizer.eos_id]
            dec_prefix = [ctrl_id]
        else:
            src_token_ids = [self.tokenizer.sos_id, ctrl_id] + raw_ids + [self.tokenizer.eos_id]
            dec_prefix = None

        src_tensor = torch.tensor([src_token_ids], dtype=torch.long, device=self.device)

        cross_attention_matrix = None
        alternatives: List[str] = []
        target_token_ids: List[int] = []

        script_mask = self.get_script_mask(tgt_code)

        with torch.no_grad():
            if num_beams > 1:
                beams = self.model.beam_search_decode(
                    src_tensor,
                    beam_size=num_beams,
                    max_len=96 if is_modern else 64,
                    sos_idx=self.tokenizer.sos_id,
                    eos_idx=self.tokenizer.eos_id,
                    prefix_tokens=dec_prefix,
                    disallowed_mask=script_mask,
                    no_repeat_ngram_size=3 if is_modern else 2,
                )
                target_token_ids = beams[0][0][0].tolist()
                best_seq = [
                    i for i in target_token_ids
                    if i not in (self.tokenizer.sos_id, self.tokenizer.eos_id, self.tokenizer.pad_id)
                ]
                translated_text = self.tokenizer.decode(best_seq).strip()

                if return_alternatives and len(beams) > 1:
                    for b_seq, _ in beams[1:]:
                        alt_ids = [
                            i for i in b_seq[0].tolist()
                            if i not in (self.tokenizer.sos_id, self.tokenizer.eos_id, self.tokenizer.pad_id)
                        ]
                        alt_text = self.tokenizer.decode(alt_ids).strip()
                        if alt_text and alt_text != translated_text and alt_text not in alternatives:
                            alternatives.append(alt_text)
            else:
                decode_res = self.model.greedy_decode(
                    src_tensor,
                    sos_idx=self.tokenizer.sos_id,
                    eos_idx=self.tokenizer.eos_id,
                    prefix_tokens=dec_prefix,
                    max_len=96 if is_modern else 64,
                    return_attention=return_attention,
                    repetition_penalty=1.15 if is_modern else 1.2,
                    disallowed_mask=script_mask,
                    no_repeat_ngram_size=3 if is_modern else 2,
                )
                if return_attention:
                    out, attn = decode_res
                    if attn is not None:
                        cross_attention_matrix = attn[0].mean(dim=0).cpu().tolist()
                else:
                    out = decode_res
                target_token_ids = out[0].tolist()
                best_seq = [
                    i for i in target_token_ids
                    if i not in (self.tokenizer.sos_id, self.tokenizer.eos_id, self.tokenizer.pad_id)
                ]
                translated_text = self.tokenizer.decode(best_seq).strip()

        elapsed = round(time.time() - start_time, 4)

        result: Dict[str, Any] = {
            "original_text": text,
            "translated_text": translated_text,
            "source_language": src_code,
            "target_language": tgt_code,
            "source_language_name": REVERSE_MAP.get(src_code, src_code),
            "target_language_name": REVERSE_MAP.get(tgt_code, tgt_code),
            "detected_language": detected_language,
            "inference_time": elapsed,
            "model_name": model_display_name,
            "alternatives": alternatives,
        }

        if return_attention:
            result["cross_attention_matrix"] = cross_attention_matrix
            result["source_subwords"] = [self.tokenizer.tokenizer.id_to_token(i) for i in src_token_ids]
            result["source_display_tokens"] = [self._decode_token_display(i) for i in src_token_ids]
            result["target_subwords"] = [self.tokenizer.tokenizer.id_to_token(i) for i in target_token_ids]
            result["target_display_tokens"] = [self._decode_token_display(i) for i in target_token_ids]
            if isinstance(self.model, ModernSeq2SeqTransformer):
                result["decoder_layer"] = self.model.config.num_decoder_layers
                result["num_heads"] = self.model.config.n_heads
            else:
                result["decoder_layer"] = self.config.get("num_decoder_layers", self.config.get("num_layers", 6))
                result["num_heads"] = self.config.get("n_heads", 6)

        return result


custom_multilingual_engine = CustomMultilingualEngine()
