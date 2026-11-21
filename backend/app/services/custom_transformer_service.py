"""
Inference Service for Custom Transformer Built From Scratch
Uses Seq2SeqTransformer implemented in ml_lab.custom_transformer
Checkpoint trained with PyTorch on Apple MPS / CUDA
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any
import torch

from ml_lab.custom_transformer import Seq2SeqTransformer
from backend.app.core.device import get_device

logger = logging.getLogger(__name__)


class CustomTransformerEngine:
    def __init__(self, checkpoint_path: str = "ml_lab/checkpoints/custom_transformer_checkpoint.pt"):
        self.checkpoint_path = checkpoint_path
        self.model: Optional[Seq2SeqTransformer] = None
        self.src_vocab: Dict[str, int] = {}
        self.tgt_vocab: Dict[str, int] = {}
        self.inv_tgt_vocab: Dict[int, str] = {}
        self.config: Dict[str, Any] = {}
        self.is_loaded = False
        self.device = get_device()

    def load_model(self):
        """Loads trained weights and vocabularies into custom Seq2SeqTransformer."""
        if not os.path.exists(self.checkpoint_path):
            logger.warning(f"Custom Transformer checkpoint not found at: {self.checkpoint_path}")
            return False

        try:
            logger.info(f"Loading custom PyTorch Transformer from {self.checkpoint_path} onto {self.device}...")
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            self.src_vocab = ckpt["src_vocab"]
            self.tgt_vocab = ckpt["tgt_vocab"]
            self.inv_tgt_vocab = {v: k for k, v in self.tgt_vocab.items()}
            self.config = ckpt.get("config", {"d_model": 128, "n_heads": 4, "num_layers": 2, "d_ff": 256})

            self.model = Seq2SeqTransformer(
                src_vocab_size=len(self.src_vocab),
                tgt_vocab_size=len(self.tgt_vocab),
                d_model=self.config.get("d_model", 128),
                n_heads=self.config.get("n_heads", 4),
                num_encoder_layers=self.config.get("num_layers", 2),
                num_decoder_layers=self.config.get("num_layers", 2),
                d_ff=self.config.get("d_ff", 256),
            ).to(self.device)

            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            self.is_loaded = True
            logger.info("Custom PyTorch Transformer loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load custom Transformer: {e}")
            self.is_loaded = False
            return False

    def translate(self, text: str, max_len: int = 32) -> Dict[str, Any]:
        """
        Translates input text using the custom trained Transformer.
        """
        if not self.is_loaded:
            success = self.load_model()
            if not success or self.model is None:
                raise RuntimeError("Custom Transformer is not loaded.")

        start_time = time.time()
        words = text.strip().lower().split()
        if not words:
            return {
                "original_text": text,
                "translated_text": "",
                "source_language": "eng_Latn",
                "target_language": "hin_Deva",
                "source_language_name": "English",
                "target_language_name": "Hindi (Custom Model)",
                "detected_language": None,
                "inference_time": 0.0,
                "model_name": "Custom PyTorch Transformer (From Scratch)",
                "source_tokens": [],
                "target_tokens": [],
                "alternatives": []
            }

        sos_id = self.src_vocab.get("<sos>", 1)
        eos_id = self.src_vocab.get("<eos>", 2)
        unk_id = self.src_vocab.get("<unk>", 3)
        pad_id = self.tgt_vocab.get("<pad>", 0)

        # Convert words to token IDs
        token_ids = [sos_id] + [self.src_vocab.get(w, unk_id) for w in words] + [eos_id]
        src_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            generated_ids = self.model.greedy_decode(
                src_tensor,
                sos_idx=self.tgt_vocab.get("<sos>", 1),
                eos_idx=self.tgt_vocab.get("<eos>", 2),
                max_len=max_len
            )

        elapsed = time.time() - start_time
        gen_list = generated_ids[0].tolist()

        translated_words = []
        for idx in gen_list:
            if idx in (self.tgt_vocab.get("<sos>", 1), pad_id):
                continue
            if idx == self.tgt_vocab.get("<eos>", 2):
                break
            translated_words.append(self.inv_tgt_vocab.get(idx, "<unk>"))

        translated_text = " ".join(translated_words)

        return {
            "original_text": text,
            "translated_text": translated_text,
            "source_language": "eng_Latn",
            "target_language": "hin_Deva",
            "source_language_name": "English",
            "target_language_name": "Hindi (Custom Model)",
            "detected_language": None,
            "inference_time": round(elapsed, 4),
            "model_name": "Custom PyTorch Transformer (Trained From Scratch)",
            "architecture": {
                "d_model": self.config.get("d_model", 128),
                "n_heads": self.config.get("n_heads", 4),
                "layers": self.config.get("num_layers", 2),
                "d_ff": self.config.get("d_ff", 256),
                "vocab_size": len(self.src_vocab)
            },
            "source_tokens": [self.src_vocab.get(w, unk_id) for w in words],
            "target_tokens": gen_list,
            "alternatives": []
        }


custom_transformer_engine = CustomTransformerEngine()
