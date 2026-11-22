"""
Multilingual Subword Tokenizer for TranslateMe
Trained strictly on our parallel corpus from scratch.
Supports English, Hindi (Devanagari), Tamil (Tamil), and Telugu (Telugu) scripts
with target language direction tokens (<2en>, <2hi>, <2ta>, <2te>).
"""

import os
from typing import List, Optional, Union, Iterable
from tokenizers import Tokenizer, decoders, pre_tokenizers, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from .language_registry import SPECIAL_TOKENS, get_control_token, is_supported_language


class MultilingualSubwordTokenizer:
    """
    Multilingual subword tokenizer wrapper using Byte-Pair Encoding (BPE).
    Trains vocabulary from scratch on multilingual text without pretrained weights.
    """
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<bos>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, tokenizer_file: Optional[str] = None):
        if tokenizer_file and os.path.exists(tokenizer_file):
            self.tokenizer = Tokenizer.from_file(tokenizer_file)
            self.tokenizer.decoder = decoders.ByteLevel()
        else:
            self.tokenizer = Tokenizer(BPE(unk_token=self.UNK_TOKEN))
            self.tokenizer.normalizer = normalizers.NFC()
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = decoders.ByteLevel()

        # Cache special token IDs
        self._cache_special_token_ids()

    def _cache_special_token_ids(self):
        try:
            self.pad_id = self.tokenizer.token_to_id("<pad>") if self.tokenizer.token_to_id("<pad>") is not None else 0
            self.unk_id = self.tokenizer.token_to_id("<unk>") if self.tokenizer.token_to_id("<unk>") is not None else 1
            bos_cand = self.tokenizer.token_to_id("<bos>")
            self.sos_id = bos_cand if bos_cand is not None else (self.tokenizer.token_to_id("<sos>") or 2)
            self.eos_id = self.tokenizer.token_to_id("<eos>") if self.tokenizer.token_to_id("<eos>") is not None else 3
        except Exception:
            self.pad_id = 0
            self.unk_id = 1
            self.sos_id = 2
            self.eos_id = 3

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def get_control_token_id(self, target_lang: str) -> int:
        """Returns the token ID for a target language control token, e.g. '<2hi>'."""
        control_token = get_control_token(target_lang)
        token_id = self.tokenizer.token_to_id(control_token)
        if token_id is None:
            raise ValueError(f"Control token '{control_token}' for language '{target_lang}' not found in vocabulary.")
        return token_id

    def train_from_iterator(
        self,
        iterator: Iterable[str],
        vocab_size: int = 8000,
        min_frequency: int = 1
    ):
        """
        Trains BPE subword vocabulary from scratch over text iterator.
        """
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=min_frequency,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True
        )
        self.tokenizer.train_from_iterator(iterator, trainer=trainer)
        self._cache_special_token_ids()

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        target_lang: Optional[str] = None
    ) -> List[int]:
        """
        Encodes text into subword token IDs.
        If target_lang is provided, prepends the corresponding target direction token (e.g. <2hi>).
        """
        encoding = self.tokenizer.encode(text.strip())
        token_ids = encoding.ids

        if add_special_tokens:
            prefix = [self.sos_id]
            if target_lang is not None:
                prefix.append(self.get_control_token_id(target_lang))
            token_ids = prefix + token_ids + [self.eos_id]

        return token_ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decodes subword token IDs back into string.
        """
        if skip_special_tokens:
            # Filter out all special tokens: pad, sos, eos, unk, and direction tokens (<2hi>, etc.)
            filtered_ids = [
                i for i in token_ids
                if i not in (self.pad_id, self.sos_id, self.eos_id)
                and not (self.tokenizer.id_to_token(i) or "").startswith("<2")
            ]
            return self.tokenizer.decode(filtered_ids)
        return self.tokenizer.decode(token_ids)

    def save(self, filepath: str):
        """Saves tokenizer state to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.tokenizer.save(filepath)

    @classmethod
    def load(cls, filepath: str) -> "MultilingualSubwordTokenizer":
        return cls(tokenizer_file=filepath)
