"""
Unit tests for Multilingual Subword Tokenizer and Language Registry
"""

import os
import pytest
from ml_lab.tokenizer import (
    MultilingualSubwordTokenizer,
    SUPPORTED_LANGUAGES,
    is_supported_language,
    get_control_token
)


def test_language_registry():
    assert is_supported_language("en")
    assert is_supported_language("hi")
    assert is_supported_language("ta")
    assert is_supported_language("te")
    assert not is_supported_language("fr")  # French not in custom trained subset

    assert get_control_token("hi") == "<2hi>"
    assert get_control_token("ta") == "<2ta>"
    assert get_control_token("te") == "<2te>"
    assert get_control_token("en") == "<2en>"


def test_tokenizer_training_and_roundtrip(tmp_path):
    tokenizer = MultilingualSubwordTokenizer()
    corpus = [
        "Hello how are you today",
        "नमस्ते आप कैसे हैं",
        "வணக்கம் நீங்கள் எப்படி இருக்கிறீர்கள்",
        "నమస్కారం మీరు ఎలా ఉన్నారు",
        "Artificial intelligence and deep learning transformers",
        "मशीन अनुवाद भाषा को जोड़ता है",
        "இயந்திர மொழிபெயர்ப்பு",
        "యంత్ర అనువాదం"
    ]

    tokenizer.train_from_iterator(corpus, vocab_size=500, min_frequency=1)
    assert tokenizer.vocab_size > 50

    # Test special token IDs
    assert tokenizer.pad_id == 0
    assert tokenizer.unk_id == 1
    assert tokenizer.sos_id == 2
    assert tokenizer.eos_id == 3

    # Test encoding with target language control token
    hi_control_id = tokenizer.get_control_token_id("hi")
    encoded_hi = tokenizer.encode("Hello", add_special_tokens=True, target_lang="hi")
    assert encoded_hi[0] == tokenizer.sos_id
    assert encoded_hi[1] == hi_control_id
    assert encoded_hi[-1] == tokenizer.eos_id

    # Test decoding roundtrip across all 4 scripts
    for sample in ["Hello how are you", "नमस्ते आप कैसे", "வணக்கம்", "నమస్కారం"]:
        enc = tokenizer.encode(sample, add_special_tokens=False)
        dec = tokenizer.decode(enc)
        assert dec.strip() == sample.strip()

    # Test save and load
    save_path = str(tmp_path / "test_tok.json")
    tokenizer.save(save_path)
    assert os.path.exists(save_path)

    loaded = MultilingualSubwordTokenizer.load(save_path)
    assert loaded.vocab_size == tokenizer.vocab_size
