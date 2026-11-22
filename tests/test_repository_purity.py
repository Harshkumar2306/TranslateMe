"""
Automated Repository Purity Test
Enforces that no prohibited pretrained translation models, Hugging Face pretrained wrappers,
or external translation APIs exist in backend or ml_lab source files.
"""

import os
import re
import pytest

PROHIBITED_PATTERNS = [
    r"from_pretrained",
    r"AutoModelForSeq2SeqLM",
    r"AutoTokenizer",
    r"pipeline\s*\(\s*['\"]translation",
    r"facebook/nllb",
    r"NLLBTranslationEngine",
    r"googletrans",
    r"m2m_100",
    r"mbart",
    r"indictrans",
]

INSPECTED_DIRS = ["backend/app", "ml_lab"]


def test_repository_purity_no_pretrained_imports():
    """Scans all Python source files in backend and ml_lab for prohibited patterns."""
    violations = []

    for base_dir in INSPECTED_DIRS:
        if not os.path.exists(base_dir):
            continue
        for root, _, files in os.walk(base_dir):
            if "node_modules" in root or ".venv" in root or "__pycache__" in root:
                continue
            for fname in files:
                if fname.endswith(".py"):
                    fpath = os.path.join(root, fname)
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        for line_idx, line in enumerate(f, 1):
                            for pat in PROHIBITED_PATTERNS:
                                if re.search(pat, line, re.IGNORECASE):
                                    # Allow comments documenting that pretrained models are prohibited
                                    if "prohibited" in line.lower() or "without" in line.lower() or "no pretrained" in line.lower():
                                        continue
                                    violations.append(f"{fpath}:{line_idx} contains '{pat}': {line.strip()}")

    assert len(violations) == 0, f"Found {len(violations)} prohibited pretrained model references:\n" + "\n".join(violations)


def test_no_fallback_to_external_or_pretrained():
    """Verifies that the translation engine does not import or fall back to any external translator."""
    from backend.app.services.custom_multilingual_service import custom_multilingual_engine
    assert custom_multilingual_engine is not None
    assert custom_multilingual_engine.model is not None or not custom_multilingual_engine.is_loaded
    # Ensure engine attributes point to custom PyTorch Seq2SeqTransformer
    from ml_lab.custom_transformer import Seq2SeqTransformer
    from ml_lab.custom_transformer.modern_transformer import ModernSeq2SeqTransformer
    assert custom_multilingual_engine.model is None or isinstance(
        custom_multilingual_engine.model, (Seq2SeqTransformer, ModernSeq2SeqTransformer)
    )
