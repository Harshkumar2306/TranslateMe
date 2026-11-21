"""
Unit Test for Zero Dataset Leakage
Verifies record isolation, fuzzy overlap, and cross-direction independence.
"""

from ml_lab.evaluation.leakage_audit import run_comprehensive_leakage_audit


def test_zero_dataset_leakage():
    report = run_comprehensive_leakage_audit()
    assert report["status"] == "PASSED"
    assert report["exact_sentence_overlap"] == 0
    assert report["fuzzy_normalized_overlap"] == 0
    assert report["cross_direction_leakage"] == 0
    assert report["tokenizer_test_leakage"] == 0
