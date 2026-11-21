"""
Evaluation Metrics Suite for Neural Machine Translation
Computes SacreBLEU, chrF++, Inference Latency, and Throughput.
"""

import time
from typing import List, Dict
import sacrebleu


def compute_sacrebleu(hypotheses: List[str], references: List[str]) -> float:
    """
    Computes standard SacreBLEU score across reference translations.
    """
    if not hypotheses or not references:
        return 0.0
    # sacrebleu expects references as list of lists (e.g. [[ref1_a, ref2_a, ...]])
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return round(bleu.score, 2)


def compute_chrf(hypotheses: List[str], references: List[str]) -> float:
    """
    Computes chrF++ character n-gram F-score.
    Particularly well-suited for morphologically rich Indian languages.
    """
    if not hypotheses or not references:
        return 0.0
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    return round(chrf.score, 2)


def benchmark_translation_speed(translate_fn, test_sentences: List[str]) -> Dict[str, float]:
    """
    Measures average inference latency and token throughput.
    """
    if not test_sentences:
        return {"avg_latency_sec": 0.0, "throughput_chars_per_sec": 0.0}

    total_chars = sum(len(s) for s in test_sentences)
    t0 = time.perf_counter()
    outputs = []
    for s in test_sentences:
        outputs.append(translate_fn(s))
    elapsed = time.perf_counter() - t0

    avg_latency = elapsed / len(test_sentences)
    throughput = total_chars / max(0.001, elapsed)

    return {
        "avg_latency_sec": round(avg_latency, 3),
        "throughput_chars_per_sec": round(throughput, 1),
        "total_time_sec": round(elapsed, 2)
    }
