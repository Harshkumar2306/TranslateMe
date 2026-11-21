"""
Dataset and Data Pipeline Package for TranslateMe
"""

from .data_pipeline import (
    normalize_unicode_text,
    filter_parallel_pair,
    deduplicate_pairs,
    clean_and_prepare_parallel_data,
    split_parallel_dataset,
    compute_dataset_stats,
    save_parallel_split,
    load_parallel_split,
)
from .curated_corpus import (
    MULTILINGUAL_CORPUS_RECORDS,
    generate_parallel_pairs,
    get_all_monolingual_sentences,
    generate_multilingual_training_set,
)

__all__ = [
    "normalize_unicode_text",
    "filter_parallel_pair",
    "deduplicate_pairs",
    "clean_and_prepare_parallel_data",
    "split_parallel_dataset",
    "compute_dataset_stats",
    "save_parallel_split",
    "load_parallel_split",
    "MULTILINGUAL_CORPUS_RECORDS",
    "generate_parallel_pairs",
    "get_all_monolingual_sentences",
    "generate_multilingual_training_set",
]
