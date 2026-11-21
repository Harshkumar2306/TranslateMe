"""
Dataset Loader for Multilingual Machine Translation Parallel Corpora
Supports sample high-quality bilingual corpora and custom CSV/TSV/JSONL files.
"""

import os
import json
import random
from typing import List, Tuple, Dict
from .preprocessing import filter_parallel_pairs

# High-quality verified sample parallel sentence pairs (English -> Hindi)
SAMPLE_EN_HI_CORPUS = [
    ("Artificial intelligence is transforming the field of language translation.", "आर्टिफिशियल इंटेलिजेंस भाषा अनुवाद के क्षेत्र को बदल रहा है।"),
    ("Transformers use self-attention mechanisms to process sequential text.", "ट्रांसफॉर्मर अनुक्रमिक पाठ को संसाधित करने के लिए सेल्फ-अटेंशन तंत्र का उपयोग करते हैं।"),
    ("Neural machine translation models can learn mappings between hundreds of languages.", "न्यूरल मशीन ट्रांसलेशन मॉडल सैकड़ों भाषाओं के बीच मैपिंग सीख सकते हैं।"),
    ("The student submitted the deep learning project before the deadline.", "छात्र ने समय सीमा से पहले डीप लर्निंग प्रोजेक्ट जमा कर दिया।"),
    ("Positional encoding allows the Transformer to understand token order.", "पोजीशनल एन्कोडिंग ट्रांसफॉर्मर को टोकन क्रम को समझने की अनुमति देता है।"),
    ("India is a land of immense linguistic diversity with hundreds of languages.", "भारत सैकड़ों भाषाओं के साथ अपार भाषाई विविधता की भूमि है।"),
    ("Deep learning researchers evaluate models using SacreBLEU and chrF scores.", "डीप लर्निंग शोधकर्ता SacreBLEU और chrF स्कोर का उपयोग करके मॉडल का मूल्यांकन करते हैं।"),
    ("The encoder extracts contextual representations from the source sentence.", "एन्कोडर स्रोत वाक्य से प्रासंगिक प्रतिनिधित्व निकालता है।"),
    ("The decoder generates target words one token at a time autoregressively.", "डिकोडर ऑटोरेग्रेसिव रूप से एक समय में एक टोकन लक्ष्य शब्द उत्पन्न करता है।"),
    ("Knowledge and education empower individuals to achieve their dreams.", "ज्ञान और शिक्षा व्यक्तियों को अपने सपनों को पूरा करने के लिए सशक्त बनाती है।"),
    ("Natural language processing bridges the communication gap across cultures.", "नेचुरल लैंग्वेज प्रोसेसिंग विभिन्न संस्कृतियों के बीच संचार अंतर को पाटती है।"),
    ("Machine learning requires quality parallel corpora for effective fine-tuning.", "मशीन लर्निंग को प्रभावी फाइन-ट्यूनिंग के लिए गुणवत्ता वाले समानांतर डेटासेट की आवश्यकता होती है।"),
    ("The library contains thousands of books on science, mathematics, and philosophy.", "पुस्तकालय में विज्ञान, गणित और दर्शन पर हजारों पुस्तकें हैं।"),
    ("Good morning, I hope you have a productive and wonderful day.", "शुभ प्रभात, मुझे आशा है कि आपका दिन उत्पादक और शानदार होगा।"),
    ("The quick brown fox jumps over the lazy dog.", "तेज भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूदती है।"),
    ("Translation systems must preserve the nuanced semantic meaning of sentences.", "अनुवाद प्रणालियों को वाक्यों के सूक्ष्म अर्थ को संरक्षित रखना चाहिए।"),
    ("Modern technology connects communities from all around the globe.", "आधुनिक तकनीक दुनिया भर के समुदायों को जोड़ती है।"),
    ("Deep neural networks learn abstract representations across hierarchical layers.", "डीप न्यूरल नेटवर्क पदानुक्रमित परतों में अमूर्त प्रतिनिधित्व सीखते हैं।"),
    ("Tamil and Telugu are classical languages with rich literary traditions.", "तमिल और तेलुगु समृद्ध साहित्यिक परंपराओं वाली शास्त्रीय भाषाएं हैं।"),
    ("The evaluation report demonstrates competitive performance across test benchmarks.", "मूल्यांकन रिपोर्ट परीक्षण बेंचमार्क में प्रतिस्पर्धी प्रदर्शन को प्रदर्शित करती है।"),
    ("Attention Is All You Need introduced the breakthrough Transformer architecture.", "अटेंशन इज़ ऑल यू नीड ने अभूतपूर्व ट्रांसफॉर्मर आर्किटेक्चर की शुरुआत की।"),
    ("Welcome to LinguaAI universal neural machine translation platform.", "लिंगुआएआई यूनिवर्सल न्यूरल मशीन ट्रांसलेशन प्लेटफॉर्म में आपका स्वागत है।"),
]


def load_parallel_corpus(
    lang_pair: str = "en-hi",
    raw_path: str = None
) -> List[Tuple[str, str]]:
    """Loads parallel sentence pairs from file or verified sample corpus."""
    if raw_path and os.path.exists(raw_path):
        pairs = []
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                if "\t" in line:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        pairs.append((parts[0], parts[1]))
        return pairs
    return SAMPLE_EN_HI_CORPUS


def prepare_dataset_splits(
    pairs: List[Tuple[str, str]],
    output_dir: str = "datasets/processed",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, str]:
    """
    Cleans, deduplicates, and splits data into train, val, test splits.
    Saves to JSONL files.
    """
    os.makedirs(output_dir, exist_ok=True)
    cleaned_pairs, stats = filter_parallel_pairs(pairs)

    random.seed(seed)
    random.shuffle(cleaned_pairs)

    n_total = len(cleaned_pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_data = cleaned_pairs[:n_train]
    val_data = cleaned_pairs[n_train:n_train + n_val]
    test_data = cleaned_pairs[n_train + n_val:]

    splits = {
        "train": (train_data, os.path.join(output_dir, "train.jsonl")),
        "val": (val_data, os.path.join(output_dir, "val.jsonl")),
        "test": (test_data, os.path.join(output_dir, "test.jsonl")),
    }

    file_paths = {}
    for name, (data, path) in splits.items():
        with open(path, "w", encoding="utf-8") as f:
            for src, tgt in data:
                f.write(json.dumps({"source": src, "target": tgt}, ensure_ascii=False) + "\n")
        file_paths[name] = path

    stats["train_count"] = len(train_data)
    stats["val_count"] = len(val_data)
    stats["test_count"] = len(test_data)

    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    file_paths["stats"] = stats_path

    print(f"Dataset preparation complete:")
    print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"  Saved to: {output_dir}")

    return file_paths


if __name__ == "__main__":
    pairs = load_parallel_corpus("en-hi")
    prepare_dataset_splits(pairs)
