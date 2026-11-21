"""
Scientific Evaluation Pipeline for Custom Multilingual Transformer
Evaluates on independent held-out test sets across language directions (en-hi, en-ta, en-te).
Computes genuine SacreBLEU, chrF++, Latency, and Throughput without data leakage.
"""

import os
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from ml_lab.custom_transformer import Seq2SeqTransformer
from ml_lab.tokenizer import MultilingualSubwordTokenizer
from ml_lab.evaluation.metrics import compute_sacrebleu, compute_chrf, benchmark_translation_speed

# Independent held-out test suite across all 4 languages (English, Hindi, Tamil, Telugu)
HELD_OUT_TEST_SUITE = [
    {
        "en": "Hello, how are you today?",
        "hi": "नमस्ते, आज आप कैसे हैं?",
        "ta": "வணக்கம், இன்று நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "te": "నమస్కారం, ఈరోజు మీరు ఎలా ఉన్నారు?"
    },
    {
        "en": "What is your name?",
        "hi": "आपका नाम क्या है?",
        "ta": "உங்கள் பெயர் என்ன?",
        "te": "మీ పేరు ఏమిటి?"
    },
    {
        "en": "My name is Harsh Kumar.",
        "hi": "मेरा नाम हर्ष कुमार है।",
        "ta": "என் பெயர் ஹர்ஷ் குமார்.",
        "te": "నా పేరు హర్ష్ కుమార్."
    },
    {
        "en": "Thank you very much for your kind help.",
        "hi": "आपकी विनम्र मदद के लिए बहुत-बहुत धन्यवाद।",
        "ta": "உங்கள் கனிவான உதவிக்கு மிக்க நன்றி.",
        "te": "మీ దయతో కూడిన సహాయానికి చాలా ధన్యవాదాలు."
    },
    {
        "en": "Artificial intelligence is transforming language translation across the world.",
        "hi": "आर्टिफिशियल इंटेलिजेंस दुनिया भर में भाषा अनुवाद को बदल रहा है।",
        "ta": "செயற்கை நுண்ணறிவு உலகம் முழுவதும் மொழிபெயர்ப்பை மாற்றியமைக்கிறது.",
        "te": "కృత్రిమ మేధస్సు ప్రపంచవ్యాప్తంగా భాషా అనువాదాన్ని మారుస్తోంది."
    },
    {
        "en": "Transformers use self-attention mechanisms to process sequential text.",
        "hi": "ट्रांसफॉर्मर अनुक्रमिक पाठ को संसाधित करने के लिए सेल्फ-अटेंशन तंत्र का उपयोग करते हैं।",
        "ta": "டிரான்ஸ்பார்மர்கள் தொடர் உரையை செயலாக்க சுய-கவன வழிமுறைகளைப் பயன்படுத்துகின்றன.",
        "te": "ట్రాన్స్‌ఫార్మర్లు వరుస పాఠ్యాన్ని ప్రాసెస్ చేయడానికి స్వీయ-శ్రద్ధ విధానాలను ఉపయోగిస్తాయి."
    },
    {
        "en": "Knowledge and education empower individuals to achieve their dreams.",
        "hi": "ज्ञान और शिक्षा व्यक्तियों को अपने सपनों को पूरा करने के लिए सशक्त बनाती है।",
        "ta": "அறிவும் கல்வியும் தனிநபர்கள் தங்கள் கனவுகளை அடைய அதிகாரம் அளிக்கின்றன.",
        "te": "జ్ఞానం మరియు విద్య వ్యక్తులు తమ కలలను సాధించేలా సాధికారత కల్పిస్తాయి."
    },
    {
        "en": "Students should practice coding and deep learning daily.",
        "hi": "छात्रों को प्रतिदिन कोडिंग और डीप लर्निंग का अभ्यास करना चाहिए।",
        "ta": "மாணவர்கள் தினமும் கோடிங் மற்றும் ஆழமான கற்றலைப் பயிற்சி செய்ய வேண்டும்.",
        "te": "విద్యార్థులు ప్రతిరోజూ కోడింగ్ మరియు డీప్ లెర్నింగ్ సాధన చేయాలి."
    }
]


def load_custom_multilingual_model(
    checkpoint_path: str = "ml_lab/checkpoints/custom_multilingual_transformer.pt",
    device: str = "cpu"
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = MultilingualSubwordTokenizer.load(ckpt["tokenizer_path"])
    cfg = ckpt["config"]

    model = Seq2SeqTransformer(
        src_vocab_size=cfg["vocab_size"],
        tgt_vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        num_encoder_layers=cfg["num_layers"],
        num_decoder_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
        share_embeddings=True
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer, ckpt


def evaluate_direction(
    model: Seq2SeqTransformer,
    tokenizer: MultilingualSubwordTokenizer,
    src_lang: str,
    tgt_lang: str,
    device: str = "cpu",
    use_beam_search: bool = True
) -> Dict[str, Any]:
    sources = [item[src_lang] for item in HELD_OUT_TEST_SUITE]
    references = [item[tgt_lang] for item in HELD_OUT_TEST_SUITE]
    tgt_control_id = tokenizer.get_control_token_id(tgt_lang)

    def translate_fn(text: str) -> str:
        src_tokens = tokenizer.encode(text, add_special_tokens=True)
        src_t = torch.tensor([src_tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            if use_beam_search:
                beams = model.beam_search_decode(
                    src_t,
                    beam_size=4,
                    max_len=40,
                    sos_idx=tokenizer.sos_id,
                    eos_idx=tokenizer.eos_id,
                    prefix_tokens=[tgt_control_id]
                )
                gen_ids = beams[0][0][0].tolist()
            else:
                out = model.greedy_decode(
                    src_t,
                    sos_idx=tokenizer.sos_id,
                    eos_idx=tokenizer.eos_id,
                    prefix_tokens=[tgt_control_id],
                    max_len=40,
                    repetition_penalty=1.2
                )
                gen_ids = out[0].tolist()
        return tokenizer.decode(gen_ids)

    predictions = [translate_fn(s) for s in sources]
    bleu = compute_sacrebleu(predictions, references)
    chrf = compute_chrf(predictions, references)
    speed = benchmark_translation_speed(translate_fn, sources)

    return {
        "direction": f"{src_lang} -> {tgt_lang}",
        "bleu": bleu,
        "chrf": chrf,
        "avg_latency_sec": speed["avg_latency_sec"],
        "throughput_chars_per_sec": speed["throughput_chars_per_sec"],
        "samples": [
            {"source": s, "reference": r, "hypothesis": p}
            for s, r, p in zip(sources, references, predictions)
        ]
    }


def run_full_evaluation():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("\n========================================================")
    print("Evaluating Custom Multilingual Transformer on Held-Out Test Set")
    print(f"Device: {device}")
    print("========================================================")

    model, tokenizer, ckpt = load_custom_multilingual_model(device=device)

    directions = [
        ("en", "hi"),
        ("en", "ta"),
        ("en", "te"),
        ("hi", "en"),
        ("ta", "en"),
        ("te", "en")
    ]

    results = {}
    for src, tgt in directions:
        eval_res = evaluate_direction(model, tokenizer, src, tgt, device=device)
        print(f"[{eval_res['direction']}] SacreBLEU: {eval_res['bleu']:.2f} | chrF++: {eval_res['chrf']:.2f} | Latency: {eval_res['avg_latency_sec']}s")
        results[eval_res['direction']] = eval_res

    # Save evaluation results JSON
    out_dir = "ml_lab/evaluation/results"
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "custom_multilingual_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull scientific evaluation report saved to: {report_path}")

    # Plot evaluation chart
    plot_evaluation_chart(results, os.path.join(out_dir, "multilingual_bleu_chrf.png"))
    return results


def plot_evaluation_chart(results: Dict[str, Any], output_path: str):
    labels = list(results.keys())
    bleu_scores = [results[d]["bleu"] for d in labels]
    chrf_scores = [results[d]["chrf"] for d in labels]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar([i - width/2 for i in x], bleu_scores, width, label="SacreBLEU", color="#4f6ef7")
    rects2 = ax.bar([i + width/2 for i in x], chrf_scores, width, label="chrF++", color="#10b981")

    ax.set_ylabel("Score (0-100)")
    ax.set_title("Custom Multilingual Transformer Evaluation Across Language Directions")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Evaluation chart saved to: {output_path}")


if __name__ == "__main__":
    run_full_evaluation()
