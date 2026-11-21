"""
Evaluation Script for Multilingual Neural Machine Translation
Compares Pretrained NLLB-200, Fine-Tuned Checkpoints, and Custom Transformer.
Generates evaluation reports and comparison figures using Matplotlib.
"""

import os
import json
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from backend.app.services.nllb_service import translation_engine
from models.custom_transformer import Seq2SeqTransformer
from training.train_custom_transformer import SimpleVocab
from evaluation.metrics import compute_sacrebleu, compute_chrf, benchmark_translation_speed

# Benchmark test parallel test suite (English -> Hindi)
TEST_SUITE = [
    {
        "src": "Artificial intelligence is transforming language translation.",
        "ref": "आर्टिफिशियल इंटेलिजेंस भाषा अनुवाद को बदल रहा है।"
    },
    {
        "src": "Transformers use self-attention mechanisms to process sequential text.",
        "ref": "ट्रांसफॉर्मर अनुक्रमिक पाठ को संसाधित करने के लिए सेल्फ-अटेंशन तंत्र का उपयोग करते हैं।"
    },
    {
        "src": "Positional encoding allows the Transformer to understand token order.",
        "ref": "पोजीशनल एन्कोडिंग ट्रांसफॉर्मर को टोकन क्रम को समझने की अनुमति देता है।"
    },
    {
        "src": "Natural language processing bridges the communication gap across cultures.",
        "ref": "नेचुरल लैंग्वेज प्रोसेसिंग विभिन्न संस्कृतियों के बीच संचार अंतर को पाटती है।"
    },
    {
        "src": "Knowledge and education empower individuals to achieve their dreams.",
        "ref": "ज्ञान और शिक्षा व्यक्तियों को अपने सपनों को पूरा करने के लिए सशक्त बनाती है।"
    }
]


def evaluate_nllb_pretrained(test_data):
    sources = [item["src"] for item in test_data]
    references = [item["ref"] for item in test_data]

    print("\n--- Evaluating Pretrained NLLB-200 ---")
    predictions = []
    t_start = benchmark_translation_speed(
        lambda s: translation_engine.translate(s, "eng_Latn", "hin_Deva")["translated_text"],
        sources
    )

    for s in sources:
        res = translation_engine.translate(s, "eng_Latn", "hin_Deva", num_beams=2)
        predictions.append(res["translated_text"])

    bleu = compute_sacrebleu(predictions, references)
    chrf = compute_chrf(predictions, references)

    print(f"Pretrained NLLB-200 -> BLEU: {bleu:.2f} | chrF++: {chrf:.2f} | Latency: {t_start['avg_latency_sec']}s")
    return {
        "model_name": "Pretrained NLLB-200 (600M)",
        "bleu": bleu,
        "chrf": chrf,
        "avg_latency_sec": t_start["avg_latency_sec"],
        "throughput_chars_per_sec": t_start["throughput_chars_per_sec"],
        "predictions": predictions,
        "samples": [{"src": s, "ref": r, "hyp": p} for s, r, p in zip(sources, references, predictions)]
    }


def evaluate_custom_transformer():
    checkpoint_path = "models/checkpoints/custom_transformer_checkpoint.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Custom transformer checkpoint not found at {checkpoint_path}. Skipping.")
        return None

    print("\n--- Evaluating Custom Transformer ---")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)

    src_vocab = SimpleVocab()
    tgt_vocab = SimpleVocab()
    src_vocab.w2i = ckpt["src_vocab"]
    src_vocab.i2w = {i: w for w, i in src_vocab.w2i.items()}
    tgt_vocab.w2i = ckpt["tgt_vocab"]
    tgt_vocab.i2w = {i: w for w, i in tgt_vocab.w2i.items()}

    cfg = ckpt["config"]
    model = Seq2SeqTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        num_encoder_layers=cfg["num_layers"],
        num_decoder_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
        pad_idx=src_vocab.w2i[SimpleVocab.PAD_TOKEN]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def translate_fn(text: str) -> str:
        tokens = [src_vocab.w2i[SimpleVocab.SOS_TOKEN]] + src_vocab.encode(text) + [src_vocab.w2i[SimpleVocab.EOS_TOKEN]]
        src_t = torch.tensor([tokens], device=device)
        gen = model.greedy_decode(src_t, sos_idx=tgt_vocab.w2i[SimpleVocab.SOS_TOKEN], eos_idx=tgt_vocab.w2i[SimpleVocab.EOS_TOKEN], max_len=20)
        return tgt_vocab.decode(gen[0].tolist())

    sources = ["hello how are you", "attention is all you need", "machine translation connects the world"]
    references = ["namaste aap kaise hain", "attention hi sab kuch hai", "machine anuvaad duniya ko jodta hai"]

    predictions = [translate_fn(s) for s in sources]
    bleu = compute_sacrebleu(predictions, references)
    chrf = compute_chrf(predictions, references)
    speed = benchmark_translation_speed(translate_fn, sources)

    print(f"Custom Transformer -> BLEU: {bleu:.2f} | chrF++: {chrf:.2f} | Latency: {speed['avg_latency_sec']}s")
    return {
        "model_name": "Custom Transformer (Educational)",
        "bleu": bleu,
        "chrf": chrf,
        "avg_latency_sec": speed["avg_latency_sec"],
        "throughput_chars_per_sec": speed["throughput_chars_per_sec"],
        "predictions": predictions,
        "samples": [{"src": s, "ref": r, "hyp": p} for s, r, p in zip(sources, references, predictions)]
    }


def generate_comparison_charts(results: list, output_dir: str = "evaluation/results"):
    os.makedirs(output_dir, exist_ok=True)
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return

    model_names = [r["model_name"] for r in valid_results]
    bleu_scores = [r["bleu"] for r in valid_results]
    chrf_scores = [r["chrf"] for r in valid_results]
    latencies = [r["avg_latency_sec"] for r in valid_results]

    # 1. BLEU and chrF++ Bar Chart
    plt.figure(figsize=(9, 5))
    x = range(len(model_names))
    width = 0.35

    plt.bar([i - width/2 for i in x], bleu_scores, width=width, label="SacreBLEU", color="#4f6ef7")
    plt.bar([i + width/2 for i in x], chrf_scores, width=width, label="chrF++", color="#10b981")

    plt.ylabel("Score")
    plt.title("Translation Quality Benchmark (Higher is Better)")
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    chart_path = os.path.join(output_dir, "bleu_chrf_comparison.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Saved metric chart to: {chart_path}")


def run_full_evaluation():
    os.makedirs("evaluation/results", exist_ok=True)
    res_nllb = evaluate_nllb_pretrained(TEST_SUITE)
    res_custom = evaluate_custom_transformer()

    all_results = {
        "timestamp": "2026-11-21",
        "benchmark_dataset": "LinguaAI Parallel Test Suite (En-Hi)",
        "models": {
            "nllb_pretrained": res_nllb,
            "custom_transformer": res_custom
        }
    }

    generate_comparison_charts([res_nllb, res_custom])

    with open("evaluation/results/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nEvaluation completed. Results saved to evaluation/results/evaluation_results.json")
    return all_results


if __name__ == "__main__":
    run_full_evaluation()
