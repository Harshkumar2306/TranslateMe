"""
Scientific Evaluation Pipeline for Custom Multilingual Transformer
Evaluates on independent held-out test sets across all 12 language directions (en, hi, ta, te).
Computes genuine SacreBLEU, chrF++, Latency, Throughput, Empty Rate, and Repetition Rate without data leakage.
Exports JSON, CSV, Markdown report, training history JSON, and evaluation chart.
"""

import os
import csv
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from typing import Dict, Any, List

from ml_lab.custom_transformer import Seq2SeqTransformer
from ml_lab.custom_transformer.modern_transformer import ModernSeq2SeqTransformer, ModernTransformerConfig
from ml_lab.tokenizer import MultilingualSubwordTokenizer
from ml_lab.evaluation.metrics import compute_sacrebleu, compute_chrf, benchmark_translation_speed
from ml_lab.datasets.curated_corpus import get_held_out_test_records


def load_custom_multilingual_model(
    checkpoint_path: str = "ml_lab/checkpoints/custom_multilingual_transformer.pt",
    device: str = "cpu"
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tok_path = ckpt.get("tokenizer_path", "ml_lab/tokenizer/multilingual_tokenizer.json")
    if not os.path.exists(tok_path):
        tok_path = "ml_lab/tokenizer/multilingual_tokenizer.json"
    tokenizer = MultilingualSubwordTokenizer.load(tok_path)
    cfg = ckpt.get("config", {})
    n_enc = cfg.get("num_encoder_layers", cfg.get("num_layers", 6))
    n_dec = cfg.get("num_decoder_layers", cfg.get("num_layers", 6))

    is_modern = (
        cfg.get("use_rope", False)
        or any("rope" in k for k in ckpt.get("model_state_dict", {}).keys())
        or cfg.get("model_type") == "modern"
    )
    if is_modern:
        modern_cfg = ModernTransformerConfig(
            vocab_size=cfg.get("vocab_size", tokenizer.vocab_size),
            d_model=cfg.get("d_model", 768),
            n_heads=cfg.get("n_heads", 12),
            num_encoder_layers=n_enc,
            num_decoder_layers=n_dec,
            d_ff=cfg.get("d_ff", 2048),
            dropout=cfg.get("dropout", 0.1),
            pad_idx=tokenizer.pad_id,
            bos_idx=getattr(tokenizer, "bos_id", tokenizer.sos_id),
            eos_idx=tokenizer.eos_id,
            use_rope=cfg.get("use_rope", True),
            use_qk_norm=cfg.get("use_qk_norm", True),
            share_embeddings=cfg.get("share_embeddings", True),
            use_layerscale=cfg.get("use_layerscale", True),
            layerscale_init=cfg.get("layerscale_init", 0.1),
            soft_cap=cfg.get("soft_cap", 30.0)
        )
        model = ModernSeq2SeqTransformer(modern_cfg).to(device)
    else:
        model = Seq2SeqTransformer(
            src_vocab_size=cfg.get("vocab_size", tokenizer.vocab_size),
            tgt_vocab_size=cfg.get("vocab_size", tokenizer.vocab_size),
            d_model=cfg.get("d_model", 256),
            n_heads=cfg.get("n_heads", 4),
            num_encoder_layers=n_enc,
            num_decoder_layers=n_dec,
            d_ff=cfg.get("d_ff", 1024),
            share_embeddings=cfg.get("share_embeddings", True),
            pre_ln=cfg.get("pre_ln", False)
        ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer, ckpt


def build_script_mask(tokenizer: MultilingualSubwordTokenizer, target_lang: str, device: str) -> torch.Tensor:
    script_map = {"hi": "Devanagari", "ta": "Tamil", "te": "Telugu", "en": "Latin"}
    target_script = script_map.get(target_lang)
    disallowed = torch.zeros(tokenizer.vocab_size, dtype=torch.bool, device=device)
    if not target_script:
        return disallowed
    for i in range(tokenizer.vocab_size):
        if i in (tokenizer.pad_id, tokenizer.sos_id, tokenizer.eos_id, tokenizer.unk_id):
            continue
        tok_str = tokenizer.tokenizer.id_to_token(i) or ""
        if tok_str.startswith("<2"):
            continue
        decoded_str = tokenizer.decode([i]) or tok_str
        for ch in decoded_str:
            cp = ord(ch)
            if 0x0900 <= cp <= 0x097F and target_script != "Devanagari":
                disallowed[i] = True
                break
            elif 0x0B80 <= cp <= 0x0BFF and target_script != "Tamil":
                disallowed[i] = True
                break
            elif 0x0C00 <= cp <= 0x0C7F and target_script != "Telugu":
                disallowed[i] = True
                break
    return disallowed


def evaluate_direction(
    model,
    tokenizer: MultilingualSubwordTokenizer,
    src_lang: str,
    tgt_lang: str,
    test_records: list,
    device: str = "cpu",
    use_beam_search: bool = True,
    beam_size: int = 2
) -> Dict[str, Any]:
    sources = [getattr(r, src_lang) for r in test_records]
    references = [getattr(r, tgt_lang) for r in test_records]
    tgt_control_id = tokenizer.get_control_token_id(tgt_lang)
    is_modern = isinstance(model, ModernSeq2SeqTransformer)
    max_len = 96 if is_modern else 64
    ngram_size = 3 if is_modern else 2
    rep_penalty = 1.15 if is_modern else 1.25
    script_mask = build_script_mask(tokenizer, tgt_lang, device)

    def translate_fn(text: str) -> str:
        raw_ids = tokenizer.encode(text, add_special_tokens=False)
        if is_modern:
            src_tokens = [tokenizer.sos_id] + raw_ids + [tokenizer.eos_id]
        else:
            src_tokens = [tokenizer.sos_id, tgt_control_id] + raw_ids + [tokenizer.eos_id]
        src_t = torch.tensor([src_tokens], dtype=torch.long, device=device)
        prefix = [tgt_control_id] if is_modern else None

        with torch.no_grad():
            if use_beam_search:
                beams = model.beam_search_decode(
                    src_t,
                    beam_size=beam_size,
                    max_len=max_len,
                    sos_idx=tokenizer.sos_id,
                    eos_idx=tokenizer.eos_id,
                    prefix_tokens=prefix,
                    disallowed_mask=script_mask,
                    no_repeat_ngram_size=ngram_size,
                    repetition_penalty=rep_penalty
                )
                gen_ids = beams[0][0][0].tolist()
            else:
                out = model.greedy_decode(
                    src_t,
                    sos_idx=tokenizer.sos_id,
                    eos_idx=tokenizer.eos_id,
                    prefix_tokens=prefix,
                    max_len=max_len,
                    repetition_penalty=rep_penalty,
                    disallowed_mask=script_mask,
                    no_repeat_ngram_size=ngram_size
                )
                gen_ids = out[0].tolist()
        return tokenizer.decode(gen_ids).strip()

    predictions = [translate_fn(s) for s in sources]
    bleu = compute_sacrebleu(predictions, references)
    chrf = compute_chrf(predictions, references)
    speed = benchmark_translation_speed(translate_fn, sources)

    avg_src_len = round(sum(len(s.split()) for s in sources) / max(1, len(sources)), 1)
    avg_gen_len = round(sum(len(p.split()) for p in predictions) / max(1, len(predictions)), 1)
    empty_count = sum(1 for p in predictions if not p.strip())
    empty_rate = round(empty_count / max(1, len(predictions)), 3)

    def calc_repetition(p: str) -> float:
        words = p.split()
        if len(words) <= 1:
            return 0.0
        repeats = sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1])
        return repeats / (len(words) - 1)

    avg_rep_rate = round(sum(calc_repetition(p) for p in predictions) / max(1, len(predictions)), 3)
    algo_desc = (
        f"Beam Search (k={beam_size}, length_penalty=0.6, rep_penalty={rep_penalty}, ngram={ngram_size})"
        if use_beam_search else f"Greedy KV-Cache (rep_penalty={rep_penalty}, ngram={ngram_size})"
    )

    return {
        "direction": f"{src_lang} -> {tgt_lang}",
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "test_sentence_count": len(sources),
        "bleu": round(bleu, 2),
        "chrf": round(chrf, 2),
        "avg_latency_sec": speed["avg_latency_sec"],
        "throughput_chars_per_sec": speed["throughput_chars_per_sec"],
        "avg_source_word_len": avg_src_len,
        "avg_target_word_len": avg_gen_len,
        "empty_rate": empty_rate,
        "repetition_rate": avg_rep_rate,
        "decoding_algorithm": algo_desc,
        "samples": [
            {"source": s, "reference": r, "hypothesis": p}
            for s, r, p in zip(sources, references, predictions)
        ]
    }


def export_checkpoint_training_history(ckpt: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    """Exports real training & validation loss trajectories from the active checkpoint."""
    train_losses = ckpt.get("train_loss_history", [])
    val_losses = ckpt.get("val_loss_history", [])
    global_step = ckpt.get("global_step")
    epoch = ckpt.get("epoch")
    best_val_loss = ckpt.get("best_val_loss")

    history = {
        "global_step": global_step,
        "epochs_completed": (epoch + 1) if isinstance(epoch, int) else None,
        "best_val_loss": round(float(best_val_loss), 4) if best_val_loss is not None else None,
        "train_loss": [round(float(x), 4) for x in train_losses],
        "val_loss": [round(float(x), 4) for x in val_losses],
        "config": ckpt.get("config", {}),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "custom_multilingual_training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return history


def run_full_evaluation():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("\n========================================================")
    print("Evaluating Custom Multilingual Transformer on Held-Out Test Set")
    print(f"Device: {device}")
    print("========================================================")

    model, tokenizer, ckpt = load_custom_multilingual_model(device=device)
    test_records = get_held_out_test_records()
    langs = ["en", "hi", "ta", "te"]
    directions = [(s, t) for s in langs for t in langs if s != t]

    results = {}
    for src, tgt in directions:
        eval_res = evaluate_direction(model, tokenizer, src, tgt, test_records, device=device, use_beam_search=True, beam_size=2)
        print(f"[{eval_res['direction']}] SacreBLEU: {eval_res['bleu']:.2f} | chrF++: {eval_res['chrf']:.2f} | Latency: {eval_res['avg_latency_sec']}s")
        results[eval_res['direction']] = eval_res

    out_dir = "ml_lab/evaluation/results"
    os.makedirs(out_dir, exist_ok=True)

    # Export real checkpoint training history
    export_checkpoint_training_history(ckpt, out_dir)

    # 1. Save JSON report
    report_json = os.path.join(out_dir, "custom_multilingual_evaluation_report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 2. Save CSV summary
    report_csv = os.path.join(out_dir, "custom_multilingual_evaluation_report.csv")
    with open(report_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Direction", "SacreBLEU", "chrF++", "Avg Latency (s)", "Throughput (chars/s)", "Avg Source Len", "Avg Gen Len", "Empty Rate", "Repetition Rate", "Decoding"])
        for _, res in results.items():
            writer.writerow([
                res["direction"], res["bleu"], res["chrf"], res["avg_latency_sec"],
                res["throughput_chars_per_sec"], res["avg_source_word_len"],
                res["avg_target_word_len"], res["empty_rate"], res["repetition_rate"],
                res["decoding_algorithm"]
            ])

    # 3. Save Markdown report
    avg_bleu = round(sum(r["bleu"] for r in results.values()) / len(results), 2)
    avg_chrf = round(sum(r["chrf"] for r in results.values()) / len(results), 2)
    n_params = model.count_parameters()
    if isinstance(n_params, dict):
        n_params = n_params.get("trainable_parameters", 0)
    report_md = os.path.join(out_dir, "custom_multilingual_evaluation_report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Scientific Evaluation Report: Custom Multilingual Transformer\n\n")
        f.write(f"Evaluated on independent held-out 4-way test records ({len(test_records)} parallel records × {len(directions)} directions = {len(test_records) * len(directions)} evaluations, 0% training overlap).\n\n")
        f.write(f"- **Active Parameters**: {n_params:,} ({n_params / 1e6:.2f}M)\n")
        f.write(f"- **Device**: `{device}`\n")
        f.write(f"- **Mean SacreBLEU**: {avg_bleu:.2f} | **Mean chrF++**: {avg_chrf:.2f}\n")
        f.write(f"- **Evaluated At**: {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write("| Direction | SacreBLEU | chrF++ | Avg Latency (s) | Throughput (c/s) | Empty Rate | Repetition Rate |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for _, res in results.items():
            f.write(f"| **{res['direction']}** | {res['bleu']} | {res['chrf']} | {res['avg_latency_sec']}s | {res['throughput_chars_per_sec']} | {res['empty_rate']*100:.1f}% | {res['repetition_rate']*100:.1f}% |\n")
        f.write("\n\n### Sample Predictions\n\n")
        for _, res in results.items():
            f.write(f"#### {res['direction']}\n")
            for idx, samp in enumerate(res["samples"][:2], 1):
                f.write(f"- **Sample {idx}**:\n")
                f.write(f"  - Source: `{samp['source']}`\n")
                f.write(f"  - Reference: `{samp['reference']}`\n")
                f.write(f"  - Hypothesis: `{samp['hypothesis']}`\n")

    # 4. Save per-direction sample JSONL
    for _, res in results.items():
        fname = f"samples_{res['src_lang']}_{res['tgt_lang']}.jsonl"
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            for s in res["samples"]:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 5. Plot evaluation chart
    plot_evaluation_chart(results, os.path.join(out_dir, "multilingual_bleu_chrf.png"))
    print(f"\nAll scientific evaluation reports (JSON, CSV, MD, JSONL, PNG) saved to: {out_dir}")
    return results


def plot_evaluation_chart(results: Dict[str, Any], output_path: str):
    labels = list(results.keys())
    bleu_scores = [results[d]["bleu"] for d in labels]
    chrf_scores = [results[d]["chrf"] for d in labels]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - width / 2 for i in x], bleu_scores, width, label="SacreBLEU", color="#7c3aed")
    ax.bar([i + width / 2 for i in x], chrf_scores, width, label="chrF++", color="#10b981")

    ax.set_ylabel("Metric Score (0-100)")
    ax.set_title("Custom Multilingual Transformer Evaluation Across 12 Directions")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    run_full_evaluation()
