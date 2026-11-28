"""
FastAPI REST API Routes for TranslateMe
Exclusively powered by Custom Multilingual PyTorch Transformer built from scratch.
"""

import os
import json
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from backend.app.core.config import settings
from backend.app.core.device import get_device_info
from backend.app.database.session import get_db
from backend.app.models.history import TranslationHistory
from backend.app.schemas.translation import (
    TranslateRequest,
    TranslateResponse,
    AttentionRequest,
    LanguageInfo,
    DetectLanguageRequest,
    DetectLanguageResponse,
    HistoryItem,
    ModelInfoResponse,
    AnalyticsResponse,
    DocumentTranslateResponse,
)
from backend.app.services.custom_multilingual_service import custom_multilingual_engine, CODE_MAP, REVERSE_MAP
from backend.app.services.language_codes import get_supported_languages
from backend.app.services.language_detector import detect_language
from backend.app.services.document_service import extract_text_from_file, translate_document
from ml_lab.custom_transformer.modern_transformer import ModernSeq2SeqTransformer

router = APIRouter()


def _to_history_item(item: TranslationHistory) -> HistoryItem:
    """Maps a SQLAlchemy TranslationHistory row to a timezone-aware HistoryItem schema."""
    return HistoryItem(
        id=item.id,
        original_text=item.source_text,
        translated_text=item.translated_text,
        source_language=item.source_language,
        source_language_name=item.source_language_name,
        target_language=item.target_language,
        target_language_name=item.target_language_name,
        detected_language=item.detected_language,
        inference_time=item.inference_time,
        model_name=item.model_name,
        created_at=item.created_at,
    )


def _downsample_loss_curve(train_losses: List[float], val_losses: List[float], global_step: int | None, is_modern: bool) -> Dict[str, Any]:
    """Builds aligned step/epoch progression points from real recorded checkpoint losses."""
    if not train_losses and not val_losses:
        return {"points": [], "x_axis_label": "Step", "summary": {}}

    n_train = len(train_losses)
    n_val = len(val_losses)
    total_steps = int(global_step) if global_step and global_step > 0 else max(n_train, n_val, 1)

    if is_modern:
        # Map step-level training losses & mid-run validation snapshots onto actual optimizer steps
        points: List[Dict[str, Any]] = []
        max_samples = 140
        step_stride = max(1, n_train // max_samples) if n_train > max_samples else 1
        sampled_indices = list(range(0, n_train, step_stride))
        if n_train > 0 and sampled_indices[-1] != n_train - 1:
            sampled_indices.append(n_train - 1)

        val_step_positions = (
            [round((i + 1) * total_steps / n_val) for i in range(n_val)]
            if n_val > 0 else []
        )

        # Build step -> val_loss map and merge sampled train points + all val points
        merged: Dict[int, Dict[str, Any]] = {}
        for idx in sampled_indices:
            step_x = max(1, round((idx + 1) * total_steps / max(1, n_train)))
            merged.setdefault(step_x, {"step": step_x})["train_loss"] = round(train_losses[idx], 4)

        for val_idx, step_x in enumerate(val_step_positions):
            step_x = max(1, step_x)
            entry = merged.setdefault(step_x, {"step": step_x})
            entry["val_loss"] = round(val_losses[val_idx], 4)
            # Attach closest train loss if missing at exact validation step
            if "train_loss" not in entry and n_train > 0:
                nearest_idx = min(n_train - 1, max(0, round(step_x * n_train / total_steps) - 1))
                entry["train_loss"] = round(train_losses[nearest_idx], 4)

        points = [merged[k] for k in sorted(merged.keys())]
        x_axis_label = "Optimizer Step"
    else:
        max_len = max(n_train, n_val)
        points = []
        for idx in range(max_len):
            row: Dict[str, Any] = {"step": idx + 1}
            if idx < n_train:
                row["train_loss"] = round(train_losses[idx], 4)
            if idx < n_val:
                row["val_loss"] = round(val_losses[idx], 4)
            points.append(row)
        x_axis_label = "Epoch"

    start_train = round(train_losses[0], 4) if train_losses else None
    end_train = round(train_losses[-1], 4) if train_losses else None
    start_val = round(val_losses[0], 4) if val_losses else None
    end_val = round(val_losses[-1], 4) if val_losses else None
    best_val = round(min(val_losses), 4) if val_losses else None
    reduction_pct = (
        round(((start_train - end_train) / start_train) * 100.0, 1)
        if (start_train and end_train is not None and start_train > 0)
        else None
    )

    return {
        "points": points,
        "x_axis_label": x_axis_label,
        "summary": {
            "start_train_loss": start_train,
            "end_train_loss": end_train,
            "start_val_loss": start_val,
            "end_val_loss": end_val,
            "best_val_loss": best_val,
            "loss_reduction_pct": reduction_pct,
            "train_points_count": n_train,
            "val_points_count": n_val,
        },
    }


@router.get("/health")
def health_check():
    """Health check endpoint displaying system and device status."""
    device_info = get_device_info()
    return {
        "status": "healthy",
        "service": "TranslateMe Custom Multilingual Translation API",
        "model_engine": custom_multilingual_engine.get_display_model_name()
        if custom_multilingual_engine.is_loaded
        else "Custom PyTorch Transformer (From Scratch)",
        "model_loaded": custom_multilingual_engine.is_loaded,
        "device": device_info,
    }


@router.get("/languages", response_model=List[LanguageInfo])
def list_languages():
    """Returns supported languages verified against the active tokenizer control tokens."""
    if not custom_multilingual_engine.is_loaded:
        custom_multilingual_engine.load_model()

    tok = custom_multilingual_engine.tokenizer
    langs = []
    for meta in get_supported_languages():
        if tok is not None:
            if tok.tokenizer.token_to_id(meta["control_token"]) is None:
                continue
        langs.append(
            LanguageInfo(
                code=meta["code"],
                name=meta["name"],
                native_name=meta["native_name"],
                script=meta["script"],
                iso=meta.get("iso"),
            )
        )
    return langs


@router.post("/detect-language", response_model=DetectLanguageResponse)
def detect_lang(req: DetectLanguageRequest):
    """Detects the language of the provided text."""
    code, name, conf = detect_language(req.text)
    norm_code = CODE_MAP.get(code, "en")
    return DetectLanguageResponse(
        detected_code=norm_code,
        detected_name=REVERSE_MAP.get(norm_code, name),
        confidence=conf,
    )


@router.post("/translate", response_model=TranslateResponse)
def translate_text(req: TranslateRequest, db: Session = Depends(get_db)):
    """Translates text between any supported language pair using our custom Transformer."""
    try:
        result = custom_multilingual_engine.translate(
            text=req.text,
            source_lang=req.source_language,
            target_lang=req.target_language,
            num_beams=req.num_beams or 1,
            return_alternatives=req.return_alternatives or False,
        )

        history_entry = TranslationHistory(
            source_text=result["original_text"],
            translated_text=result["translated_text"],
            source_language=result["source_language"],
            source_language_name=result["source_language_name"],
            target_language=result["target_language"],
            target_language_name=result["target_language_name"],
            detected_language=result["detected_language"],
            inference_time=result["inference_time"],
            model_name=result["model_name"],
        )
        db.add(history_entry)
        db.commit()

        return TranslateResponse(**result)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.post("/lab/attention")
def extract_live_attention(req: AttentionRequest):
    """
    Extracts real live Cross-Attention weights from the custom Transformer
    for interactive heatmaps in the Deep Learning Lab.
    """
    try:
        result = custom_multilingual_engine.translate(
            text=req.text,
            source_lang=req.source_lang,
            target_lang=req.target_lang,
            num_beams=1,
            return_attention=True,
        )
        return {
            "source_text": req.text,
            "translated_text": result["translated_text"],
            "source_language": result["source_language"],
            "target_language": result["target_language"],
            "source_tokens": result.get("source_display_tokens", result.get("source_subwords", [])),
            "target_tokens": result.get("target_display_tokens", result.get("target_subwords", [])),
            "raw_source_tokens": result.get("source_subwords", []),
            "raw_target_tokens": result.get("target_subwords", []),
            "attention_matrix": result.get("cross_attention_matrix", []),
            "decoder_layer": result.get("decoder_layer"),
            "num_heads": result.get("num_heads"),
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=List[HistoryItem])
def get_translation_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Fetches user translation history ordered by most recent."""
    items = (
        db.query(TranslationHistory)
        .order_by(desc(TranslationHistory.created_at), desc(TranslationHistory.id))
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_to_history_item(item) for item in items]


@router.delete("/history/{item_id}")
def delete_history_item(item_id: int, db: Session = Depends(get_db)):
    """Deletes a specific history record."""
    item = db.query(TranslationHistory).filter(TranslationHistory.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="History entry not found")
    db.delete(item)
    db.commit()
    return {"message": "Entry successfully deleted", "id": item_id}


@router.delete("/history")
def clear_all_history(db: Session = Depends(get_db)):
    """Clears all translation history."""
    db.query(TranslationHistory).delete()
    db.commit()
    return {"message": "All history cleared"}


@router.post("/document/translate", response_model=DocumentTranslateResponse)
async def translate_doc_endpoint(
    file: UploadFile = File(...),
    source_language: str = Form(...),
    target_language: str = Form(...),
):
    """
    Extracts text from uploaded PDF or TXT document and translates paragraph by paragraph
    using our custom Transformer, preserving structure and enforcing upload size limits.
    """
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file size ({len(content) / (1024 * 1024):.1f}MB) exceeds the 10MB limit.",
        )

    max_paragraphs = 50
    filename = file.filename or "uploaded_file"
    paragraphs = extract_text_from_file(file, content)

    try:
        full_translated, elapsed = translate_document(
            paragraphs=paragraphs,
            source_lang=source_language,
            target_lang=target_language,
            max_paragraphs=max_paragraphs,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    norm_src = custom_multilingual_engine.normalize_code(source_language)
    norm_tgt = custom_multilingual_engine.normalize_code(target_language)

    return DocumentTranslateResponse(
        filename=filename,
        total_paragraphs=len(paragraphs),
        translated_paragraphs=min(len(paragraphs), max_paragraphs),
        truncated=len(paragraphs) > max_paragraphs,
        translated_text=full_translated,
        inference_time=elapsed,
        source_language=norm_src,
        target_language=norm_tgt,
    )


@router.get("/analytics", response_model=AnalyticsResponse)
def get_analytics(db: Session = Depends(get_db)):
    """Aggregates metrics for the Analytics dashboard directly from SQLite."""
    total = db.query(TranslationHistory).count()
    avg_latency = db.query(func.avg(TranslationHistory.inference_time)).scalar() or 0.0

    src_counts = (
        db.query(
            TranslationHistory.source_language_name,
            func.count(TranslationHistory.id).label("count"),
        )
        .group_by(TranslationHistory.source_language_name)
        .order_by(desc("count"))
        .limit(10)
        .all()
    )
    top_sources = [{"language": row[0] or "Unknown", "count": row[1]} for row in src_counts]

    tgt_counts = (
        db.query(
            TranslationHistory.target_language_name,
            func.count(TranslationHistory.id).label("count"),
        )
        .group_by(TranslationHistory.target_language_name)
        .order_by(desc("count"))
        .limit(10)
        .all()
    )
    top_targets = [{"language": row[0] or "Unknown", "count": row[1]} for row in tgt_counts]

    pair_counts = (
        db.query(
            TranslationHistory.source_language,
            TranslationHistory.target_language,
            func.count(TranslationHistory.id).label("count"),
        )
        .group_by(TranslationHistory.source_language, TranslationHistory.target_language)
        .order_by(desc("count"))
        .limit(12)
        .all()
    )
    top_language_pairs = [
        {"pair": f"{row[0].upper()} → {row[1].upper()}", "count": row[2]}
        for row in pair_counts
    ]

    distinct_src = {r[0] for r in db.query(TranslationHistory.source_language).distinct().all() if r[0]}
    distinct_tgt = {r[0] for r in db.query(TranslationHistory.target_language).distinct().all() if r[0]}
    unique_languages_count = len(distinct_src | distinct_tgt)
    active_directions_count = len(pair_counts)

    recent_items = (
        db.query(TranslationHistory)
        .order_by(desc(TranslationHistory.created_at), desc(TranslationHistory.id))
        .limit(5)
        .all()
    )
    recent_activity = [_to_history_item(i) for i in recent_items]

    return AnalyticsResponse(
        total_translations=total,
        average_inference_time_ms=round(avg_latency * 1000, 1),
        unique_languages_count=unique_languages_count,
        active_directions_count=active_directions_count,
        top_source_languages=top_sources,
        top_target_languages=top_targets,
        top_language_pairs=top_language_pairs,
        recent_activity=recent_activity,
    )


@router.get("/lab/info")
def get_deep_learning_lab_info():
    """Provides real architectural specs, live checkpoint training curves, and real held-out evaluation data."""
    if not custom_multilingual_engine.is_loaded:
        custom_multilingual_engine.load_model()

    model = custom_multilingual_engine.model
    meta = custom_multilingual_engine.checkpoint_meta
    cfg = custom_multilingual_engine.config
    is_modern = isinstance(model, ModernSeq2SeqTransformer)
    n_params = custom_multilingual_engine.get_model_parameter_count()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint is not loaded. Ensure custom_multilingual_transformer.pt exists.",
        )

    if is_modern:
        mcfg = model.config
        enc_layers = mcfg.num_encoder_layers
        dec_layers = mcfg.num_decoder_layers
        n_heads = mcfg.n_heads
        d_model = mcfg.d_model
        d_ff = mcfg.d_ff
        vocab_size = mcfg.vocab_size
        use_rope = mcfg.use_rope
        use_qk_norm = mcfg.use_qk_norm
        use_layerscale = mcfg.use_layerscale
        soft_cap = mcfg.soft_cap
        share_embeddings = mcfg.share_embeddings
    else:
        enc_layers = len(model.encoder.layers)
        dec_layers = len(model.decoder.layers)
        n_heads = model.encoder.layers[0].self_attn.n_heads
        d_model = model.d_model
        d_ff = model.encoder.layers[0].feed_forward.w_1.out_features
        vocab_size = model.generator.proj.out_features
        use_rope = False
        use_qk_norm = False
        use_layerscale = False
        soft_cap = None
        share_embeddings = model.share_embeddings

    train_losses = meta.get("train_loss_history", [])
    val_losses = meta.get("val_loss_history", [])
    if not train_losses and os.path.exists(settings.TRAINING_HISTORY_PATH):
        try:
            with open(settings.TRAINING_HISTORY_PATH, "r", encoding="utf-8") as f:
                hist_json = json.load(f)
                train_losses = hist_json.get("train_loss", [])
                val_losses = hist_json.get("val_loss", [])
        except Exception:
            pass

    curve_data = _downsample_loss_curve(train_losses, val_losses, meta.get("global_step"), is_modern=is_modern)

    eval_report = None
    eval_summary = None
    if os.path.exists(settings.EVAL_REPORT_PATH):
        try:
            with open(settings.EVAL_REPORT_PATH, "r", encoding="utf-8") as f:
                eval_report = json.load(f)
            if isinstance(eval_report, dict) and eval_report:
                vals = list(eval_report.values())
                eval_summary = {
                    "directions_count": len(vals),
                    "test_sentence_count_per_dir": vals[0].get("test_sentence_count", 0),
                    "avg_bleu": round(sum(v.get("bleu", 0.0) for v in vals) / len(vals), 2),
                    "avg_chrf": round(sum(v.get("chrf", 0.0) for v in vals) / len(vals), 2),
                    "avg_latency_sec": round(sum(v.get("avg_latency_sec", 0.0) for v in vals) / len(vals), 3),
                    "decoding_algorithm": vals[0].get("decoding_algorithm", "Beam Search"),
                    "dataset_source": "Independent Held-Out 4-Way Parallel Test Records (0% Training Overlap)",
                }
        except Exception:
            pass

    if is_modern:
        components = [
            {
                "name": "QK-Norm & Fused SDPA",
                "formula": "Attention(Q,K,V) = softmax((RMSNorm(Q) RMSNorm(K)^T) / sqrt(d_k) + Mask) V",
                "description": "Applies RMSNorm per head to Q and K before dot-product scoring to prevent attention logit growth in deep stacks.",
            },
            {
                "name": "Rotary Position Embeddings (RoPE)",
                "formula": "R_m(x_2i, x_2i+1) = (x_2i cos(m*theta_i) - x_2i+1 sin(m*theta_i), x_2i sin(m*theta_i) + x_2i+1 cos(m*theta_i))",
                "description": "Rotates query and key pairs by relative token distance m without absolute position embedding lookup tables.",
            },
            {
                "name": "SwiGLU Gated Feed-Forward",
                "formula": "SwiGLU(x) = (SiLU(x W_gate) * (x W_up)) W_down",
                "description": "Bilinear gated activation with 3 projection matrices replacing standard ReLU 2-layer FFNs.",
            },
            {
                "name": "Pre-LN RMSNorm, LayerScale & Soft-Cap",
                "formula": "x' = x + gamma * Sublayer(RMSNorm(x)),  logits = cap * tanh(z / cap)",
                "description": "Pre-normalization residual scaling combined with Gemma-2 style logit soft-capping (cap=30.0).",
            },
        ]
    else:
        components = [
            {
                "name": "Multi-Head Attention",
                "formula": "MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W^O",
                "description": "Jointly attends to information from multiple representation subspaces.",
            },
            {
                "name": "Scaled Dot-Product Attention",
                "formula": "Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V",
                "description": "Calculates normalized similarity weights between query and key tokens, scaled by sqrt(d_k).",
            },
            {
                "name": "Sinusoidal Positional Encoding",
                "formula": "PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))",
                "description": "Injects sequence position awareness into token vectors.",
            },
            {
                "name": "Residual Connections & LayerNorm",
                "formula": "Output = LayerNorm(x + Sublayer(x))",
                "description": "Enables smooth gradient flow across deep encoder and decoder stacks.",
            },
        ]

    return {
        "custom_transformer": {
            "name": custom_multilingual_engine.get_display_model_name(),
            "is_modern": is_modern,
            "encoder_layers": enc_layers,
            "decoder_layers": dec_layers,
            "attention_heads": n_heads,
            "head_dim": d_model // max(1, n_heads),
            "d_model": d_model,
            "d_ff": d_ff,
            "parameters": n_params,
            "shared_vocab_size": vocab_size,
            "use_rope": use_rope,
            "use_qk_norm": use_qk_norm,
            "use_layerscale": use_layerscale,
            "soft_cap": soft_cap,
            "share_embeddings": share_embeddings,
            "epochs_completed": meta.get("epochs_completed"),
            "global_step": meta.get("global_step"),
            "best_val_loss": round(meta["best_val_loss"], 4) if meta.get("best_val_loss") is not None else None,
            "checkpoint_size_mb": meta.get("checkpoint_size_mb"),
            "weight_dtype": meta.get("weight_dtype"),
            "device": str(custom_multilingual_engine.device),
            "training_curve": curve_data["points"],
            "training_x_label": curve_data["x_axis_label"],
            "training_summary": curve_data["summary"],
        },
        "evaluation_benchmark": eval_report,
        "evaluation_summary": eval_summary,
        "transformer_components": components,
    }


@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_specifications():
    """Returns real technical specifications read directly from the loaded custom model & checkpoint."""
    if not custom_multilingual_engine.is_loaded:
        custom_multilingual_engine.load_model()

    model = custom_multilingual_engine.model
    meta = custom_multilingual_engine.checkpoint_meta
    cfg = custom_multilingual_engine.config
    n_params = custom_multilingual_engine.get_model_parameter_count()
    langs_count = len(get_supported_languages())

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint is not loaded. Ensure custom_multilingual_transformer.pt exists.",
        )

    if isinstance(model, ModernSeq2SeqTransformer):
        mcfg = model.config
        head_dim = mcfg.d_model // max(1, mcfg.n_heads)
        best_val = meta.get("best_val_loss")
        gstep = meta.get("global_step")
        epochs_done = meta.get("epochs_completed")
        ckpt_mb = meta.get("checkpoint_size_mb")
        dtype = meta.get("weight_dtype", "float32")

        limitations = [
            f"Supports {langs_count} languages ({langs_count * (langs_count - 1)} translation directions) via shared <2lang> decoder control tokens",
            f"Vocabulary size is {mcfg.vocab_size:,} ByteLevel BPE subwords trained from scratch",
            f"Max configured context length: {mcfg.max_len:,} tokens with dynamic RoPE extrapolation",
            f"Loaded weights: {ckpt_mb} MB ({dtype}) checkpoint"
            + (f" at step {gstep:,}" if gstep else "")
            + (f" (best val loss {best_val:.4f})" if best_val is not None else ""),
        ]

        return ModelInfoResponse(
            model_name=custom_multilingual_engine.get_display_model_name(),
            model_type="Modern Asymmetric Seq2Seq Transformer (RoPE + SwiGLU + Pre-LN RMSNorm + QK-Norm + LayerScale)"
            if mcfg.num_encoder_layers != mcfg.num_decoder_layers
            else "Modern Seq2Seq Transformer (RoPE + SwiGLU + Pre-LN RMSNorm + QK-Norm + LayerScale)",
            num_parameters=f"{n_params:,} ({n_params / 1e6:.2f}M)",
            parameters=f"{n_params:,} ({n_params / 1e6:.2f}M)",
            layers=mcfg.num_encoder_layers,
            encoder_layers=mcfg.num_encoder_layers,
            decoder_layers=mcfg.num_decoder_layers,
            attention_heads=mcfg.n_heads,
            head_dim=head_dim,
            d_model=mcfg.d_model,
            d_ff=mcfg.d_ff,
            max_len=mcfg.max_len,
            dropout=mcfg.dropout,
            vocab_size=f"{mcfg.vocab_size:,} subwords",
            tokenizer_type="Byte-Pair Encoding (BPE) Multilingual Subword Tokenizer (ByteLevel + NFC)",
            supported_languages_count=langs_count,
            framework="Pure PyTorch (Pre-LN RMSNorm, RoPE, SwiGLU, QK-Norm, SDPA)",
            device=str(custom_multilingual_engine.device),
            license="MIT Open Source (Own Independent Implementation)",
            attribution="Trained completely from scratch in PyTorch. Zero pretrained translation weights.",
            limitations=limitations,
            is_modern=True,
            weight_tying="3-Way Shared Embeddings (Enc + Dec + LM Head)" if mcfg.share_embeddings else "Untied Projections",
            use_rope=mcfg.use_rope,
            use_qk_norm=mcfg.use_qk_norm,
            use_layerscale=mcfg.use_layerscale,
            soft_cap=mcfg.soft_cap,
            checkpoint_size_mb=ckpt_mb,
            weight_dtype=dtype,
            epochs_completed=epochs_done,
            global_step=gstep,
            best_val_loss=round(best_val, 4) if best_val is not None else None,
            training_details={
                "epochs_completed": epochs_done,
                "global_step": gstep,
                "best_val_loss": round(best_val, 4) if best_val is not None else None,
                "checkpoint_size_mb": ckpt_mb,
                "weight_dtype": dtype,
                "train_history_points": len(meta.get("train_loss_history", [])),
                "val_history_points": len(meta.get("val_loss_history", [])),
                "soft_cap": mcfg.soft_cap,
                "kv_cache_decoding": "Enabled (O(L) Greedy & Batched Beam Search)",
            },
            references=[
                {"title": "Attention Is All You Need", "citation": "Vaswani et al. (NeurIPS 2017)"},
                {"title": "RoFormer: Enhanced Transformer with Rotary Position Embedding", "citation": "Su et al. (2021)"},
                {"title": "Root Mean Square Layer Normalization", "citation": "Zhang & Sennrich (NeurIPS 2019)"},
                {"title": "GLU Variants Improve Transformer (SwiGLU)", "citation": "Shazeer (2020)"},
                {"title": "Scaling Vision Transformers to 22B (QK-Norm)", "citation": "Dehghani et al. (ICML 2023)"},
            ],
        )

    enc_layers = len(model.encoder.layers)
    dec_layers = len(model.decoder.layers)
    n_heads = model.encoder.layers[0].self_attn.n_heads
    d_model = model.d_model
    d_ff = model.encoder.layers[0].feed_forward.w_1.out_features
    vocab_num = model.generator.proj.out_features
    best_val = meta.get("best_val_loss")

    return ModelInfoResponse(
        model_name=custom_multilingual_engine.get_display_model_name(),
        model_type="Seq2Seq Transformer (Encoder-Decoder)",
        num_parameters=f"{n_params:,} ({n_params / 1e6:.2f}M)",
        parameters=f"{n_params:,} ({n_params / 1e6:.2f}M)",
        layers=enc_layers,
        encoder_layers=enc_layers,
        decoder_layers=dec_layers,
        attention_heads=n_heads,
        head_dim=d_model // max(1, n_heads),
        d_model=d_model,
        d_ff=d_ff,
        max_len=cfg.get("max_len", model.encoder.embedding.pe.pe.size(1)),
        dropout=cfg.get("dropout", 0.1),
        vocab_size=f"{vocab_num:,} subwords",
        tokenizer_type="Byte-Pair Encoding (BPE) Multilingual Subword Tokenizer (ByteLevel)",
        supported_languages_count=langs_count,
        framework="Pure PyTorch (Vaswani et al., 2017)",
        device=str(custom_multilingual_engine.device),
        license="MIT Open Source (Own Independent Implementation)",
        attribution="Trained completely from scratch with PyTorch. No pretrained translation weights.",
        limitations=[
            f"Supports {langs_count} languages via shared BPE vocabulary ({vocab_num:,} subwords)",
            f"Loaded weights: {meta.get('checkpoint_size_mb')} MB ({meta.get('weight_dtype')})",
        ],
        is_modern=False,
        weight_tying="3-Way Shared Embeddings" if model.share_embeddings else "Untied Projections",
        checkpoint_size_mb=meta.get("checkpoint_size_mb"),
        weight_dtype=meta.get("weight_dtype"),
        epochs_completed=meta.get("epochs_completed"),
        global_step=meta.get("global_step"),
        best_val_loss=round(best_val, 4) if best_val is not None else None,
        references=[
            {"title": "Attention Is All You Need", "citation": "Vaswani et al. (NeurIPS 2017)"},
        ],
    )
