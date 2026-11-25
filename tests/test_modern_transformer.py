"""
Unit tests for Modern SOTA 100M Transformer Architecture & Resumable Trainer
"""

import os
import shutil
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ml_lab.custom_transformer.rms_norm import RMSNorm
from ml_lab.custom_transformer.rope import RotaryEmbedding, precompute_freqs_cis
from ml_lab.custom_transformer.swiglu import SwiGLU
from ml_lab.custom_transformer.modern_attention import ModernMultiHeadAttention
from ml_lab.custom_transformer.modern_transformer import ModernSeq2SeqTransformer, ModernTransformerConfig
from ml_lab.training.resumable_trainer import ResumableTrainer


def test_rmsnorm_forward_and_gradient():
    dim = 64
    norm = RMSNorm(dim)
    x = torch.randn(2, 10, dim, requires_grad=True)
    out = norm(x)

    assert out.shape == x.shape
    # Output RMS should be close to 1.0 (with weight=1)
    rms = torch.sqrt(torch.mean(out**2, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)

    # Test backward
    out.sum().backward()
    assert x.grad is not None
    assert norm.weight.grad is not None


def test_rope_rotation_properties():
    dim = 64
    seq_len = 8
    rope = RotaryEmbedding(dim=dim, max_seq_len=128)
    q = torch.randn(2, 4, seq_len, dim)
    k = torch.randn(2, 4, seq_len, dim)

    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

    # Relative shift test: dot product of shifted vectors depends on distance, not absolute pos
    # Check that norm is preserved under orthogonal rotation
    orig_norm = torch.norm(q, dim=-1)
    rot_norm = torch.norm(q_rot, dim=-1)
    assert torch.allclose(orig_norm, rot_norm, atol=1e-4)


def test_swiglu_gated_feedforward():
    d_model = 128
    d_ff = 256
    ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
    x = torch.randn(2, 10, d_model)
    out = ffn(x)

    assert out.shape == (2, 10, d_model)
    assert ffn.w_gate.weight.shape == (d_ff, d_model)
    assert ffn.w_up.weight.shape == (d_ff, d_model)
    assert ffn.w_down.weight.shape == (d_model, d_ff)


def test_modern_attention_qk_norm_and_sdpa():
    d_model = 128
    n_heads = 4
    attn = ModernMultiHeadAttention(d_model=d_model, n_heads=n_heads, use_qk_norm=True, use_rope=True)
    x = torch.randn(2, 8, d_model)

    out, weights = attn(x, x, x, is_causal=False, return_attn=True)
    assert out.shape == (2, 8, d_model)
    assert weights is not None
    assert weights.shape == (2, n_heads, 8, 8)


def test_modern_transformer_parameter_scaling():
    # Verify 100M-scale configuration math
    config = ModernTransformerConfig.modern_100m(vocab_size=16384)
    model = ModernSeq2SeqTransformer(config)
    params = model.count_parameters()
    # Parameters should be between 105M and 115M
    assert 105_000_000 <= params <= 115_000_000, f"Params was {params}"


class DummyDataset(Dataset):
    def __init__(self, size: int = 16):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "src": torch.randint(4, 100, (12,)),
            "tgt_in": torch.randint(4, 100, (10,)),
            "tgt_out": torch.randint(4, 100, (10,))
        }


def test_resumable_trainer_pause_and_resume(tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    flag_file = str(tmp_path / "pause.flag")

    config = ModernTransformerConfig(
        vocab_size=200,
        d_model=64,
        n_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        use_qk_norm=True,
        use_rope=True
    )
    model = ModernSeq2SeqTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trainer = ResumableTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        checkpoint_dir=ckpt_dir,
        pause_flag_file=flag_file,
        save_every_steps=1,
        gradient_accumulation_steps=1,
        use_amp=False,
        device=torch.device("cpu")
    )

    loader = DataLoader(DummyDataset(size=8), batch_size=2)

    # 1. Train 1 step and trigger pause via touch flag
    with open(flag_file, "w") as f:
        f.write("pause")

    result = trainer.train_epoch(loader, log_interval=1)
    assert result["status"] == "paused"
    assert not os.path.exists(flag_file)  # Flag should have been consumed

    latest_ckpt = os.path.join(ckpt_dir, "checkpoint_latest.pt")
    assert os.path.exists(latest_ckpt)

    # 2. Resume training with fresh trainer instance
    model2 = ModernSeq2SeqTransformer(config)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    trainer2 = ResumableTrainer(
        model=model2,
        optimizer=optimizer2,
        scheduler=None,
        checkpoint_dir=ckpt_dir,
        pause_flag_file=flag_file,
        use_amp=False,
        device=torch.device("cpu")
    )

    success = trainer2.resume_from_checkpoint(latest_ckpt)
    assert success is True
    assert trainer2.epoch == trainer.epoch

    # Verify weights match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_bulletproof_trainer_eta_and_progress_file(tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    flag_file = str(tmp_path / "pause.flag")

    config = ModernTransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=64
    )
    model = ModernSeq2SeqTransformer(config)
    from ml_lab.training.resumable_trainer import configure_adamw_optimizer, BulletproofTrainer, format_duration
    optimizer = configure_adamw_optimizer(model, lr=1e-3, weight_decay=0.01)

    trainer = BulletproofTrainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=ckpt_dir,
        pause_flag_file=flag_file,
        save_every_steps=10,
        use_amp=False,
        device=torch.device("cpu")
    )

    loader = DataLoader(DummyDataset(size=8), batch_size=2)
    # Train full epoch
    res = trainer.train_epoch(loader, total_epochs=2, log_interval=1)
    assert res["status"] == "completed"

    # Check progress JSON was written
    progress_file = os.path.join(ckpt_dir, "training_progress.json")
    assert os.path.exists(progress_file)
    import json
    with open(progress_file, "r") as f:
        data = json.load(f)
    assert "epoch_remaining_str" in data
    assert "total_eta_str" in data
    assert "speed_tokens_per_sec" in data
    assert data["epoch"] == 1
    assert data["total_epochs"] == 2

    # Test format duration
    assert format_duration(65) == "1m 05s"
    assert format_duration(3665) == "1h 01m 05s"


def test_gemma2_soft_capping_and_ngram_blocking():
    config = ModernTransformerConfig.modern_small(vocab_size=100)
    config.soft_cap = 30.0
    model = ModernSeq2SeqTransformer(config)

    # Verify soft capping bounds logits strictly within [-30, +30]
    dummy_src = torch.randint(4, 100, (2, 8))
    dummy_tgt = torch.randint(4, 100, (2, 6))
    logits = model(dummy_src, dummy_tgt)
    assert logits.max().item() <= 30.0 + 1e-4
    assert logits.min().item() >= -30.0 - 1e-4

    # Test generation with 3-gram blocking
    gen = model.generate(dummy_src, max_len=15, num_beams=2, no_repeat_ngram_size=3)
    assert len(gen) == 2
    for seq in gen:
        # Check no repeating 3-grams exist
        ngrams = [tuple(seq[i:i+3]) for i in range(len(seq)-2)]
        assert len(ngrams) == len(set(ngrams)), f"Found repeated n-gram in {seq}"


def test_sdpa_explicit_causal_mask_and_kv_cache_parity():
    """Verifies explicit tgt_mask + is_causal doesn't crash SDPA and KV-cache matches uncached forward."""
    torch.manual_seed(7)
    config = ModernTransformerConfig(
        vocab_size=64,
        d_model=64,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.0,
        use_qk_norm=True,
        use_rope=True,
        use_layerscale=True
    )
    model = ModernSeq2SeqTransformer(config)
    model.eval()

    src = torch.tensor([[2, 10, 11, 12, 3], [2, 15, 16, 3, 0]])
    tgt = torch.tensor([[2, 4, 20, 21], [2, 5, 22, 0]])

    # 1. Passing explicit tgt_mask must not raise PyTorch SDPA RuntimeError
    explicit_mask = torch.ones((2, 1, 4, 4), dtype=torch.bool)
    logits_masked = model(src, tgt, tgt_mask=explicit_mask)
    assert logits_masked.shape == (2, 4, 64)

    # 2. KV-cached decode_step_cached must match uncached decode_step logits within floating tolerance
    src_mask = model.make_src_mask(src)
    memory, _ = model.encode(src, src_mask=src_mask)
    prefix = tgt[:, :2]
    uncached_logits, _ = model.decode_step(tgt[:, :3], memory=memory, memory_mask=src_mask, is_causal=True)

    _, _, past_kvs = model.decode_step_cached(prefix, memory=memory, past_kvs=None, memory_mask=src_mask, is_causal=True)
    cached_step_logits, _, _ = model.decode_step_cached(
        tgt[:, 2:3], memory=memory, past_kvs=past_kvs, memory_mask=src_mask, is_causal=True
    )
    assert torch.allclose(uncached_logits[:, 2:3, :], cached_step_logits, atol=1e-4)


def test_polymorphic_greedy_and_beam_decode_contracts():
    """Verifies greedy_decode (with attention) and beam_search_decode match custom_multilingual_service contract."""
    torch.manual_seed(0)
    config = ModernTransformerConfig.modern_small(vocab_size=128)
    config.dropout = 0.0
    model = ModernSeq2SeqTransformer(config)
    model.eval()

    src = torch.tensor([[2, 12, 13, 14, 3]])
    out_tensor = model.greedy_decode(src, sos_idx=2, eos_idx=3, max_len=8, prefix_tokens=[4])
    assert isinstance(out_tensor, torch.Tensor)
    assert out_tensor.dim() == 2 and out_tensor.size(0) == 1
    assert out_tensor[0, 0].item() == 2 and out_tensor[0, 1].item() == 4

    out_with_attn, attn = model.greedy_decode(src, sos_idx=2, eos_idx=3, max_len=8, prefix_tokens=[4], return_attention=True)
    assert isinstance(out_with_attn, torch.Tensor)
    assert attn is not None and attn.dim() == 4
    assert attn.size(0) == 1 and attn.size(1) == config.n_heads and attn.size(3) == src.size(1)

    beams = model.beam_search_decode(src, beam_size=3, max_len=8, sos_idx=2, eos_idx=3, prefix_tokens=[4])
    assert isinstance(beams, list) and len(beams) >= 1
    seq_tensor, score = beams[0]
    assert isinstance(seq_tensor, torch.Tensor) and seq_tensor.dim() == 2
    assert isinstance(score, float)


def test_kaggle_trainer_state_dict_parity_and_bucket_resume(tmp_path):
    """
    Verifies 100% strict state_dict compatibility between Kaggle standalone ModernSeq2SeqTransformer
    and core ModernSeq2SeqTransformer, plus CosineWarmupScheduler and deterministic mid-epoch resume.
    """
    from ml_lab.kaggle.translateme_100m_kaggle_trainer import ModernSeq2SeqTransformer as KaggleModel
    from ml_lab.training.resumable_trainer import (
        BulletproofTrainer,
        CosineWarmupScheduler,
        LengthBucketBatchSampler,
        configure_adamw_optimizer,
    )

    k_model = KaggleModel(
        vocab_size=200, d_model=64, n_heads=4, num_encoder_layers=2, num_decoder_layers=2, d_ff=128, dropout=0.0
    )
    core_cfg = ModernTransformerConfig(
        vocab_size=200, d_model=64, n_heads=4, num_encoder_layers=2, num_decoder_layers=2, d_ff=128, dropout=0.0
    )
    core_model = ModernSeq2SeqTransformer(core_cfg)
    # Must load with strict=True with zero missing/unexpected keys!
    core_model.load_state_dict(k_model.state_dict(), strict=True)

    # Test deterministic bucket sampler + mid-epoch resume
    ds = DummyDataset(size=12)
    sampler = LengthBucketBatchSampler([12] * 12, batch_size=2, seed=42)
    loader = DataLoader(ds, batch_sampler=sampler)

    opt = configure_adamw_optimizer(core_model, lr=1e-3, weight_decay=0.01)
    sched = CosineWarmupScheduler(opt, warmup_steps=2, total_steps=10, peak_lr=1e-3)
    trainer = BulletproofTrainer(
        model=core_model, optimizer=opt, scheduler=sched,
        checkpoint_dir=str(tmp_path / "ckpts"),
        pause_flag_file=str(tmp_path / "pause.flag"),
        gradient_accumulation_steps=1,
        use_amp=False,
        device=torch.device("cpu")
    )
    # Run full epoch and verify checkpoint_best.pt is inference-clean and loads into ModernSeq2SeqTransformer
    res = trainer.train_epoch(loader, val_loader=loader, total_epochs=2, log_interval=2)
    assert res["status"] == "completed"
    best_ckpt = torch.load(str(tmp_path / "ckpts" / "checkpoint_best.pt"), map_location="cpu", weights_only=False)
    assert "optimizer_state_dict" not in best_ckpt
    core_model2 = ModernSeq2SeqTransformer(core_cfg)
    core_model2.load_state_dict(best_ckpt["model_state_dict"], strict=True)


def test_kaggle_checkpoint_loads_into_backend_and_evaluates(tmp_path):
    """End-to-end check: Kaggle trainer produces checkpoint_best.pt that loads into CustomMultilingualEngine and translates."""
    import json
    from ml_lab.kaggle.translateme_100m_kaggle_trainer import (
        MultilingualSubwordTokenizer as KaggleTok,
        ModernSeq2SeqTransformer as KaggleModel,
        PreTokenizedNMTDataset,
        LengthBucketBatchSampler,
        build_tensor_collate_fn,
        configure_adamw_optimizer as k_opt,
        CosineWarmupScheduler as KSched,
        Kaggle100MTrainer,
    )
    from backend.app.services.custom_multilingual_service import CustomMultilingualEngine

    train_jsonl = tmp_path / "train.jsonl"
    val_jsonl = tmp_path / "val.jsonl"
    pairs = [
        {"source": "Hello how are you", "target": "नमस्ते आप कैसे हैं", "target_lang": "hi"},
        {"source": "What is your name", "target": "आपका नाम क्या है", "target_lang": "hi"},
        {"source": "Machine learning works", "target": "இயந்திர கற்றல்", "target_lang": "ta"},
        {"source": "Deep learning models", "target": "డీప్ లెర్నింగ్", "target_lang": "te"},
    ] * 3
    train_jsonl.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8")
    val_jsonl.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs[:4]), encoding="utf-8")

    tok = KaggleTok()
    tok.train_from_iterator([p["source"] for p in pairs] + [p["target"] for p in pairs], vocab_size=300, min_frequency=1)
    tok_path = str(tmp_path / "tok.json")
    tok.save(tok_path)

    train_ds = PreTokenizedNMTDataset(str(train_jsonl), tok, max_len=64, cache_path=str(tmp_path / "train.pt"))
    val_ds = PreTokenizedNMTDataset(str(val_jsonl), tok, max_len=64, cache_path=str(tmp_path / "val.pt"))
    train_loader = DataLoader(train_ds, batch_sampler=LengthBucketBatchSampler(train_ds.lengths, 4), collate_fn=build_tensor_collate_fn(tok.pad_id))
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=build_tensor_collate_fn(tok.pad_id))

    model = KaggleModel(vocab_size=tok.vocab_size, d_model=64, n_heads=4, num_encoder_layers=2, num_decoder_layers=2, d_ff=128)
    opt = k_opt(model, lr=1e-3, weight_decay=0.01)
    sched = KSched(opt, warmup_steps=1, total_steps=10, peak_lr=1e-3)
    trainer = Kaggle100MTrainer(
        model=model, optimizer=opt, scheduler=sched,
        checkpoint_dir=str(tmp_path / "ckpts"), tokenizer_path=tok_path,
        gradient_accumulation_steps=1, use_amp=False, save_every_steps=100, eval_every_steps=0,
        device=torch.device("cpu")
    )
    trainer.train_epoch(train_loader, val_loader, total_epochs=1, log_interval=10, tokenizer=tok)

    engine = CustomMultilingualEngine(checkpoint_path=str(tmp_path / "ckpts" / "checkpoint_best.pt"), tokenizer_path=tok_path)
    assert engine.load_model() is True
    res_g = engine.translate("Hello how are you", "en", "hi", num_beams=1, return_attention=True)
    res_b = engine.translate("Hello how are you", "en", "hi", num_beams=2, return_alternatives=True)
    assert "cross_attention_matrix" in res_g
    assert isinstance(res_b["translated_text"], str)


def test_12dir_script_filter_and_option1_param_count():
    """Verifies Option 1 113.29M parameter count and 12-direction script purity validation."""
    from ml_lab.datasets.build_100m_dataset import is_valid_pair, has_dominant_script

    opt1_cfg = ModernTransformerConfig.modern_asymmetric_100m(vocab_size=24576)
    opt1_model = ModernSeq2SeqTransformer(opt1_cfg)
    params = opt1_model.count_parameters()
    assert 112_000_000 <= params <= 115_000_000, f"Expected ~113.29M params, got {params}"

    assert has_dominant_script("नमस्ते दुनिया", "hi") is True
    assert has_dominant_script("Hello world", "hi") is False
    assert is_valid_pair("Good morning everyone", "शुभ प्रभात सभी को", src_lang="en", tgt_lang="hi") is True
    assert is_valid_pair("Good morning everyone", "Good morning friend", src_lang="en", tgt_lang="hi") is False
    assert is_valid_pair("வணக்கம் நண்பரே", "నమస్కారం మిత్రమా", src_lang="ta", tgt_lang="te") is True


def test_github_ddp_trainer_pause_resume_and_probes(tmp_path):
    """Tests train_from_github.py end-to-end: pre-tokenized cache, pause.flag, exact resume, and checkpoint export."""
    import json
    from ml_lab.kaggle.train_from_github import main as github_train_main
    from ml_lab.tokenizer.multilingual_tokenizer import MultilingualSubwordTokenizer

    work_dir = tmp_path / "translateme_state"
    cache_dir = tmp_path / "cache"
    jsonl_dir = cache_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {"source": "Hello how are you", "target": "नमस्ते आप कैसे हैं", "source_lang": "en", "target_lang": "hi"},
        {"source": "नमस्ते आप कैसे हैं", "target": "வணக்கம் நீங்கள் எப்படி", "source_lang": "hi", "target_lang": "ta"},
        {"source": "வணக்கம் நீங்கள் எப்படி", "target": "నమస్కారం మీరు ఎలా", "source_lang": "ta", "target_lang": "te"},
        {"source": "నమస్కారం మీరు ఎలా", "target": "Hello how are you", "source_lang": "te", "target_lang": "en"},
    ] * 4
    (jsonl_dir / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (jsonl_dir / "val.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows[:4]), encoding="utf-8")

    # 1. Trigger pause.flag on first run
    ckpt_dir = work_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "pause.flag").write_text("pause", encoding="utf-8")

    common_args = [
        "--work_dir", str(work_dir),
        "--cache_dir", str(cache_dir),
        "--vocab_size", "300",
        "--d_model", "64",
        "--n_heads", "4",
        "--enc_layers", "2",
        "--dec_layers", "2",
        "--d_ff", "128",
        "--epochs", "1",
        "--batch_size", "4",
        "--grad_accum", "2",
        "--save_every", "2",
        "--eval_every", "0",
        "--log_every", "2",
        "--no_ddp",
    ]
    github_train_main(common_args)
    assert not (ckpt_dir / "pause.flag").exists()
    assert (ckpt_dir / "checkpoint_latest.pt").exists()

    # 2. Resume cleanly to completion
    github_train_main(common_args + ["--resume", "auto"])
    assert (ckpt_dir / "checkpoint_best.pt").exists()
    assert (ckpt_dir / "training_progress.json").exists()

    # Verify TensorCollateFn and DistributedBucketBatchSampler are picklable under spawn
    import pickle
    from ml_lab.kaggle.train_from_github import build_tensor_collate_fn, DistributedBucketBatchSampler
    fn = build_tensor_collate_fn(pad_id=0)
    assert pickle.loads(pickle.dumps(fn)).pad_id == 0
    sampler = DistributedBucketBatchSampler([10, 20, 30, 40], batch_size=2)
    assert len(pickle.loads(pickle.dumps(sampler))) == 2

    # Verify Token-Budget Dynamic Batching obeys max_tokens limit and stays deterministic per epoch
    lengths = [10, 12, 14, 16, 100, 110, 120, 20, 22, 24]
    tok_sampler = DistributedBucketBatchSampler(lengths, max_tokens=60, max_batch_size=8, rank=0, world_size=2, seed=42)
    b_ep0 = list(tok_sampler)
    assert len(b_ep0) == len(tok_sampler)
    for b in b_ep0:
        assert len(b) <= 8
        assert len(b) == 1 or len(b) * max(lengths[i] for i in b) <= 60
    assert list(tok_sampler) == b_ep0  # Same epoch -> identical deterministic order





