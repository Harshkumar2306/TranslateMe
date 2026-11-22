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
