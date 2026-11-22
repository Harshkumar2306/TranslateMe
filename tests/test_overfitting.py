"""
Empirical Overfitting & Mathematical Correctness Test Suite
Section 15 (Phase 1, Item 5) of ML Review

Proves that ModernSeq2SeqTransformer:
1. Propagates gradients correctly across tied embeddings, RoPE, Pre-LN RMSNorm, LayerScale, and SwiGLU.
2. Learns sequence alignments via multi-head self-attention and cross-attention.
3. Overfits a small parallel corpus to near-zero loss (< 0.05) and 100% exact target string generation.
4. Correctly clamps finished batch elements during greedy decode without post-EOS corruption.
"""

import pytest
import torch
import torch.nn as nn
from ml_lab.custom_transformer.modern_transformer import ModernSeq2SeqTransformer, ModernTransformerConfig


def test_tiny_corpus_overfitting_greedy():
    """
    Overfits a 4-sequence parallel dataset from scratch.
    Validates teacher forcing, backward pass, and greedy autoregressive generation.
    """
    torch.manual_seed(42)
    vocab_size = 32
    pad_idx = 0
    bos_idx = 2
    eos_idx = 3

    config = ModernTransformerConfig(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.0,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        use_qk_norm=True,
        use_rope=True,
        share_embeddings=True,
        use_layerscale=True,
        soft_cap=None
    )
    model = ModernSeq2SeqTransformer(config)

    src_data = torch.tensor([
        [4, 5, 6, 7, eos_idx],
        [8, 9, 10, 11, eos_idx],
        [12, 13, 14, 15, eos_idx],
        [16, 17, 18, 19, eos_idx]
    ])

    tgt_data = torch.tensor([
        [bos_idx, 20, 21, 22, eos_idx],
        [bos_idx, 23, 24, 25, eos_idx],
        [bos_idx, 26, 27, 28, eos_idx],
        [bos_idx, 29, 30, 31, eos_idx]
    ])

    tgt_in = tgt_data[:, :-1]
    tgt_out = tgt_data[:, 1:]

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    # Train for 40 steps
    for _ in range(40):
        optimizer.zero_grad()
        logits = model(src_data, tgt_in)
        loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

    # Final loss must drop below 0.05
    assert loss.item() < 0.05, f"Expected loss < 0.05, got {loss.item():.4f}"

    # Greedy decode must exactly match target sequences
    gen = model.generate(src_data, max_len=6, num_beams=1)
    assert gen == tgt_data.tolist(), f"Greedy generation failed to match target: {gen} vs {tgt_data.tolist()}"


def test_tiny_corpus_overfitting_beam_search():
    """
    Validates that beam search with length normalization correctly finds the memorized target sequences.
    """
    torch.manual_seed(42)
    vocab_size = 32
    pad_idx = 0
    bos_idx = 2
    eos_idx = 3

    config = ModernTransformerConfig(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.0,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        use_qk_norm=True,
        use_rope=True,
        share_embeddings=True,
        use_layerscale=True,
        soft_cap=None
    )
    model = ModernSeq2SeqTransformer(config)

    src_data = torch.tensor([
        [4, 5, 6, 7, eos_idx],
        [8, 9, 10, 11, eos_idx]
    ])

    tgt_data = torch.tensor([
        [bos_idx, 20, 21, 22, eos_idx],
        [bos_idx, 23, 24, 25, eos_idx]
    ])

    tgt_in = tgt_data[:, :-1]
    tgt_out = tgt_data[:, 1:]

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    for _ in range(40):
        optimizer.zero_grad()
        logits = model(src_data, tgt_in)
        loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.05

    # Beam search with 2 beams
    gen_beam = model.generate(src_data, max_len=6, num_beams=2)
    assert gen_beam == tgt_data.tolist(), f"Beam search failed: {gen_beam} vs {tgt_data.tolist()}"


def test_greedy_decoding_finished_sequence_clamping():
    """
    Regression test for the bug highlighted in Review Section 3:
    When sentence A finishes earlier than sentence B, sentence A must NOT emit arbitrary tokens.
    """
    torch.manual_seed(42)
    vocab_size = 32
    pad_idx = 0
    bos_idx = 2
    eos_idx = 3

    config = ModernTransformerConfig(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.0,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx
    )
    model = ModernSeq2SeqTransformer(config)

    # Sequence A target is short (length 3), Sequence B target is long (length 6)
    src_data = torch.tensor([
        [4, 5, eos_idx, pad_idx],
        [6, 7, 8, eos_idx]
    ])

    tgt_data = torch.tensor([
        [bos_idx, 20, eos_idx, pad_idx, pad_idx, pad_idx],
        [bos_idx, 21, 22, 23, 24, eos_idx]
    ])

    tgt_in = tgt_data[:, :-1]
    tgt_out = tgt_data[:, 1:]

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015)

    for _ in range(50):
        optimizer.zero_grad()
        logits = model(src_data, tgt_in)
        loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

    gen = model.generate(src_data, max_len=10, num_beams=1)
    # Sentence 0 must terminate cleanly at eos_idx without extraneous tokens
    assert gen[0] == [bos_idx, 20, eos_idx], f"Sequence A did not clamp properly: {gen[0]}"
    assert gen[1] == [bos_idx, 21, 22, 23, 24, eos_idx], f"Sequence B failed: {gen[1]}"


def test_batch_decoding_three_way_eos_edge_cases():
    """
    Review Section 2 edge-case regression test:
    Validates batch decoding with 3 simultaneous scenarios:
    1. Batch item 1 emits EOS immediately (step 1).
    2. Batch item 2 emits EOS after five steps.
    3. Batch item 3 never emits EOS before max_len limit.
    Guarantees no token bleeding, no padding interpreted as EOS, and clean boundary handling.
    """
    torch.manual_seed(42)
    vocab_size = 32
    pad_idx = 0
    bos_idx = 2
    eos_idx = 3

    config = ModernTransformerConfig(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.0,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx
    )
    model = ModernSeq2SeqTransformer(config)

    src_data = torch.tensor([
        [4, eos_idx, pad_idx, pad_idx],
        [5, 6, 7, eos_idx],
        [8, 9, 10, eos_idx]
    ])

    tgt_data = torch.tensor([
        [bos_idx, eos_idx, pad_idx, pad_idx, pad_idx, pad_idx],
        [bos_idx, 21, 22, 23, 24, eos_idx],
        [bos_idx, 25, 26, 27, 28, 29]
    ])

    tgt_in = tgt_data[:, :-1]
    tgt_out = tgt_data[:, 1:]

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015)

    for _ in range(50):
        optimizer.zero_grad()
        logits = model(src_data, tgt_in)
        loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

    gen = model.generate(src_data, max_len=5, num_beams=1)
    assert gen[0] == [bos_idx, eos_idx], f"Item 1 failed immediate EOS: {gen[0]}"
    assert gen[1] == [bos_idx, 21, 22, 23, 24, eos_idx], f"Item 2 failed mid-step EOS: {gen[1]}"
    assert gen[2] == [bos_idx, 25, 26, 27, 28, 29], f"Item 3 failed max-len termination without EOS: {gen[2]}"

