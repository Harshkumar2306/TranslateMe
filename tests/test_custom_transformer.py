"""
Unit tests for Custom Transformer implementation
"""

import pytest
import torch
from models.custom_transformer import (
    TokenEmbedding,
    PositionalEncoding,
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionwiseFeedForward,
    LayerNorm,
    EncoderLayer,
    TransformerEncoder,
    DecoderLayer,
    TransformerDecoder,
    Seq2SeqTransformer,
    make_src_mask,
    make_tgt_mask
)


def test_token_embedding_and_positional_encoding():
    vocab_size = 100
    d_model = 64
    seq_len = 10
    batch_size = 2

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    emb_layer = TokenEmbedding(vocab_size, d_model)
    pe_layer = PositionalEncoding(d_model, max_len=100)

    emb = emb_layer(tokens)
    assert emb.shape == (batch_size, seq_len, d_model)

    pos_emb = pe_layer(emb)
    assert pos_emb.shape == (batch_size, seq_len, d_model)


def test_scaled_dot_product_attention():
    batch_size = 2
    n_heads = 4
    seq_len = 6
    d_k = 16

    q = torch.randn(batch_size, n_heads, seq_len, d_k)
    k = torch.randn(batch_size, n_heads, seq_len, d_k)
    v = torch.randn(batch_size, n_heads, seq_len, d_k)

    attn = ScaledDotProductAttention(dropout=0.0)
    out, weights = attn(q, k, v)

    assert out.shape == (batch_size, n_heads, seq_len, d_k)
    assert weights.shape == (batch_size, n_heads, seq_len, seq_len)
    # Check that attention weights sum to 1 across key dimension
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, n_heads, seq_len), atol=1e-5)


def test_multi_head_attention():
    batch_size = 2
    seq_len = 8
    d_model = 64
    n_heads = 8

    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    out = mha(x, x, x)
    assert out.shape == (batch_size, seq_len, d_model)


def test_causal_mask():
    tgt = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    mask = make_tgt_mask(tgt, pad_idx=0)
    # Shape should be (batch_size, 1, seq_len, seq_len)
    assert mask.shape == (2, 1, 4, 4)
    # Token at index 0 should NOT be able to attend to token at index 1
    assert mask[0, 0, 0, 1] == False
    # Token at index 1 should be able to attend to token at index 0
    assert mask[0, 0, 1, 0] == True
    # Padding token at index 3 in first sequence should be masked out
    assert mask[0, 0, 3, 3] == False


def test_seq2seq_transformer_forward():
    src_vocab = 50
    tgt_vocab = 60
    d_model = 64
    n_heads = 4

    model = Seq2SeqTransformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=d_model,
        n_heads=n_heads,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        max_len=100
    )

    src = torch.randint(1, src_vocab, (2, 7))
    tgt = torch.randint(1, tgt_vocab, (2, 5))

    output = model(src, tgt)
    assert output.shape == (2, 5, tgt_vocab)


def test_greedy_decode():
    src_vocab = 50
    tgt_vocab = 60
    d_model = 64

    model = Seq2SeqTransformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=d_model,
        n_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=128,
        max_len=50
    )

    src = torch.randint(1, src_vocab, (2, 6))
    generated = model.greedy_decode(src, sos_idx=1, eos_idx=2, max_len=8)
    assert generated.shape[0] == 2
    assert generated.shape[1] <= 8
    assert (generated[:, 0] == 1).all()
