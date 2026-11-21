"""
Unit tests for Custom Transformer implementation
"""

import pytest
import torch
from ml_lab.custom_transformer import (
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
    TransformerConfig,
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
    assert mask.shape == (2, 1, 4, 4)
    assert mask[0, 0, 0, 1] == False
    assert mask[0, 0, 1, 0] == True
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


def test_future_token_leakage():
    """
    Verifies that the causal mask prevents future token information leakage.
    Changing tokens in target positions > t must NOT change the decoder output at position t.
    """
    config = TransformerConfig.tiny(vocab_size=50)
    model = Seq2SeqTransformer.from_config(config)
    model.eval()

    src = torch.tensor([[10, 20, 30]])
    tgt1 = torch.tensor([[1, 5, 8, 12]])
    tgt2 = torch.tensor([[1, 5, 8, 49]])  # Only the last token is different

    with torch.no_grad():
        out1 = model(src, tgt1)
        out2 = model(src, tgt2)

    # Outputs at position 0, 1, 2 must be IDENTICAL within machine precision
    assert torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-5), "Future token leaked into previous positions!"
    # Output at position 3 can differ
    assert not torch.allclose(out1[:, 3, :], out2[:, 3, :], atol=1e-3)


def test_beam_search_decode():
    config = TransformerConfig.tiny(vocab_size=50)
    model = Seq2SeqTransformer.from_config(config)
    model.eval()

    src = torch.tensor([[5, 12, 18]])
    beams = model.beam_search_decode(src, beam_size=3, max_len=6, sos_idx=1, eos_idx=2)
    assert len(beams) == 3
    # Top beam has valid sequence
    top_seq, top_score = beams[0]
    assert top_seq.shape[0] == 1
    assert top_seq[0, 0].item() == 1  # starts with SOS


def test_tied_embeddings():
    config = TransformerConfig.tiny(vocab_size=100)
    config.share_embeddings = True
    model = Seq2SeqTransformer.from_config(config)

    # Weights between generator projection and shared embedding must be identical
    assert torch.equal(model.generator.proj.weight, model.encoder.embedding.lut.weight)
    assert torch.equal(model.decoder.embedding.lut.weight, model.encoder.embedding.lut.weight)


def test_cross_attention_extraction():
    config = TransformerConfig.tiny(vocab_size=50)
    model = Seq2SeqTransformer.from_config(config)
    model.eval()

    src = torch.tensor([[10, 15, 20]])
    generated, cross_attn = model.greedy_decode(src, sos_idx=1, eos_idx=2, max_len=5, return_attention=True)
    assert cross_attn is not None
    # Shape of cross attention: (batch, n_heads, tgt_len, src_len)
    assert cross_attn.shape[0] == 1
    assert cross_attn.shape[1] == config.n_heads
    assert cross_attn.shape[3] == 3

