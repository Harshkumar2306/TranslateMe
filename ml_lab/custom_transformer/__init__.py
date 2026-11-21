"""
TranslateMe Custom Transformer Package
From-scratch PyTorch implementation of the Transformer Architecture (Vaswani et al., 2017)
"""

from .embeddings import TokenEmbedding, PositionalEncoding, TransformerEmbedding
from .attention import ScaledDotProductAttention, MultiHeadAttention
from .feed_forward import PositionwiseFeedForward
from .sublayers import LayerNorm, SublayerConnection
from .encoder import EncoderLayer, TransformerEncoder
from .decoder import DecoderLayer, TransformerDecoder
from .transformer import (
    Seq2SeqTransformer,
    Generator,
    TransformerConfig,
    make_src_mask,
    make_tgt_mask,
    make_subsequent_mask
)

__all__ = [
    "TokenEmbedding",
    "PositionalEncoding",
    "TransformerEmbedding",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "PositionwiseFeedForward",
    "LayerNorm",
    "SublayerConnection",
    "EncoderLayer",
    "TransformerEncoder",
    "DecoderLayer",
    "TransformerDecoder",
    "Seq2SeqTransformer",
    "Generator",
    "TransformerConfig",
    "make_src_mask",
    "make_tgt_mask",
    "make_subsequent_mask",
]
