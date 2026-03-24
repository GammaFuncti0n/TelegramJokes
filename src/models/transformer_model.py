import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, dropout, model_max_length, **kwargs):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.model_max_length = model_max_length

        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size
        )
        # self.pe = nn.Embedding(
        #     num_embeddings=self.model_max_length,
        #     embedding_dim=self.hidden_size
        # )  
        #nn.TransformerEncoderLayer(
        encoder_layer = TransformerEncoderLayerRoPE(
            d_model=self.hidden_size, 
            nhead=self.num_heads,
            dim_feedforward=2*self.hidden_size,
            dropout=self.dropout,
            batch_first=True
            )
        self.transformer_decoder = nn.TransformerEncoder(encoder_layer, self.num_layers, enable_nested_tensor=False)

        self.head = nn.Linear(hidden_size, self.vocab_size)
    def forward(self, x):
        B, T = x.shape
        #positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        emb = self.embeddings(x) # + self.pe(positions)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        out = self.transformer_decoder(emb, mask=mask)
        out = self.head(out)
        return out

# =========================
# RoPE (Rotary Embeddings)
# =========================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)

        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x, seq_len):
        return (
            self.cos[:seq_len, :].to(x.device),
            self.sin[:seq_len, :].to(x.device),
        )


def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q, k, cos, sin):
    # q, k: (batch, heads, seq, dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,seq,dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)

    return q, k


# =========================
# Multihead Attention + RoPE
# =========================
class MultiheadAttentionRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        if not self.batch_first:
            x = x.transpose(0, 1)  # (seq, batch, dim) -> (batch, seq, dim)

        B, S, D = x.shape

        qkv = self.qkv_proj(x)  # (B, S, 3D)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, Hd)

        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rope(x, S)
        q, k = apply_rope(q, k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)  # (B, H, S, Hd)
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        return out


# =========================
# Transformer Encoder Layer (RoPE)
# =========================
class TransformerEncoderLayerRoPE(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        batch_first=False,
        norm_first=False,
    ):
        super().__init__()

        self.self_attn = MultiheadAttentionRoPE(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError("Unsupported activation")

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, attn_mask, key_padding_mask)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if self.norm_first:
            # Pre-norm
            src = src + self._sa_block(
                self.norm1(src), src_mask, src_key_padding_mask
            )
            src = src + self._ff_block(self.norm2(src))
        else:
            # Post-norm (как в torch)
            src = self.norm1(
                src + self._sa_block(src, src_mask, src_key_padding_mask)
            )
            src = self.norm2(src + self._ff_block(src))

        return src


# =========================
# Full Encoder (stack)
# =========================
# class TransformerEncoderRoPE(nn.Module):
#     def __init__(self, encoder_layer, num_layers):
#         super().__init__()
#         self.layers = nn.ModuleList(
#             [encoder_layer for _ in range(num_layers)]
#         )

#     def forward(self, src, mask=None, src_key_padding_mask=None):
#         output = src
#         for layer in self.layers:
#             output = layer(output, mask, src_key_padding_mask)
#         return output