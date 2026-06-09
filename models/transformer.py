# models/transformer.py
from typing import Optional

import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        """
        Multi-Head Self-Attention from scratch.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, S, E)
            mask: additive mask with shape broadcastable to (B, H, S, S)
                  Typically 0 for allowed, -inf for blocked.
        Returns:
            (B, S, E)
        """
        B, S, E = x.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(x)  # (B, S, E)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape to heads
        q = q.view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)
        k = k.view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)
        v = v.view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)

        # attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # (B, H, S, S)

        if mask is not None:
            # mask expected additive: 0 or -inf, broadcastable to scores
            scores = scores + mask

        attn = torch.softmax(scores, dim=-1)  # (B, H, S, S)
        out = torch.matmul(attn, v)          # (B, H, S, D)

        out = out.transpose(1, 2).contiguous().view(B, S, E)  # (B, S, E)
        out = self.out_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int):
        """
        Fixed sinusoidal positional encoding.
        """
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)  # (max_len, E)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, E)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        return x + self.pe[:, :S, :]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN attention
        h = self.norm1(x)
        h = self.attn(h, mask=mask)
        x = x + self.drop1(h)

        # Pre-LN MLP
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + self.drop2(h)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        mlp_dim: int,
        num_heads: int,
        num_layers: int,
        pad_token: int,
        max_len: int,
        dropout: float,
    ):
        super().__init__()
        self.pad_token = pad_token
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=pad_token)
        self.pos = PositionalEncoding(embed_dim, max_len=max_len)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, S)
            lengths: (B,)
            mask: optional causal mask (S, S) with -inf above diagonal (as created in lightning).
        Returns:
            (B, S, E)
        """
        B, S = x.shape
        device = x.device

        # padding mask (mask KEYS where pad)
        pad = (x == self.pad_token)  # (B, S) bool

        # Build additive attention mask broadcastable to (B, H, S, S)
        # Start with zeros
        attn_mask = torch.zeros(B, 1, S, S, device=device)

        # 1) causal mask: (S, S) with -inf on forbidden positions
        if mask is not None:
            if mask.dim() == 2:
                attn_mask = attn_mask + mask.view(1, 1, S, S)
            else:
                attn_mask = attn_mask + mask

        # 2) padding mask for keys: if key is pad -> -inf for all queries
        # pad: (B, S) -> (B, 1, 1, S) then broadcast to (B,1,S,S)
        key_pad = pad.view(B, 1, 1, S)
        attn_mask = attn_mask.masked_fill(key_pad, float("-inf"))

        # embeddings
        h = self.embed(x)          # (B, S, E)
        h = self.pos(h)
        h = self.drop(h)

        for layer in self.layers:
            h = layer(h, mask=attn_mask)

        # Optionally zero-out pad positions (not strictly necessary, but cleaner)
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        return h


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        mlp_dim: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        pad_token: int,
        max_len: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            pad_token=pad_token,
            max_len=max_len,
            dropout=dropout,
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        h = self.encoder(x, lengths, mask=mask)
        h = self.final_norm(h)

        # ---- Last valid token pooling (GPT-style) ----
        B = h.size(0)
        idx = torch.arange(B, device=h.device)
        last_idx = lengths - 1
        seq_repr = h[idx, last_idx, :]  # (B, E)

        logits = self.head(seq_repr)    # (B, C)

        if return_attention:
            return logits, None
        return logits
