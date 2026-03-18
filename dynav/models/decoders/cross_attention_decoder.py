"""Cross-attention decoder for Map Navigation Model.

Observation tokens query the map tokens for spatial guidance.
Q = obs, K/V = map — each block progressively fuses map context into obs tokens.
"""

from typing import Optional

import torch
import torch.nn as nn


class CrossAttentionDecoderBlock(nn.Module):
    """Single cross-attention decoder block.

    Sub-layer order:
        1. Self-attention on obs_tokens (obs tokens attend to each other)
        2. Cross-attention: Q=obs, K/V=map (obs tokens query the map)
        3. FFN: position-wise feed-forward

    Each sub-layer uses Pre-Norm + residual connection.

    Args:
        token_dim: Token embedding dimension (d).
        n_heads: Number of attention heads.
        d_ff: Hidden dimension of the FFN.
        dropout: Dropout probability applied inside FFN and after attention.
    """

    def __init__(
        self,
        token_dim: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Sub-layer 1: self-attention on obs tokens
        self.self_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(token_dim)

        # Sub-layer 2: cross-attention (Q=obs, K/V=map)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(token_dim)

        # Sub-layer 3: FFN
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, token_dim),
        )
        self.norm3 = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        obs_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply one cross-attention decoder block.

        Args:
            obs_tokens: Observation token sequence of shape (B, N_o, d).
            map_tokens: Map spatial token sequence of shape (B, N_m, d).
            return_attention: If True, return cross-attention weights.

        Returns:
            Tuple of:
                - obs_tokens_out: Updated obs tokens of shape (B, N_o, d).
                - attn_weights: Cross-attention weights of shape
                  (B, N_o, N_m) if return_attention else None.
                  Note: nn.MultiheadAttention averages weights over heads.
        """
        # ── Sub-layer 1: self-attention ────────────────────────────────────────
        residual = obs_tokens                                          # (B, N_o, d)
        x = self.norm1(obs_tokens)
        x, _ = self.self_attn(x, x, x, need_weights=False)            # (B, N_o, d)
        obs_tokens = residual + self.dropout(x)                        # (B, N_o, d)

        # ── Sub-layer 2: cross-attention ───────────────────────────────────────
        residual = obs_tokens                                          # (B, N_o, d)
        x = self.norm2(obs_tokens)
        x, attn_weights = self.cross_attn(
            query=x,
            key=map_tokens,
            value=map_tokens,
            need_weights=return_attention,
            average_attn_weights=True,     # average over heads → (B, N_o, N_m)
        )                                                              # (B, N_o, d)
        obs_tokens = residual + self.dropout(x)                        # (B, N_o, d)

        # ── Sub-layer 3: FFN ───────────────────────────────────────────────────
        residual = obs_tokens                                          # (B, N_o, d)
        x = self.norm3(obs_tokens)
        obs_tokens = residual + self.dropout(self.ffn(x))              # (B, N_o, d)

        return obs_tokens, attn_weights  # attn_weights is None when not requested


class CrossAttentionDecoder(nn.Module):
    """Stacked cross-attention decoder that fuses map context into obs tokens.

    Applies N CrossAttentionDecoderBlocks sequentially, then mean-pools the
    final obs token sequence to produce a single context vector.

    Args:
        token_dim: Token embedding dimension (d).
        n_layers: Number of stacked decoder blocks (L).
        n_heads: Number of attention heads per block.
        d_ff: FFN hidden dimension.
        dropout: Dropout probability.

    Example:
        >>> dec = CrossAttentionDecoder(token_dim=256, n_layers=4, n_heads=4, d_ff=512)
        >>> obs = torch.randn(2, 4, 256)
        >>> mp  = torch.randn(2, 49, 256)
        >>> ctx, attn = dec(obs, mp, return_attention=True)
        >>> ctx.shape    # (2, 256)
        >>> len(attn)    # 4  — one per layer
    """

    def __init__(
        self,
        token_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([
            CrossAttentionDecoderBlock(token_dim, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        obs_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        """Fuse map context into obs tokens and produce a context vector.

        Args:
            obs_tokens: Observation token sequence of shape (B, N_o, d).
            map_tokens: Map spatial token sequence of shape (B, N_m, d).
            return_attention: If True, collect cross-attention weights from
                every layer.

        Returns:
            Tuple of:
                - context: Mean-pooled context vector of shape (B, d).
                - attn_weights_per_layer: List of (B, N_o, N_m) tensors,
                  one per layer, if return_attention else None.
        """
        attn_weights_per_layer: Optional[list[torch.Tensor]] = (
            [] if return_attention else None
        )

        x = obs_tokens  # (B, N_o, d)
        for block in self.blocks:
            x, attn = block(x, map_tokens, return_attention=return_attention)
            # x:    (B, N_o, d)
            # attn: (B, N_o, N_m) or None
            if return_attention and attn is not None:
                attn_weights_per_layer.append(attn)  # type: ignore[union-attr]

        context = x.mean(dim=1)  # (B, d)
        return context, attn_weights_per_layer
