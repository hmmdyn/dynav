"""Self-attention decoder for Map Navigation Model.

Concatenates obs and map tokens, applies full self-attention over the combined
sequence, then extracts obs positions for mean pooling.

Uses a custom pre-norm Transformer encoder (instead of nn.TransformerEncoder)
so per-layer attention weights can be returned for interpretability analysis
(e.g. how much obs tokens attend to the map token). Module/parameter names
match nn.TransformerEncoderLayer exactly, so checkpoints trained with the
previous nn.TransformerEncoder implementation load without key remapping.

Interface is identical to CrossAttentionDecoder for drop-in replacement.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PreNormEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer that can expose attention weights.

    Structurally and nominally identical to
    ``nn.TransformerEncoderLayer(norm_first=True, activation="relu")`` —
    same submodule names (self_attn, linear1, dropout, linear2, norm1, norm2,
    dropout1, dropout2) for state-dict compatibility.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        need_weights: bool = False,
        average_attn_weights: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply pre-norm self-attention + FFN.

        Args:
            x: Token sequence of shape (B, N, d).
            need_weights: If True, also return attention weights.
            average_attn_weights: If True, average weights over heads
                → (B, N, N); otherwise per-head → (B, n_heads, N, N).

        Returns:
            Tuple of (output tokens (B, N, d), attention weights or None).
        """
        h = self.norm1(x)
        attn_out, attn_w = self.self_attn(
            h, h, h,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
        x = x + self.dropout1(attn_out)
        h = self.norm2(x)
        x = x + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(h)))))
        return x, attn_w


class _Encoder(nn.Module):
    """Stack of _PreNormEncoderLayer — holds ``.layers`` to keep the
    ``transformer.layers.{i}.*`` state-dict prefix of nn.TransformerEncoder."""

    def __init__(self, layer_factory, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([layer_factory() for _ in range(n_layers)])


class SelfAttentionDecoder(nn.Module):
    """Self-attention decoder over the combined obs+map token sequence.

    Concatenates obs_tokens and map_tokens along the sequence dimension,
    adds learnable token-type embeddings to distinguish obs vs. map tokens,
    passes the combined sequence through a stack of pre-norm Transformer
    encoder layers, then extracts only the obs positions and mean-pools them
    to a context vector.

    The forward interface matches CrossAttentionDecoder exactly, so the decoder
    can be swapped via config (decoder.type: "self_attention") without changing
    any calling code.

    Args:
        token_dim: Token embedding dimension (d).
        n_obs: Number of observation tokens (N_o). Required to know which
            positions to extract after the combined self-attention pass.
        n_layers: Number of stacked Transformer encoder layers (L).
        n_heads: Number of attention heads per layer.
        d_ff: FFN hidden dimension inside each Transformer layer.
        dropout: Dropout probability.

    Example:
        >>> dec = SelfAttentionDecoder(token_dim=256, n_obs=4, n_layers=4, n_heads=4, d_ff=512)
        >>> obs = torch.randn(2, 4, 256)
        >>> mp  = torch.randn(2, 1, 256)
        >>> ctx, attn = dec(obs, mp, return_attention=True)
        >>> ctx.shape       # (2, 256)
        >>> attn[0].shape   # (2, 5, 5) — per-layer full attention matrix
    """

    def __init__(
        self,
        token_dim: int = 256,
        n_obs: int = 4,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_obs = n_obs

        # ── Learnable token-type embeddings ────────────────────────────────────
        # Shape: (1, 1, d) each — broadcast over batch and sequence length.
        # Lets the model distinguish "this is an obs token" vs "this is a map token".
        self.obs_type_embed = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        self.map_type_embed = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)

        # ── Transformer encoder stack (self-attention only, pre-norm) ──────────
        self.transformer = _Encoder(
            lambda: _PreNormEncoderLayer(token_dim, n_heads, d_ff, dropout),
            n_layers,
        )

    def forward(
        self,
        obs_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
        return_attention: bool = False,
        return_per_head: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        """Fuse obs and map tokens via full self-attention and produce a context vector.

        Self-attention over the concatenated sequence means every obs token can
        attend to every map token (and vice-versa), giving the model maximum
        flexibility — at the cost of less explicit structure than cross-attention.

        Args:
            obs_tokens: Observation token sequence of shape (B, N_o, d).
            map_tokens: Map spatial token sequence of shape (B, N_m, d).
            return_attention: If True, return per-layer head-averaged attention
                matrices, each (B, N_o+N_m, N_o+N_m). Token order: obs then map.
            return_per_head: If True, return per-layer per-head attention
                (B, n_heads, N_o+N_m, N_o+N_m). Takes priority over
                return_attention.

        Returns:
            Tuple of:
                - context: Mean-pooled context vector of shape (B, d).
                - attention weights: List of per-layer tensors, or None if
                  neither return flag is set.
        """
        # Add token-type embeddings to mark origin of each token
        obs = obs_tokens + self.obs_type_embed   # (B, N_o, d)
        mp  = map_tokens + self.map_type_embed   # (B, N_m, d)

        # Concatenate along sequence dimension
        tokens = torch.cat([obs, mp], dim=1)     # (B, N_o + N_m, d)

        need_weights = return_attention or return_per_head
        attn_list: list[torch.Tensor] = []
        for layer in self.transformer.layers:
            tokens, attn_w = layer(
                tokens,
                need_weights=need_weights,
                average_attn_weights=not return_per_head,
            )
            if need_weights:
                attn_list.append(attn_w)

        # Extract obs positions only (first N_o tokens)
        obs_out = tokens[:, : self.n_obs, :]      # (B, N_o, d)

        context = obs_out.mean(dim=1)             # (B, d)
        return context, (attn_list if need_weights else None)
