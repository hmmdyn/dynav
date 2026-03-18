"""Self-attention decoder for Map Navigation Model (ViNT-style baseline).

Concatenates obs and map tokens, applies full self-attention over the combined
sequence, then extracts obs positions for mean pooling. Used as an ablation
baseline against CrossAttentionDecoder.

Interface is identical to CrossAttentionDecoder for drop-in replacement.
"""

from typing import Optional

import torch
import torch.nn as nn


class SelfAttentionDecoder(nn.Module):
    """ViNT-style self-attention decoder (ablation baseline).

    Concatenates obs_tokens and map_tokens along the sequence dimension,
    adds learnable token-type embeddings to distinguish obs vs. map tokens,
    passes the combined sequence through a stack of Transformer encoder layers,
    then extracts only the obs positions and mean-pools them to a context vector.

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
        >>> mp  = torch.randn(2, 49, 256)
        >>> ctx, _ = dec(obs, mp)
        >>> ctx.shape   # (2, 256)
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

        # ── Transformer encoder stack (self-attention only) ────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-Norm (same convention as CrossAttentionDecoder)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )

    def forward(
        self,
        obs_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, None]:
        """Fuse obs and map tokens via full self-attention and produce a context vector.

        Self-attention over the concatenated sequence means every obs token can
        attend to every map token (and vice-versa), giving the model maximum
        flexibility — at the cost of less explicit structure than cross-attention.

        Args:
            obs_tokens: Observation token sequence of shape (B, N_o, d).
            map_tokens: Map spatial token sequence of shape (B, N_m, d).
            return_attention: Accepted for interface compatibility; always returns
                None because self-attention weights are not exposed by
                nn.TransformerEncoder without custom hooks.

        Returns:
            Tuple of:
                - context: Mean-pooled context vector of shape (B, d).
                - None: Placeholder to match CrossAttentionDecoder interface.
        """
        # Add token-type embeddings to mark origin of each token
        obs = obs_tokens + self.obs_type_embed   # (B, N_o, d)
        mp  = map_tokens + self.map_type_embed   # (B, N_m, d)

        # Concatenate along sequence dimension
        tokens = torch.cat([obs, mp], dim=1)     # (B, N_o + N_m, d)

        # Full self-attention over combined sequence
        tokens_out = self.transformer(tokens)    # (B, N_o + N_m, d)

        # Extract obs positions only (first N_o tokens)
        obs_out = tokens_out[:, : self.n_obs, :]  # (B, N_o, d)

        context = obs_out.mean(dim=1)             # (B, d)
        return context, None
