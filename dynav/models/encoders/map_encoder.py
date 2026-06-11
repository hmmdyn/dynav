"""Map encoder for OSM-rendered top-down map+path images.

Encodes a single map image via EfficientNet-B0 features, then pools to a
configurable number of spatial tokens:

- n_tokens=1  (default): GlobalAvgPool — fully shift-invariant. Robust to
  GPS/heading/centerline offsets (which appear as displacement in heading-up
  maps) but also erases the *signal* component of displacement (robot-to-route
  offset, distance-to-turn). Adopted 2026-05-19.
- n_tokens=9  (3×3) / n_tokens=49 (7×7): spatial grid tokens + learnable 2D
  positional encoding. Preserves coarse layout at the cost of shift
  sensitivity. Whether invariance helps or hurts is an open empirical
  question — these variants exist for the token-count ablation
  (ablation_map_tokens9/49 configs).
"""

import math

import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class MapEncoder(nn.Module):
    """Encodes a map+path image into one or more token embeddings.

    EfficientNet-B0 features (B, 1280, 7, 7) are pooled to an n×n grid
    (n = sqrt(n_tokens)), flattened to tokens, and projected to token_dim.
    For n_tokens > 1 a learnable 2D positional encoding is added so the
    decoder can distinguish grid positions.

    Args:
        token_dim: Dimension of output token embedding (d).
        pretrained: Whether to load ImageNet pretrained EfficientNet-B0 weights.
        n_tokens: Number of output tokens — 1 (GAP), 9 (3×3), or 49 (7×7).

    Example:
        >>> enc = MapEncoder(token_dim=256)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> tokens = enc(x)   # (2, 1, 256)
        >>> enc9 = MapEncoder(token_dim=256, n_tokens=9)
        >>> enc9(x).shape     # (2, 9, 256)
    """

    _EFFICIENTNET_FEAT_DIM: int = 1280
    _VALID_N_TOKENS = (1, 9, 49)

    def __init__(
        self,
        token_dim: int = 256,
        pretrained: bool = True,
        n_tokens: int = 1,
    ) -> None:
        super().__init__()

        if n_tokens not in self._VALID_N_TOKENS:
            raise ValueError(
                f"n_tokens must be one of {self._VALID_N_TOKENS}, got {n_tokens}"
            )

        self.token_dim = token_dim
        self.n_tokens = n_tokens
        self._grid = int(math.isqrt(n_tokens))

        # ── Backbone (features only — no avgpool, no classifier) ───────────────
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features   # output: (B, 1280, 7, 7)

        # ── Pooling: 7×7 → grid×grid ──────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d((self._grid, self._grid))

        # ── Projection head ───────────────────────────────────────────────────
        self.proj = nn.Linear(self._EFFICIENTNET_FEAT_DIM, token_dim)

        # ── Learnable 2D positional encoding (spatial variants only) ──────────
        if n_tokens > 1:
            self.pos_enc = nn.Parameter(torch.randn(1, n_tokens, token_dim) * 0.02)
        else:
            self.pos_enc = None

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of map images into spatial token embeddings.

        Args:
            x: Map+path images of shape (B, 3, H, W).

        Returns:
            Token embeddings of shape (B, n_tokens, token_dim).
        """
        feat = self.features(x)                             # (B, 1280, 7, 7)
        feat = self.gap(feat)                               # (B, 1280, g, g)
        feat = rearrange(feat, "b c gh gw -> b (gh gw) c")  # (B, n_tokens, 1280)
        tokens = self.proj(feat)                            # (B, n_tokens, token_dim)
        if self.pos_enc is not None:
            tokens = tokens + self.pos_enc
        return tokens

    # ── Freeze / unfreeze backbone ────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze EfficientNet-B0 feature extractor weights."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze EfficientNet-B0 feature extractor weights for fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True
