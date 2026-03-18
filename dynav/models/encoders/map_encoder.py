"""Map encoder for OSM-rendered top-down map+path images.

Encodes a single map image via EfficientNet-B0 features (no GAP) to preserve
spatial layout, then projects the 7×7 feature grid to 49 tokens and adds a
learnable 2D positional encoding.
"""

import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class MapEncoder(nn.Module):
    """Encodes a map+path image into a sequence of spatial token embeddings.

    Unlike VisualEncoder, the spatial 7×7 feature map is kept intact so that
    the cross-attention decoder can attend to specific map regions (e.g., where
    the route turns). Each of the 49 spatial locations is projected to token_dim
    and receives a learnable 2D positional encoding.

    Args:
        token_dim: Dimension of output token embeddings (d).
        pretrained: Whether to load ImageNet pretrained EfficientNet-B0 weights.

    Example:
        >>> enc = MapEncoder(token_dim=256)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> tokens = enc(x)   # (2, 49, 256)
    """

    # EfficientNet-B0 spatial output: (B, 1280, 7, 7)
    _EFFICIENTNET_FEAT_DIM: int = 1280
    _SPATIAL_GRID: int = 7          # 7 × 7 = 49 tokens

    def __init__(
        self,
        token_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.token_dim = token_dim
        self.n_tokens = self._SPATIAL_GRID ** 2  # 49

        # ── Backbone (features only — no avgpool, no classifier) ───────────────
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features   # output: (B, 1280, 7, 7)

        # ── Projection head ───────────────────────────────────────────────────
        self.proj = nn.Linear(self._EFFICIENTNET_FEAT_DIM, token_dim)

        # ── Learnable 2D positional encoding ──────────────────────────────────
        # Shape: (1, 49, token_dim) — broadcast over batch dimension
        # Initialized small so early training is not dominated by pos enc
        self.pos_enc_2d = nn.Parameter(
            torch.randn(1, self.n_tokens, token_dim) * 0.02
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of map images into spatial token sequences.

        Args:
            x: Map+path images of shape (B, 3, H, W).

        Returns:
            Spatial token embeddings of shape (B, 49, token_dim).
        """
        # x: (B, 3, H, W)
        feat = self.features(x)                             # (B, 1280, 7, 7)

        # Flatten spatial dims → token sequence
        tokens = rearrange(feat, "b c h w -> b (h w) c")   # (B, 49, 1280)

        tokens = self.proj(tokens)                          # (B, 49, token_dim)

        # Add 2D positional encoding
        tokens = tokens + self.pos_enc_2d                   # (B, 49, token_dim)

        return tokens

    # ── Freeze / unfreeze backbone ────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze EfficientNet-B0 feature extractor weights.

        Projection head and positional encoding remain trainable.
        Useful for the initial warm-up phase (see encoder.freeze_epochs in config).
        """
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze EfficientNet-B0 feature extractor weights for fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True
