"""Map encoder for OSM-rendered top-down map+path images.

Encodes a single map image via EfficientNet-B0 features, then collapses
to a single global token via GlobalAvgPool (AdaptiveAvgPool2d(1,1)).
Single-token GAP is more robust to GPS, heading, and path centerline offset
errors than spatial grids, since all real-world noise sources cause primarily
horizontal displacement in heading-up rendered maps.
"""

import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class MapEncoder(nn.Module):
    """Encodes a map+path image into a single global token embedding.

    EfficientNet-B0 features (B, 1280, 7, 7) are collapsed to a single
    1280-d vector via GlobalAvgPool, then projected to token_dim.
    No positional encoding is used (single token has no spatial position).

    This design is robust to the dominant noise sources in real navigation:
    GPS position error, IMU/compass heading error, and path centerline offset
    all cause horizontal displacement in heading-up map images —
    GlobalAvgPool is invariant to such shifts.

    Args:
        token_dim: Dimension of output token embedding (d).
        pretrained: Whether to load ImageNet pretrained EfficientNet-B0 weights.

    Example:
        >>> enc = MapEncoder(token_dim=256)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> tokens = enc(x)   # (2, 1, 256)
    """

    _EFFICIENTNET_FEAT_DIM: int = 1280

    def __init__(
        self,
        token_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.token_dim = token_dim
        self.n_tokens = 1

        # ── Backbone (features only — no avgpool, no classifier) ───────────────
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features   # output: (B, 1280, 7, 7)

        # ── Global average pooling: 7×7 → 1×1 ────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # ── Projection head ───────────────────────────────────────────────────
        self.proj = nn.Linear(self._EFFICIENTNET_FEAT_DIM, token_dim)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of map images into a single global token.

        Args:
            x: Map+path images of shape (B, 3, H, W).

        Returns:
            Global token embedding of shape (B, 1, token_dim).
        """
        feat = self.features(x)                             # (B, 1280, 7, 7)
        feat = self.gap(feat)                               # (B, 1280, 1, 1)
        feat = rearrange(feat, "b c 1 1 -> b 1 c")         # (B, 1, 1280)
        tokens = self.proj(feat)                            # (B, 1, token_dim)
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
