"""Visual encoder for egocentric observation images.

Encodes N_obs observation images independently via EfficientNet-B0 (GAP path),
projects each to a token vector, then adds learnable temporal positional encoding.
"""

import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class VisualEncoder(nn.Module):
    """Encodes a sequence of observation images into token embeddings.

    Each image is independently passed through EfficientNet-B0 features +
    AdaptiveAvgPool (GAP), then projected to token_dim. A learnable temporal
    positional encoding is added to distinguish frame order.

    Args:
        token_dim: Dimension of output token embeddings (d).
        n_obs: Number of observation frames (N_obs). Must match the second
            dimension of the input tensor at forward time.
        pretrained: Whether to load ImageNet pretrained EfficientNet-B0 weights.

    Example:
        >>> enc = VisualEncoder(token_dim=256, n_obs=4)
        >>> x = torch.randn(2, 4, 3, 224, 224)
        >>> tokens = enc(x)   # (2, 4, 256)
    """

    # EfficientNet-B0 feature output channels after features + avgpool
    _EFFICIENTNET_FEAT_DIM: int = 1280

    def __init__(
        self,
        token_dim: int = 256,
        n_obs: int = 4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.token_dim = token_dim
        self.n_obs = n_obs

        # ── Backbone ──────────────────────────────────────────────────────────
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        # Keep only features + avgpool; discard classifier
        self.features = backbone.features   # (B, 1280, 7, 7)
        self.avgpool = backbone.avgpool     # AdaptiveAvgPool2d → (B, 1280, 1, 1)

        # ── Projection head ───────────────────────────────────────────────────
        self.proj = nn.Linear(self._EFFICIENTNET_FEAT_DIM, token_dim)

        # ── Learnable temporal positional encoding ────────────────────────────
        # Shape: (1, N_obs, token_dim) — broadcast over batch dimension
        self.temporal_pos_enc = nn.Parameter(
            torch.randn(1, n_obs, token_dim) * 0.02
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of observation image sequences.

        Args:
            x: Observation images of shape (B, N_obs, 3, H, W).

        Returns:
            Token embeddings of shape (B, N_obs, token_dim).
        """
        B, N, C, H, W = x.shape  # (B, N_obs, 3, H, W)

        # Merge batch and frame dims so EfficientNet sees a flat batch
        x_flat = rearrange(x, "b n c h w -> (b n) c h w")  # (B*N, 3, H, W)

        feat = self.features(x_flat)        # (B*N, 1280, 7, 7)
        feat = self.avgpool(feat)           # (B*N, 1280, 1, 1)
        feat = feat.flatten(start_dim=1)    # (B*N, 1280)

        tokens = self.proj(feat)            # (B*N, token_dim)
        tokens = rearrange(tokens, "(b n) d -> b n d", b=B, n=N)  # (B, N_obs, token_dim)

        # Add temporal positional encoding
        tokens = tokens + self.temporal_pos_enc  # (B, N_obs, token_dim)

        return tokens

    # ── Freeze / unfreeze backbone ────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze EfficientNet-B0 feature extractor weights.

        Projection head and positional encoding remain trainable.
        Useful for the initial warm-up phase (see encoder.freeze_epochs in config).
        """
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze EfficientNet-B0 feature extractor weights for fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True
        for param in self.avgpool.parameters():
            param.requires_grad = True
