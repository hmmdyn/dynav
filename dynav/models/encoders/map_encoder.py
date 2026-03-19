"""Map encoder for OSM-rendered top-down map+path images.

Encodes a single map image via EfficientNet-B0 features (no GAP) to preserve
spatial layout, then projects the 7×7 feature grid to 49 tokens and adds a
2D positional encoding (learnable or sinusoidal).
"""

import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def _build_2d_sinusoidal_encoding(grid_size: int, dim: int) -> torch.Tensor:
    """Build 2D sinusoidal positional encoding for a grid_size × grid_size grid.

    Splits dim into two halves: first half encodes row position,
    second half encodes column position, using sinusoidal functions
    at geometrically spaced frequencies (following ViT/Transformer conventions).

    Args:
        grid_size: Spatial grid side length (7 for EfficientNet-B0).
        dim: Token embedding dimension. Must be divisible by 4
            (sin+cos for each of 2 axes).

    Returns:
        Positional encoding tensor of shape (grid_size * grid_size, dim).
    """
    assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"
    half_dim = dim // 2
    quarter_dim = dim // 4

    # Frequency bands (same as standard sinusoidal PE)
    omega = 1.0 / (10000.0 ** (torch.arange(quarter_dim).float() / quarter_dim))

    # Row and column indices
    rows = torch.arange(grid_size).float().unsqueeze(1)  # (G, 1)
    cols = torch.arange(grid_size).float().unsqueeze(1)  # (G, 1)

    # Sinusoidal encoding for rows: (G, quarter_dim) sin + (G, quarter_dim) cos
    row_enc = torch.cat([
        torch.sin(rows * omega.unsqueeze(0)),  # (G, quarter_dim)
        torch.cos(rows * omega.unsqueeze(0)),  # (G, quarter_dim)
    ], dim=-1)  # (G, half_dim)

    # Same for columns
    col_enc = torch.cat([
        torch.sin(cols * omega.unsqueeze(0)),
        torch.cos(cols * omega.unsqueeze(0)),
    ], dim=-1)  # (G, half_dim)

    # Combine: for each (row, col) pair, concat row_enc and col_enc
    # Result shape: (G*G, dim)
    pos_enc = torch.cat([
        row_enc.unsqueeze(1).expand(-1, grid_size, -1).reshape(-1, half_dim),
        col_enc.unsqueeze(0).expand(grid_size, -1, -1).reshape(-1, half_dim),
    ], dim=-1)  # (G*G, dim)

    return pos_enc


class MapEncoder(nn.Module):
    """Encodes a map+path image into a sequence of spatial token embeddings.

    Unlike VisualEncoder, the spatial 7×7 feature map is kept intact so that
    the cross-attention decoder can attend to specific map regions (e.g., where
    the route turns). Each of the 49 spatial locations is projected to token_dim
    and receives a 2D positional encoding.

    Args:
        token_dim: Dimension of output token embeddings (d).
        pretrained: Whether to load ImageNet pretrained EfficientNet-B0 weights.
        pos_enc_type: ``"learnable"`` (default) uses a trainable
            ``nn.Parameter``; ``"sinusoidal"`` uses a fixed 2D sinusoidal
            encoding registered as a buffer.

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
        pos_enc_type: str = "learnable",
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

        # ── 2D positional encoding ─────────────────────────────────────────────
        if pos_enc_type == "learnable":
            # Trainable parameter — shape (1, 49, token_dim)
            # Initialized small so early training is not dominated by pos enc
            self.pos_enc_2d = nn.Parameter(
                torch.randn(1, self.n_tokens, token_dim) * 0.02
            )
        elif pos_enc_type == "sinusoidal":
            # Fixed buffer — shape (1, 49, token_dim)
            enc = _build_2d_sinusoidal_encoding(self._SPATIAL_GRID, token_dim)
            self.register_buffer("pos_enc_2d", enc.unsqueeze(0))  # (1, 49, d)
        else:
            raise ValueError(
                f"Unknown pos_enc_type '{pos_enc_type}'. "
                "Expected 'learnable' or 'sinusoidal'."
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
