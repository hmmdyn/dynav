"""Map Navigation Model — top-level assembly.

Wires together VisualEncoder, MapEncoder, a config-selected Decoder,
and WaypointHead into a single nn.Module.
"""

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from dynav.models.decoders import (
    CrossAttentionDecoder,
    SelfAttentionDecoder,
    WaypointHead,
)
from dynav.models.encoders import MapEncoder, VisualEncoder


class dynavModel(nn.Module):
    """End-to-end Map Navigation Model.

    Accepts a Hydra/OmegaConf DictConfig that follows the schema in
    ``configs/default.yaml`` and assembles the full encoder → decoder →
    action-head pipeline.

    Args:
        cfg: OmegaConf DictConfig with keys ``model``, ``encoder``,
            ``decoder``, and ``action_head``.

    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.load("configs/default.yaml")
        >>> model = dynavModel(cfg)
        >>> obs = torch.randn(2, 4, 3, 224, 224)
        >>> mp  = torch.randn(2, 3, 224, 224)
        >>> out = model(obs, mp)
        >>> out["waypoints"].shape   # (2, 5, 2)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        # ── Config aliases ─────────────────────────────────────────────────────
        token_dim: int = cfg.model.token_dim
        # N_obs: current front + K past front frames + current rear
        n_obs: int = cfg.model.obs_context_length + 2
        horizon: int = cfg.model.prediction_horizon
        pretrained: bool = cfg.encoder.pretrained

        d_dec = cfg.decoder
        d_head = cfg.action_head

        # ── Encoders ───────────────────────────────────────────────────────────
        self.visual_encoder = VisualEncoder(
            token_dim=token_dim,
            n_obs=n_obs,
            pretrained=pretrained,
        )
        self.map_encoder = MapEncoder(
            token_dim=token_dim,
            pretrained=pretrained,
        )

        # ── Decoder (selected by config) ───────────────────────────────────────
        if d_dec.type == "cross_attention":
            self.decoder: nn.Module = CrossAttentionDecoder(
                token_dim=token_dim,
                n_layers=d_dec.n_layers,
                n_heads=d_dec.n_heads,
                d_ff=d_dec.d_ff,
                dropout=d_dec.dropout,
            )
        elif d_dec.type == "self_attention":
            self.decoder = SelfAttentionDecoder(
                token_dim=token_dim,
                n_obs=n_obs,
                n_layers=d_dec.n_layers,
                n_heads=d_dec.n_heads,
                d_ff=d_dec.d_ff,
                dropout=d_dec.dropout,
            )
        else:
            raise ValueError(
                f"Unknown decoder type '{d_dec.type}'. "
                "Expected 'cross_attention' or 'self_attention'."
            )

        # ── Action head ────────────────────────────────────────────────────────
        self.waypoint_head = WaypointHead(
            token_dim=token_dim,
            hidden_dim=d_head.hidden_dim,
            prediction_horizon=horizon,
        )

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        observations: torch.Tensor,
        map_image: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Run the full navigation pipeline.

        Args:
            observations: Stacked observation images of shape
                (B, N_obs, 3, H, W).
            map_image: OSM map+path image of shape (B, 3, H, W).
            return_attention: If True and decoder is CrossAttentionDecoder,
                include per-layer attention weights in the output dict.

        Returns:
            Dictionary with:
                - ``"waypoints"``: Predicted waypoints (B, H, 2) in [-1, 1].
                - ``"attention_weights"``: List of (B, N_o, N_m) tensors
                  (one per layer) if return_attention and cross-attention
                  decoder is used, else None.
        """
        obs_tokens = self.visual_encoder(observations)    # (B, N_obs, d)
        map_tokens = self.map_encoder(map_image)          # (B, 49, d)

        context, attn_weights = self.decoder(
            obs_tokens, map_tokens, return_attention=return_attention
        )                                                 # context: (B, d)

        waypoints = self.waypoint_head(context)           # (B, H, 2)

        return {"waypoints": waypoints, "attention_weights": attn_weights}

    # ── Encoder freeze helpers ─────────────────────────────────────────────────

    def freeze_encoders(self) -> None:
        """Freeze EfficientNet-B0 backbones in both encoders.

        Useful during the initial warm-up phase controlled by
        ``encoder.freeze_epochs`` in the training config.
        Projection heads and positional encodings remain trainable.
        """
        self.visual_encoder.freeze_backbone()
        self.map_encoder.freeze_backbone()

    def unfreeze_encoders(self) -> None:
        """Unfreeze EfficientNet-B0 backbones in both encoders for fine-tuning."""
        self.visual_encoder.unfreeze_backbone()
        self.map_encoder.unfreeze_backbone()

    # ── Parameter counting ─────────────────────────────────────────────────────

    def count_parameters(self) -> dict[str, int]:
        """Count trainable parameters per component and total.

        Returns:
            Dictionary with keys ``"visual_encoder"``, ``"map_encoder"``,
            ``"decoder"``, ``"waypoint_head"``, and ``"total"``, each
            mapping to the number of trainable parameters in that component.
        """
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {
            "visual_encoder": _count(self.visual_encoder),
            "map_encoder":    _count(self.map_encoder),
            "decoder":        _count(self.decoder),
            "waypoint_head":  _count(self.waypoint_head),
        }
        counts["total"] = sum(counts.values())
        return counts

    def print_parameter_summary(self) -> None:
        """Print a formatted summary of trainable parameter counts."""
        counts = self.count_parameters()
        width = 20
        print("─" * (width + 15))
        for name, n in counts.items():
            if name == "total":
                print("─" * (width + 15))
            print(f"  {name:<{width}} {n:>12,}")
        print("─" * (width + 15))

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "dynavModel":
        """Construct a dynavModel from an OmegaConf DictConfig.

        Args:
            cfg: Config loaded via ``OmegaConf.load`` or Hydra.

        Returns:
            Instantiated dynavModel.
        """
        return cls(cfg)
