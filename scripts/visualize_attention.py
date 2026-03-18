"""Visualize cross-attention weights from a trained dynavModel checkpoint.

For each observation token (front_current, front_past1, front_past2, rear),
shows how strongly it attends to each 7×7 cell of the map across all decoder
layers.

Layout:
    rows: N_obs observation tokens (4)
    cols: obs image | layer-0 heatmap | … | layer-{L-1} heatmap | map image

Usage:
    python scripts/visualize_attention.py --checkpoint outputs/.../checkpoints/best.pt
    python scripts/visualize_attention.py --checkpoint best.pt --output viz.png --sample 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Utilities ──────────────────────────────────────────────────────────────────

def _load_model_and_cfg(checkpoint: str, config_override: str | None):
    """Load dynavModel and config from a checkpoint file.

    Args:
        checkpoint: Path to ``.pt`` checkpoint written by ``train.py``.
        config_override: Optional path to a YAML config that overrides the
            config embedded in the checkpoint.

    Returns:
        Tuple of (model, cfg).
    """
    from omegaconf import OmegaConf
    from dynav.models.map_nav_model import dynavModel

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if config_override:
        cfg = OmegaConf.load(config_override)
    elif "cfg" in ckpt:
        cfg = OmegaConf.create(ckpt["cfg"])
    else:
        raise ValueError(
            "Checkpoint has no embedded config. Provide --config <path>."
        )

    model = dynavModel.from_config(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def _to_display(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) tensor → (H, W, 3) numpy array in [0, 1]."""
    img = t.permute(1, 2, 0).numpy()
    lo, hi = img.min(), img.max()
    return (img - lo) / max(hi - lo, 1e-8)


# ── Core visualization ─────────────────────────────────────────────────────────

def visualize(
    checkpoint: str,
    config: str | None = None,
    output: str = "attention_viz.png",
    sample_idx: int = 0,
) -> None:
    """Generate and save the cross-attention heatmap grid.

    Args:
        checkpoint: Path to trained model checkpoint.
        config: Optional YAML config override path.
        output: Output PNG file path.
        sample_idx: Which sample from DummydynavDataset to visualize.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for CI / headless servers
    import matplotlib.pyplot as plt

    model, cfg = _load_model_and_cfg(checkpoint, config)

    if cfg.decoder.type != "cross_attention":
        print(
            f"[visualize_attention] Decoder type is '{cfg.decoder.type}'. "
            "Attention visualization requires 'cross_attention' decoder.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Build a sample from DummydynavDataset ──────────────────────────────────
    from dynav.data.dataset import DummydynavDataset

    n_obs = cfg.model.obs_context_length + 2
    ds = DummydynavDataset(
        size=max(sample_idx + 1, 10),
        n_obs=n_obs,
        image_size=cfg.data.image_size,
        horizon=cfg.model.prediction_horizon,
        seed=0,
    )
    sample = ds[sample_idx]
    obs = sample["observations"].unsqueeze(0)   # (1, N_obs, 3, H, W)
    mp  = sample["map_image"].unsqueeze(0)       # (1, 3, H, W)

    # ── Forward pass ───────────────────────────────────────────────────────────
    with torch.no_grad():
        out = model(obs, mp, return_attention=True)

    waypoints = out["waypoints"][0].numpy()      # (H, 2) — for title display
    attn_list = out["attention_weights"]         # List[(1, N_o, 49)]

    if attn_list is None:
        print(
            "[visualize_attention] No attention weights returned. "
            "Ensure the model uses CrossAttentionDecoder.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_layers = len(attn_list)
    obs_labels = ["front (t)", "front (t-1)", "front (t-2)", "rear (t)"]

    # ── Build figure ───────────────────────────────────────────────────────────
    # Columns: obs_img | layer-0 | … | layer-(n-1) | map_img
    n_cols = 1 + n_layers + 1
    fig, axes = plt.subplots(
        n_obs, n_cols,
        figsize=(2.8 * n_cols, 2.8 * n_obs),
        squeeze=False,
    )

    wp_str = "  ".join(f"({x:.2f},{y:.2f})" for x, y in waypoints)
    fig.suptitle(
        f"Cross-Attention Heatmaps  |  Sample {sample_idx}\n"
        f"Predicted waypoints (Δx,Δy): {wp_str}",
        fontsize=9,
    )

    # Column headers (top row only)
    col_titles = (
        ["Observation"]
        + [f"Attn L{i}" for i in range(n_layers)]
        + ["Map+Route"]
    )
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=8, fontweight="bold", pad=4)

    map_display = _to_display(mp[0])             # (H, W, 3) for reuse across rows

    for row, label in enumerate(obs_labels):
        # Column 0: observation image
        axes[row, 0].imshow(_to_display(obs[0, row]))
        axes[row, 0].set_ylabel(label, fontsize=8, rotation=90, labelpad=3)
        axes[row, 0].axis("off")

        # Columns 1 … n_layers: attention heatmaps
        for layer_idx in range(n_layers):
            ax = axes[row, layer_idx + 1]
            # attn_list[layer_idx]: (1, N_o, 49) — select batch 0, obs row
            attn_flat = attn_list[layer_idx][0, row]          # (49,)
            attn_map  = attn_flat.reshape(7, 7).numpy()       # (7, 7)

            im = ax.imshow(attn_map, cmap="hot", vmin=0.0, vmax=attn_map.max())
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([])
            ax.set_yticks([])

        # Last column: map image (same for every row)
        axes[row, -1].imshow(map_display)
        axes[row, -1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize_attention] Saved → {output}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize cross-attention weights of a trained dynavModel."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to .pt checkpoint file (produced by scripts/train.py).",
    )
    parser.add_argument(
        "--config", default=None,
        help="Optional YAML config override (default: use config embedded in checkpoint).",
    )
    parser.add_argument(
        "--output", default="attention_viz.png",
        help="Output image file path (default: attention_viz.png).",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="DummydynavDataset sample index to visualize (default: 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    visualize(
        checkpoint=args.checkpoint,
        config=args.config,
        output=args.output,
        sample_idx=args.sample,
    )
