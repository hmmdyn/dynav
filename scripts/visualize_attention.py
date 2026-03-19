"""Visualize cross-attention weights from a trained DyNavModel checkpoint.

For each observation token (front_current, front_past1, front_past2, rear),
shows how strongly it attends to each 7×7 cell of the map across all decoder
layers.

Default layout (averaged):
    rows: N_obs observation tokens (4)
    cols: obs image | layer-0 heatmap | … | layer-{L-1} heatmap | map image

Per-head layout (--per-head):
    rows: N_obs observation tokens (4)
    cols: obs image | L0H0 | L0H1 | … | L{L-1}H{H-1} | map image
    Each cell is titled "Layer L, Head H".

Usage:
    python scripts/visualize_attention.py --checkpoint outputs/.../checkpoints/best.pt
    python scripts/visualize_attention.py --checkpoint best.pt --output viz.png --sample 2
    python scripts/visualize_attention.py --checkpoint best.pt --per-head
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
    """Load DyNavModel and config from a checkpoint file.

    Args:
        checkpoint: Path to ``.pt`` checkpoint written by ``train.py``.
        config_override: Optional path to a YAML config that overrides the
            config embedded in the checkpoint.

    Returns:
        Tuple of (model, cfg).
    """
    from omegaconf import OmegaConf
    from dynav.models.map_nav_model import DyNavModel

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if config_override:
        cfg = OmegaConf.load(config_override)
    elif "cfg" in ckpt:
        cfg = OmegaConf.create(ckpt["cfg"])
    else:
        raise ValueError(
            "Checkpoint has no embedded config. Provide --config <path>."
        )

    model = DyNavModel.from_config(cfg)
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
    per_head: bool = False,
) -> None:
    """Generate and save the cross-attention heatmap grid.

    Args:
        checkpoint: Path to trained model checkpoint.
        config: Optional YAML config override path.
        output: Output PNG file path.
        sample_idx: Which sample from DummyDyNavDataset to visualize.
        per_head: If True, show per-head heatmaps instead of averaged.
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

    # ── Build a sample from DummyDyNavDataset ──────────────────────────────────
    from dynav.data.dataset import DummyDyNavDataset

    n_obs = cfg.model.obs_context_length + 2
    ds = DummyDyNavDataset(
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
        out = model(obs, mp, return_attention=not per_head, return_per_head=per_head)

    waypoints = out["waypoints"][0].numpy()      # (H, 2) — for title display
    attn_list = out["attention_weights"]         # List[(1, N_o, 9)] or List[(1, n_heads, N_o, 9)]

    if attn_list is None:
        print(
            "[visualize_attention] No attention weights returned. "
            "Ensure the model uses CrossAttentionDecoder.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_layers = len(attn_list)
    obs_labels = ["front (t)", "front (t-1)", "front (t-2)", "rear (t)"]

    wp_str = "  ".join(f"({x:.2f},{y:.2f})" for x, y in waypoints)

    if per_head:
        # attn_list[layer]: (1, n_heads, N_o, 9)
        n_heads = attn_list[0].shape[1]
        _visualize_per_head(
            fig_title=(
                f"Per-Head Cross-Attention  |  Sample {sample_idx}\n"
                f"Predicted waypoints (Δx,Δy): {wp_str}"
            ),
            attn_list=attn_list,
            obs=obs,
            mp=mp,
            obs_labels=obs_labels,
            n_layers=n_layers,
            n_heads=n_heads,
            n_obs=n_obs,
            output=output,
            plt=plt,
        )
    else:
        _visualize_averaged(
            fig_title=(
                f"Cross-Attention Heatmaps  |  Sample {sample_idx}\n"
                f"Predicted waypoints (Δx,Δy): {wp_str}"
            ),
            attn_list=attn_list,
            obs=obs,
            mp=mp,
            obs_labels=obs_labels,
            n_layers=n_layers,
            n_obs=n_obs,
            output=output,
            plt=plt,
        )


def _visualize_averaged(
    fig_title: str,
    attn_list: list,
    obs: torch.Tensor,
    mp: torch.Tensor,
    obs_labels: list[str],
    n_layers: int,
    n_obs: int,
    output: str,
    plt,
) -> None:
    """Render and save head-averaged attention heatmaps."""
    # Columns: obs_img | layer-0 | … | layer-(n-1) | map_img
    n_cols = 1 + n_layers + 1
    fig, axes = plt.subplots(
        n_obs, n_cols,
        figsize=(2.8 * n_cols, 2.8 * n_obs),
        squeeze=False,
    )

    fig.suptitle(fig_title, fontsize=9)

    col_titles = (
        ["Observation"]
        + [f"Attn L{i}" for i in range(n_layers)]
        + ["Map+Route"]
    )
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=8, fontweight="bold", pad=4)

    map_display = _to_display(mp[0])

    for row, label in enumerate(obs_labels):
        axes[row, 0].imshow(_to_display(obs[0, row]))
        axes[row, 0].set_ylabel(label, fontsize=8, rotation=90, labelpad=3)
        axes[row, 0].axis("off")

        for layer_idx in range(n_layers):
            ax = axes[row, layer_idx + 1]
            # attn_list[layer_idx]: (1, N_o, 9) — select batch 0, obs row
            attn_flat = attn_list[layer_idx][0, row]          # (9,)
            attn_map  = attn_flat.reshape(3, 3).numpy()       # (3, 3)
            im = ax.imshow(attn_map, cmap="hot", vmin=0.0, vmax=attn_map.max())
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([])
            ax.set_yticks([])

        axes[row, -1].imshow(map_display)
        axes[row, -1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize_attention] Saved → {output}")


def _visualize_per_head(
    fig_title: str,
    attn_list: list,
    obs: torch.Tensor,
    mp: torch.Tensor,
    obs_labels: list[str],
    n_layers: int,
    n_heads: int,
    n_obs: int,
    output: str,
    plt,
) -> None:
    """Render and save per-head attention heatmaps.

    Layout: rows = obs tokens, cols = obs_img | L0H0 | L0H1 | … | L{L}H{H} | map_img
    """
    n_attn_cols = n_layers * n_heads
    n_cols = 1 + n_attn_cols + 1
    fig, axes = plt.subplots(
        n_obs, n_cols,
        figsize=(2.2 * n_cols, 2.2 * n_obs),
        squeeze=False,
    )

    fig.suptitle(fig_title, fontsize=9)

    # Column headers: obs | L0H0 L0H1 … | map
    col_titles = ["Observation"]
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            col_titles.append(f"L{layer_idx}H{head_idx}")
    col_titles.append("Map+Route")
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=7, fontweight="bold", pad=4)

    map_display = _to_display(mp[0])

    for row, label in enumerate(obs_labels):
        axes[row, 0].imshow(_to_display(obs[0, row]))
        axes[row, 0].set_ylabel(label, fontsize=7, rotation=90, labelpad=3)
        axes[row, 0].axis("off")

        col = 1
        for layer_idx in range(n_layers):
            # attn_list[layer_idx]: (1, n_heads, N_o, 9)
            for head_idx in range(n_heads):
                ax = axes[row, col]
                attn_flat = attn_list[layer_idx][0, head_idx, row]   # (9,)
                attn_map  = attn_flat.reshape(3, 3).numpy()           # (3, 3)
                im = ax.imshow(attn_map, cmap="hot", vmin=0.0, vmax=attn_map.max())
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xticks([])
                ax.set_yticks([])
                col += 1

        axes[row, -1].imshow(map_display)
        axes[row, -1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize_attention] Saved → {output}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize cross-attention weights of a trained DyNavModel."
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
        help="DummyDyNavDataset sample index to visualize (default: 0).",
    )
    parser.add_argument(
        "--per-head", action="store_true", default=False,
        help="Show per-head heatmaps instead of head-averaged (default: off).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    visualize(
        checkpoint=args.checkpoint,
        config=args.config,
        output=args.output,
        sample_idx=args.sample,
        per_head=args.per_head,
    )
