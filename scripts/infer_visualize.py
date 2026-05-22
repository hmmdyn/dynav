"""Inference visualization: compare predicted waypoints across checkpoints.

Layout per sample row:
    [obs_0 (current cam)] | [map+route image] | [BEV trajectory: GT vs checkpoints]

Usage (run from dynav/ repo root on the training server):
    python scripts/infer_visualize.py \\
        --data_dir /home/moai/dain_ws/dataset \\
        --checkpoints scripts/checkpoints/best.pt \\
                       scripts/checkpoints/epoch_0009.pt \\
                       scripts/checkpoints/epoch_0019.pt \\
                       scripts/checkpoints/epoch_0034.pt \\
        --n_samples 12 \\
        --output inference_viz.png

    # specific sample indices
    python scripts/infer_visualize.py --data_dir ... --sample_indices 0 5 10 50 100
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── colour palette ─────────────────────────────────────────────────────────────

_PALETTE = [
    ("#e6194b", "●"),   # red
    ("#3cb44b", "■"),   # green
    ("#4363d8", "▲"),   # blue
    ("#f58231", "◆"),   # orange
    ("#911eb4", "★"),   # purple
]
_GT_COLOR = "#000000"


# ── model helpers ──────────────────────────────────────────────────────────────

def _load_model(ckpt_path: Path, device: torch.device):
    from omegaconf import OmegaConf
    from dynav.models.map_nav_model import DyNavModel

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = OmegaConf.create(state["cfg"])
    model = DyNavModel.from_config(cfg).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, cfg, state.get("epoch", "?")


@torch.no_grad()
def _predict(model, obs_t: torch.Tensor, map_t: torch.Tensor, device) -> np.ndarray:
    """Return (H, 2) numpy array of predicted waypoints (normalized)."""
    obs_t = obs_t.unsqueeze(0).to(device)   # (1, N, 3, H, W)
    map_t = map_t.unsqueeze(0).to(device)   # (1, 3, H, W)
    out   = model(obs_t, map_t)
    return out["waypoints"][0].cpu().numpy()  # (H, 2)


# ── BEV plot ───────────────────────────────────────────────────────────────────

def _draw_bev(ax, gt_wp, pred_wps, labels, colors, max_dist: float, route_dir: float):
    """Draw bird's-eye-view trajectory comparison.

    Args:
        gt_wp:    (H, 2) GT waypoints, normalised.
        pred_wps: list of (H, 2) arrays, one per checkpoint.
        labels:   checkpoint labels.
        colors:   hex colour strings.
        max_dist: metres per normalised unit (max_waypoint_distance).
        route_dir: route direction in radians (body frame), for reference arrow.
    """
    # Convert normalised steps → accumulated positions in metres.
    def _accum(wp_norm):
        steps_m = wp_norm * max_dist          # (H, 2) in metres
        pos = np.zeros((len(steps_m) + 1, 2))
        pos[1:] = np.cumsum(steps_m, axis=0)
        return pos                             # (H+1, 2), starts at robot (0,0)

    gt_pos = _accum(gt_wp)

    ax.set_aspect("equal")
    ax.set_facecolor("#f8f8f8")
    ax.grid(True, linewidth=0.4, color="#cccccc")

    # Route direction reference arrow
    arrow_len = max_dist * 0.6
    ax.annotate(
        "", xy=(np.sin(route_dir) * arrow_len, np.cos(route_dir) * arrow_len),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1.2),
    )
    ax.text(np.sin(route_dir) * arrow_len * 1.1,
            np.cos(route_dir) * arrow_len * 1.1,
            "route", fontsize=6, color="#999999", ha="center")

    # Robot position
    ax.plot(0, 0, "o", color="#555555", ms=7, zorder=10)
    ax.text(0.1, -0.3, "robot", fontsize=6, color="#555555")

    # GT trajectory
    ax.plot(gt_pos[:, 1], gt_pos[:, 0],     # lateral (y→right), forward (x→up)
            "o-", color=_GT_COLOR, lw=2, ms=5, label="GT", zorder=8, alpha=0.9)

    # Predicted trajectories
    for pred_wp, label, color in zip(pred_wps, labels, colors):
        pred_pos = _accum(pred_wp)
        ax.plot(pred_pos[:, 1], pred_pos[:, 0],
                "o--", color=color, lw=1.4, ms=4, label=label, zorder=7, alpha=0.85)

    # Axis labels
    lim = max_dist * 2.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim * 0.5, lim * 1.5)
    ax.set_xlabel("lateral (m) →", fontsize=7)
    ax.set_ylabel("forward (m) ↑", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.7)


# ── image helpers ──────────────────────────────────────────────────────────────

def _tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalised tensor → (H, W, 3) uint8."""
    img = t.permute(1, 2, 0).numpy()
    lo, hi = img.min(), img.max()
    img = (img - lo) / max(hi - lo, 1e-8)
    return (img * 255).astype(np.uint8)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Checkpoint paths (best.pt first recommended)")
    parser.add_argument("--split",       default="val")
    parser.add_argument("--n_samples",   type=int, default=12,
                        help="Number of samples (evenly spread across split)")
    parser.add_argument("--sample_indices", nargs="+", type=int, default=None,
                        help="Explicit sample indices (overrides --n_samples)")
    parser.add_argument("--output",      default="inference_viz.png")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dynav.data.dataset import DyNavDataset
    from omegaconf import OmegaConf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoints ───────────────────────────────────────────────────────
    models, labels, colors = [], [], []
    max_waypoint_distance = 2.5  # default; overridden from first checkpoint cfg

    for i, ckpt_path in enumerate(args.checkpoints):
        model, cfg, epoch = _load_model(Path(ckpt_path), device)
        if i == 0:
            max_waypoint_distance = cfg.data.get("max_waypoint_distance", 2.5)
        label = f"ep{epoch} ({Path(ckpt_path).stem})"
        models.append(model)
        labels.append(label)
        colors.append(_PALETTE[i % len(_PALETTE)][0])
        print(f"  Loaded: {label}")

    # ── Dataset ────────────────────────────────────────────────────────────────
    # Use eval transforms (no augmentation) for consistent inference
    ds = DyNavDataset(args.data_dir, split=args.split,
                      image_size=cfg.data.image_size)
    n_total = len(ds)
    print(f"Dataset: {n_total} samples ({args.split})")

    if args.sample_indices:
        indices = args.sample_indices
    else:
        step = max(1, n_total // args.n_samples)
        indices = list(range(0, min(n_total, step * args.n_samples), step))

    n_rows = len(indices)
    n_cols = 3  # obs | map | BEV

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4.5, n_rows * 4.0),
        squeeze=False,
    )

    col_titles = ["Observation (t)", "Map + Route", "BEV Trajectory"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight="bold", pad=6)

    # ── Per-sample inference ───────────────────────────────────────────────────
    for row, idx in enumerate(indices):
        sample = ds[idx]
        obs_t  = sample["observations"]     # (N_obs, 3, H, W)
        map_t  = sample["map_image"]        # (3, H, W)
        gt_wp  = sample["gt_waypoints"].numpy()   # (H, 2)
        rdir   = float(sample["route_direction"])

        # Load meta for subtitle
        meta_path = ds.samples[idx] / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        subtitle = (
            f"#{idx}  ride={meta.get('ride_id','?')}  "
            f"hdg={meta.get('heading_deg',0):.0f}°  "
            f"snap={meta.get('osm_snap_mean_m','?')}m"
        )

        # col 0: current obs image
        ax = axes[row, 0]
        ax.imshow(_tensor_to_rgb(obs_t[0]))
        ax.set_ylabel(subtitle, fontsize=6, rotation=0, labelpad=4,
                      ha="right", va="center")
        ax.axis("off")

        # col 1: map image
        axes[row, 1].imshow(_tensor_to_rgb(map_t))
        axes[row, 1].axis("off")

        # col 2: BEV
        pred_wps = [_predict(m, obs_t, map_t, device) for m in models]
        _draw_bev(axes[row, 2], gt_wp, pred_wps, labels, colors,
                  max_waypoint_distance, rdir)

        print(f"  [{row+1}/{n_rows}] sample {idx} done")

    plt.suptitle(
        f"Inference Comparison — {args.split} split  |  "
        f"{', '.join(Path(p).stem for p in args.checkpoints)}",
        fontsize=10, fontweight="bold", y=1.002,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
