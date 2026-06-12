"""GPS-noise degradation curve — the H3 (map-vision fusion) experiment.

Evaluates trained checkpoints on val while injecting increasing geometric
noise into the MAP input only (rotation = compass/heading error, translation
= GPS position error). Obs images are untouched.

Interpretation (decision experiment for "fix the model vs fix the eval"):
  - full degrades *slower* than map-only → the camera is already being used
    to compensate map errors (H3 evidence; architecture sufficient, the
    flat-noise on-route ADE was simply the wrong axis to show it).
  - full degrades *like* map-only → fusion is not happening; structural fix
    (patch tokens / map_tokens > 1 / encoder change) is warranted.
  - obs-only is the noise-independent control (flat line by construction).

Usage (on the machine with checkpoints + dataset):

    python scripts/eval_gps_noise_curve.py \
        --data-dir "/media/moai/MoAI Nav/dynav_frodo7k" \
        --ckpt full=outputs/.../checkpoints/best.pt \
        --ckpt maponly=outputs/.../checkpoints/best.pt \
        --ckpt obsonly=outputs/.../checkpoints/best.pt \
        [--levels "0:0,5:0.015,10:0.03,20:0.06,40:0.12"] [--batch-size 128]

Levels are "rot_deg:translate_frac" pairs; 0.03 ≈ ±3 m at zoom 19 (the
train-time augmentation calibration), so the default sweep spans 0 → ~4×
the assumed sensor noise. Outputs a table, gps_noise_curve.csv, and (if
matplotlib is available) gps_noise_curve.png.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynav.utils.metrics import compute_ade_fde  # noqa: E402

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# maneuver label → reporting bucket
_BUCKET = {
    "straight": "straight",
    "slight_left": "slight", "slight_right": "slight",
    "turn_left": "turn", "turn_right": "turn",
    "uturn": "uturn",
}


def _noise_map_transform(image_size: int, deg: float, trans: float):
    """Eval-time map transform with calibrated geometric noise."""
    tfs = [transforms.Resize((image_size, image_size))]
    if deg > 0 or trans > 0:
        tfs.append(transforms.RandomAffine(
            degrees=(-deg, deg),
            translate=(trans, trans) if trans > 0 else None,
        ))
    tfs += [transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)]
    return transforms.Compose(tfs)


def _load_model(ckpt_path: Path, device: torch.device):
    from omegaconf import OmegaConf
    from dynav.models.map_nav_model import DyNavModel

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    model = DyNavModel.from_config(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def _eval(model, loader, device) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    fde_sum, n = 0.0, 0
    for batch in loader:
        obs = batch["observations"].to(device)
        mp = batch["map_image"].to(device)
        gt = batch["gt_waypoints"].to(device)
        norm_m = batch["waypoint_norm_m"].to(device)
        out = model(obs, mp)
        ade, fde = compute_ade_fde(out["waypoints"].float(), gt, norm_m)
        fde_sum += fde.sum().item()
        n += gt.shape[0]
        for a, man in zip(ade.tolist(), batch["maneuver"]):
            b = _BUCKET.get(man, "other")
            sums[b] += a
            counts[b] += 1
            sums["all"] += a
            counts["all"] += 1
    row = {f"ade_{b}": sums[b] / counts[b] for b in counts}
    row["fde_all"] = fde_sum / max(n, 1)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt", action="append", required=True,
                    help="label=path/to/best.pt (repeatable)")
    ap.add_argument("--levels", default="0:0,5:0.015,10:0.03,20:0.06,40:0.12",
                    help="comma list of rot_deg:translate_frac")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", default="gps_noise_curve")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    levels = []
    for tok in args.levels.split(","):
        d, t = tok.split(":")
        levels.append((float(d), float(t)))

    models = {}
    for spec in args.ckpt:
        label, path = spec.split("=", 1)
        models[label], _ = _load_model(Path(path), device)
        print(f"[load] {label}: {path}")

    from dynav.data.dataset import DyNavDataset

    results: list[dict] = []
    for deg, trans in levels:
        torch.manual_seed(0)   # same noise draws for every model at this level
        ds = DyNavDataset(
            args.data_dir, split="val",
            map_transform=_noise_map_transform(224, deg, trans),
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        for label, model in models.items():
            torch.manual_seed(0)   # identical augmentation stream per model
            row = {"model": label, "rot_deg": deg, "trans_frac": trans,
                   **_eval(model, loader, device)}
            results.append(row)
            print(f"σ=({deg:>4.0f}°, {trans:.3f})  {label:<8} "
                  f"ADE={row['ade_all']:.3f}  turn={row.get('ade_turn', float('nan')):.3f}  "
                  f"uturn={row.get('ade_uturn', float('nan')):.3f}  "
                  f"straight={row.get('ade_straight', float('nan')):.3f}")

    # ── CSV ────────────────────────────────────────────────────────────────────
    keys = sorted({k for r in results for k in r}, key=lambda k: (k != "model", k))
    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"\nsaved {args.out}.csv")

    # ── Plot (optional) ───────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, ax in zip(("ade_all", "ade_turn"), axes):
        for label in models:
            xs = [r["rot_deg"] for r in results if r["model"] == label]
            ys = [r.get(metric) for r in results if r["model"] == label]
            ax.plot(xs, ys, "-o", label=label)
        ax.set_xlabel("map rotation noise σ (deg)  [translation scales with it]")
        ax.set_ylabel(f"val {metric} (m)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Map-noise degradation: full vs map-only (H3)")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    print(f"saved {args.out}.png")


if __name__ == "__main__":
    main()
