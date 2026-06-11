"""No-learning baselines for the frodo7k waypoint task.

Quantifies how much of val ADE/FDE is explained by trivial priors, i.e. the
floor against which learned models must be judged:

  zero        : predict "stay in place".
  train-mean  : predict the train-set mean waypoint trajectory (global prior,
                no inputs at all).
  route-dir   : analytic map baseline — train-mean step magnitudes laid out
                along the sample's route_direction (uses the route geometry
                but no camera and no learning).

If a learned model only marginally beats train-mean / route-dir, the metric
is saturated by motion-extrapolation priors and cannot evidence vision use.

Usage (numpy only, runs anywhere the built dataset is mounted):

    python scripts/eval_trivial_baselines.py --data-dir /path/to/dynav_frodo7k \
        [--limit-train 20000]
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_split(split_dir: Path, limit: int | None = None):
    """Returns (wp_m (N,H,2), route_dir (N,), maneuver list[N])."""
    wps, rds, mans = [], [], []
    dirs = sorted(d for d in split_dir.iterdir() if (d / "meta.json").exists())
    if limit:
        dirs = dirs[:limit]
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text())
        wps.append(meta["gt_waypoints_m"])
        rds.append(meta["route_direction"])
        mans.append(meta.get("labels", {}).get("maneuver", "unknown"))
    return np.asarray(wps, dtype=np.float64), np.asarray(rds), mans


def _report(name: str, pred: np.ndarray, gt: np.ndarray, mans: list[str]) -> None:
    de = np.linalg.norm(pred - gt, axis=-1)          # (N, H) meters
    ade, fde = de.mean(axis=1), de[:, -1]
    by_man: dict[str, list[float]] = defaultdict(list)
    for a, m in zip(ade, mans):
        by_man[m].append(a)
    strat = "  ".join(f"{m}={np.mean(v):.2f}" for m, v in sorted(by_man.items()))
    print(f"{name:<12} ADE={ade.mean():.3f}m  FDE={fde.mean():.3f}m  | {strat}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--limit-train", type=int, default=20000,
                    help="train samples for the mean prior (default 20000)")
    args = ap.parse_args()

    root = Path(args.data_dir)
    train_wp, _, _ = _load_split(root / "train", args.limit_train)
    val_wp, val_rd, val_man = _load_split(root / "val")
    n, h = val_wp.shape[:2]
    print(f"train prior from {len(train_wp)} samples; eval on {n} val samples\n")

    # 1. zero — stay in place
    _report("zero", np.zeros_like(val_wp), val_wp, val_man)

    # 2. train-mean — global prior, input-blind
    mean_wp = train_wp.mean(axis=0)                  # (H, 2)
    _report("train-mean", np.broadcast_to(mean_wp, val_wp.shape), val_wp, val_man)

    # 3. route-dir — train-mean step radii along the sample's route direction
    radii = np.linalg.norm(train_wp, axis=-1).mean(axis=0)   # (H,)
    dir_vec = np.stack([np.cos(val_rd), np.sin(val_rd)], axis=-1)  # (N, 2)
    pred = radii[None, :, None] * dir_vec[:, None, :]              # (N, H, 2)
    _report("route-dir", pred, val_wp, val_man)

    print("\n해석: 학습 모델의 val ADE가 train-mean/route-dir과 비슷하면 지표가"
          " 외삽 prior로 포화된 것 — vision 기여는 이 지표로 입증 불가.")


if __name__ == "__main__":
    main()
