"""Verify frame-rate/timing assumptions of a built frodo7k dataset.

GT waypoints are *index*-parameterized (wp_offsets frames at nominal 10 fps),
but segment QA only splits on video-timestamp gaps > video_gap_s (0.25 s) —
steps between 0.1 s and 0.25 s pass silently, stretching the actual elapsed
time of the "1–5 s" horizon. This script measures, for every built sample,
the true elapsed time of each waypoint offset and each obs stride from the
zarr timestamps, and reports the deviation distribution.

Run on the machine that has the zarr (e.g. the 5090):

    python scripts/verify_frodo7k_timing.py --config configs/frodo7k.yaml \
        --split val [--split train] [--limit 5000]

Verdict: PASS if <1% of samples deviate >10% from nominal; otherwise lists
the worst offenders so they can be gated in a rebuild.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynav.frodo7k.reader import Frodo7kReader  # noqa: E402

_SAMPLE_RE = re.compile(r"ep(\d{6})_(\d{6})$")


def _collect_samples(split_dir: Path, limit: int | None) -> dict[int, list[int]]:
    """{episode: [anchor idx, ...]} parsed from sample directory names."""
    by_ep: dict[int, list[int]] = defaultdict(list)
    dirs = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
    if limit:
        dirs = dirs[:limit]
    for name in dirs:
        m = _SAMPLE_RE.search(name)
        if m:
            by_ep[int(m.group(1))].append(int(m.group(2)))
    return by_ep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/frodo7k.yaml")
    ap.add_argument("--split", action="append", default=None,
                    help="dataset split(s) to check (default: val)")
    ap.add_argument("--limit", type=int, default=None,
                    help="max samples per split (default: all)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    scfg = cfg["sample"]
    wp_offsets = list(scfg["wp_offsets"])
    obs_stride = int(scfg["obs_stride"])
    n_obs = int(scfg["n_obs"])
    out_root = Path(cfg["paths"]["output_root"])

    reader = Frodo7kReader(cfg["paths"]["frodo7k_root"],
                           bounds_cache=out_root / "ep_bounds.npy")

    splits = args.split or ["val"]
    # dt[offset_frames] = list of measured elapsed seconds
    wp_dt: dict[int, list[float]] = {off: [] for off in wp_offsets}
    obs_dt: list[float] = []           # full obs window (idx-15 → idx)
    n_samples = 0
    worst: list[tuple[float, str]] = []

    for split in splits:
        split_dir = out_root / split
        if not split_dir.exists():
            print(f"[skip] {split_dir} not found")
            continue
        by_ep = _collect_samples(split_dir, args.limit)
        print(f"[{split}] {sum(len(v) for v in by_ep.values())} samples "
              f"in {len(by_ep)} episodes")

        for e, idxs in sorted(by_ep.items()):
            ep = reader.episode(e)
            ts = ep.video_ts
            for idx in idxs:
                n_samples += 1
                for off in wp_offsets:
                    j = min(idx + off, ep.n - 1)
                    wp_dt[off].append(float(ts[j] - ts[idx]))
                lb = obs_stride * (n_obs - 1)
                obs_dt.append(float(ts[idx] - ts[max(idx - lb, 0)]))
                dev = abs(wp_dt[wp_offsets[-1]][-1] - wp_offsets[-1] / 10.0)
                if dev > 0.5:
                    worst.append((dev, f"{split}/ep{e:06d}_{idx:06d}"))

    if not n_samples:
        print("No samples found.")
        sys.exit(1)

    print(f"\n=== timing report ({n_samples} samples) ===")
    print(f"{'offset':>8} {'nominal':>8} {'mean':>7} {'p50':>7} {'p95':>7} "
          f"{'max':>7} {'>10% dev':>9} {'>25% dev':>9}")
    bad_total = 0
    for off in wp_offsets:
        arr = np.array(wp_dt[off])
        nom = off / 10.0
        dev = np.abs(arr - nom) / nom
        bad10 = float((dev > 0.10).mean())
        bad25 = float((dev > 0.25).mean())
        bad_total = max(bad_total, bad10)
        print(f"wp+{off:<5} {nom:>7.1f}s {arr.mean():>6.2f}s {np.median(arr):>6.2f}s "
              f"{np.percentile(arr, 95):>6.2f}s {arr.max():>6.2f}s "
              f"{bad10:>8.2%} {bad25:>8.2%}")

    arr = np.array(obs_dt)
    nom = obs_stride * (n_obs - 1) / 10.0
    print(f"obs win  {nom:>7.1f}s {arr.mean():>6.2f}s {np.median(arr):>6.2f}s "
          f"{np.percentile(arr, 95):>6.2f}s {arr.max():>6.2f}s")

    if worst:
        worst.sort(reverse=True)
        print(f"\nworst {min(10, len(worst))} samples (|Δt - nominal| at wp+{wp_offsets[-1]}):")
        for dev, name in worst[:10]:
            print(f"  +{dev:.2f}s  {name}")

    verdict = "PASS" if bad_total < 0.01 else "FAIL — consider time-based wp gating in rebuild"
    print(f"\nverdict: {verdict} (worst >10% deviation rate: {bad_total:.2%})")


if __name__ == "__main__":
    main()
