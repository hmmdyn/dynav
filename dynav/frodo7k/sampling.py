"""Candidate enumeration, stratified selection, and ride-level splitting.

Composition is controlled in two complementary places:

1. ``candidate_indices`` — *event-aware densification*: inside a segment,
   samples are taken every ``base_stride`` frames on straights but every
   ``turn_stride`` frames where the heading rate is high.  Turning data is
   rare and short-lived; densifying at source beats discarding straights.

2. ``select_balanced`` — *global stratified quota fill*: given all labelled
   candidates, fill per-(maneuver, env) bucket quotas derived from target
   fractions in the config.  Buckets that cannot meet their quota donate the
   remainder proportionally to the others, so the requested dataset size is
   met whenever enough candidates exist.

Train/val split is by *ride* (md5 hash), never by sample — segments of the
same ride share scenery, so a sample-level split would leak.
"""

from __future__ import annotations

import hashlib

import numpy as np


# ── Candidate enumeration ─────────────────────────────────────────────────────

def candidate_indices(
    heading: np.ndarray,
    seg_start: int,
    seg_end: int,
    obs_lookback: int,
    horizon_frames: int,
    base_stride: int = 20,
    turn_stride: int = 5,
    turn_rate_deg_s: float = 12.0,
    smooth_frames: int = 11,
) -> list[int]:
    """Frame offsets (within episode) eligible as sample anchors.

    A frame is valid if it has ``obs_lookback`` history and
    ``horizon_frames`` future inside the segment.  Stride is ``turn_stride``
    wherever the smoothed |heading rate| exceeds ``turn_rate_deg_s``,
    else ``base_stride``.
    """
    h = np.unwrap(heading[seg_start:seg_end])
    if len(h) < 2:
        return []
    rate = np.abs(np.gradient(h)) * 10.0                     # rad/s @10 fps
    k = np.ones(smooth_frames) / smooth_frames
    rate = np.convolve(rate, k, mode="same")
    turning = np.degrees(rate) > turn_rate_deg_s

    lo = seg_start + obs_lookback
    hi = seg_end - horizon_frames
    out, i = [], lo
    while i < hi:
        out.append(i)
        i += turn_stride if turning[i - seg_start] else base_stride
    return out


# ── Ride-level split ──────────────────────────────────────────────────────────

def ride_split(ride_id: str, val_fraction: float = 0.1) -> str:
    """Deterministic train/val assignment by ride id hash."""
    h = int(hashlib.md5(ride_id.encode()).hexdigest()[:8], 16)
    return "val" if (h % 1000) < int(val_fraction * 1000) else "train"


# ── Stratified selection ──────────────────────────────────────────────────────

def select_balanced(
    candidates: list[dict],
    target_total: int,
    maneuver_targets: dict[str, float],
    scene_targets: dict[str, float] | None = None,
    max_per_segment: int = 60,
    seed: int = 0,
) -> list[dict]:
    """Pick ≤ target_total candidates matching a 2-axis composition.

    Quotas form a (maneuver × scene) grid: quota[m][s] = total · w_m · w_s.
    Shortfall is redistributed in two passes — first across scenes within
    the same maneuver row (preserves the maneuver mix), then across
    maneuver rows (meets the requested total when candidates exist).

    Args:
        candidates:       Labelled candidate dicts; need keys ``maneuver``,
                          ``scene``, ``segment_key``.
        target_total:     Desired number of samples (per split).
        maneuver_targets: e.g. {"straight": .35, "slight": .20, "turn": .33,
                          "uturn": .12}.  "slight"/"turn" aggregate left+right.
        scene_targets:    e.g. {"city": .45, "park": .20, "straight_road": .20,
                          "other": .15}.  None → single "all" scene column.
        max_per_segment:  Cap per segment so one long ride cannot dominate.
        seed:             RNG seed (selection is deterministic).

    Returns:
        Selected candidate dicts (subset of input).
    """
    rng = np.random.default_rng(seed)

    def man_bucket(man: str) -> str:
        if man in maneuver_targets:
            return man
        base = man.split("_")[0]                 # slight_left → slight
        return base if base in maneuver_targets else man

    def scene_bucket(c: dict) -> str:
        if scene_targets is None:
            return "all"
        s = c.get("scene", "other")
        return s if s in scene_targets else "other"

    # cap per segment first (random, deterministic)
    by_seg: dict[str, list[dict]] = {}
    for c in candidates:
        by_seg.setdefault(c["segment_key"], []).append(c)
    capped: list[dict] = []
    for key in sorted(by_seg):
        group = by_seg[key]
        if len(group) > max_per_segment:
            sel = rng.choice(len(group), max_per_segment, replace=False)
            group = [group[i] for i in sorted(sel)]
        capped.extend(group)

    # (maneuver, scene) pools
    pools: dict[tuple[str, str], list[dict]] = {}
    for c in capped:
        pools.setdefault((man_bucket(c["maneuver"]), scene_bucket(c)), []).append(c)

    s_targets = scene_targets if scene_targets is not None else {"all": 1.0}
    mw = sum(maneuver_targets.values())
    sw = sum(s_targets.values())
    quotas: dict[tuple[str, str], int] = {
        (m, s): int(round(target_total * wm / mw * ws / sw))
        for m, wm in maneuver_targets.items()
        for s, ws in s_targets.items()
    }

    def _redistribute(groups: list[list[tuple[str, str]]]) -> None:
        """Move unmet quota to cells with spare candidates, group by group."""
        for cells in groups:
            short = sum(max(0, quotas[c] - len(pools.get(c, []))) for c in cells)
            if short <= 0:
                continue
            room = {c: len(pools.get(c, [])) - quotas[c] for c in cells}
            donors = {c: r for c, r in room.items() if r > 0}
            total_room = sum(donors.values())
            if total_room <= 0:
                continue
            give = min(short, total_room)
            for c, r in donors.items():
                quotas[c] += int(round(give * r / total_room))

    # pass 1: within each maneuver row (across scenes)
    _redistribute([[(m, s) for s in s_targets] for m in maneuver_targets])
    # pass 2: across everything (handles maneuver-level shortfall)
    _redistribute([list(quotas.keys())])

    selected: list[dict] = []
    for cell, quota in quotas.items():
        pool = pools.get(cell, [])
        take = min(quota, len(pool))
        if take <= 0:
            continue
        sel = rng.choice(len(pool), take, replace=False)
        selected += [pool[i] for i in sorted(sel)]

    return selected
