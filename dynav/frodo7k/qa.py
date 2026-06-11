"""Segment refinement and quality gates for FrodoBots-7k episodes.

Two layers of automated QA, both with *explicit, recorded reasons* so that
every accepted/rejected segment is auditable from the index files:

1. ``refine_segments`` — splits an episode into clean sub-segments on hard
   discontinuities (video timestamp gaps, long stops, EKF↔GPS divergence)
   and trims stationary head/tail.  This replaces the legacy
   ``segment_gps_episode`` heuristics that operated on interpolated 1 Hz GPS.

2. ``segment_qa`` — metric gates on each refined segment.  Notably there is
   **no straightness gate**: the legacy ``straightness >= 0.75`` filter
   removed exactly the intersection-turn data the model needs.  Turn-rich
   segments are kept and balanced later in sampling.

The decisive new gate is ``heading_err_med``: median angular difference
between the EKF heading and the actual motion direction on moving frames.
A segment whose heading disagrees with its own motion produces wrong
body-frame waypoints *by construction* — this single gate removes the
"ground-truth waypoints don't match" failure mode at the source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from .reader import EpisodeData


# ── Tunables (overridden via configs/frodo7k.yaml → build script) ────────────

@dataclass
class QAConfig:
    # refinement
    video_gap_s: float = 0.25        # split when front-video ts jumps more than this
    stop_speed_ms: float = 0.15      # below this = stationary
    stop_split_s: float = 5.0        # internal stop longer than this → split
    gps_dev_split_m: float = 15.0    # sustained EKF↔GPS divergence → split
    min_frames: int = 145            # ≥ obs lookback (3×5) + horizon (50) + slack
    # gates
    min_arc_m: float = 20.0
    min_net_disp_m: float = 8.0
    min_speed_ms: float = 0.25
    max_stationary_frac: float = 0.35
    max_gps_dev_p95_m: float = 10.0
    max_heading_err_med_rad: float = 0.30
    max_speed_ms: float = 2.5        # implausible for these robots → bad EKF


@dataclass
class SegmentQA:
    """Metrics + verdict for one refined segment."""

    start: int                       # frame offset within episode
    end: int                         # exclusive
    arc_m: float
    net_disp_m: float
    speed_mean_ms: float
    speed_max_ms: float
    stationary_frac: float
    gps_dev_p95_m: float
    heading_err_med_rad: float
    turn_total_deg: float            # Σ|Δheading| — turn content, used for balancing
    accepted: bool
    reject_reasons: list[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.integer):
                d[k] = int(v)
            elif isinstance(v, np.floating):
                d[k] = float(v)
        return d


# ── Refinement ────────────────────────────────────────────────────────────────

def refine_segments(ep: EpisodeData, cfg: QAConfig) -> list[tuple[int, int]]:
    """Split an episode into clean [start, end) frame ranges.

    Split points: video timestamp gaps, internal stops ≥ stop_split_s,
    sustained EKF↔GPS divergence.  Stationary head/tail of each piece is
    trimmed.  Pieces shorter than cfg.min_frames are dropped.
    """
    n = ep.n
    if n < cfg.min_frames:
        return []

    speed = ep.speed()
    moving = speed >= cfg.stop_speed_ms

    # hard split mask between frame i-1 and i
    split = np.zeros(n, dtype=bool)
    split[1:] |= np.diff(ep.video_ts) > cfg.video_gap_s

    dev = np.linalg.norm((ep.utm - ep.utm[0]) - (ep.pos - ep.pos[0]), axis=1)
    split[1:] |= (dev[1:] > cfg.gps_dev_split_m) & (dev[:-1] <= cfg.gps_dev_split_m)

    # internal long stops: find contiguous stationary runs
    stop_run = 0
    stop_frames = int(cfg.stop_split_s * 10)
    for i in range(n):
        if not moving[i]:
            stop_run += 1
            if stop_run == stop_frames:
                split[max(0, i - stop_run + 1)] = True
        else:
            stop_run = 0

    # assemble pieces between split points
    cut_points = [0] + list(np.where(split)[0]) + [n]
    pieces: list[tuple[int, int]] = []
    for a, b in zip(cut_points[:-1], cut_points[1:]):
        if b - a < cfg.min_frames:
            continue
        # trim stationary head/tail
        idx = np.where(moving[a:b])[0]
        if len(idx) == 0:
            continue
        s, t = a + int(idx[0]), a + int(idx[-1]) + 1
        if t - s >= cfg.min_frames:
            pieces.append((s, t))
    return pieces


# ── Gates ─────────────────────────────────────────────────────────────────────

def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def segment_qa(ep: EpisodeData, start: int, end: int, cfg: QAConfig) -> SegmentQA:
    """Compute QA metrics for ep[start:end] and apply acceptance gates."""
    pos = ep.pos[start:end]
    heading = ep.heading[start:end]
    utm = ep.utm[start:end]

    v = np.diff(pos, axis=0)
    sp = np.linalg.norm(v, axis=1) * 10.0
    arc = float(sp.sum() / 10.0)
    net = float(np.linalg.norm(pos[-1] - pos[0]))

    dev = np.linalg.norm((utm - utm[0]) - (pos - pos[0]), axis=1)

    moving = sp > max(0.3, 2 * cfg.stop_speed_ms)
    if moving.sum() >= 10:
        motion_ang = np.arctan2(v[moving, 1], v[moving, 0])
        herr = float(np.median(np.abs(_wrap(motion_ang - heading[1:][moving]))))
    else:
        herr = float("inf")

    turn_total = float(np.degrees(np.abs(np.diff(np.unwrap(heading))).sum()))

    m = SegmentQA(
        start=start, end=end,
        arc_m=arc, net_disp_m=net,
        speed_mean_ms=float(sp.mean()), speed_max_ms=float(sp.max()),
        stationary_frac=float((sp < cfg.stop_speed_ms).mean()),
        gps_dev_p95_m=float(np.percentile(dev, 95)),
        heading_err_med_rad=herr,
        turn_total_deg=turn_total,
        accepted=True, reject_reasons=[],
    )

    gates = [
        (m.arc_m < cfg.min_arc_m,                       "arc_too_short"),
        (m.net_disp_m < cfg.min_net_disp_m,             "net_disp_too_small"),
        (m.speed_mean_ms < cfg.min_speed_ms,            "too_slow"),
        (m.speed_max_ms > cfg.max_speed_ms,             "implausible_speed"),
        (m.stationary_frac > cfg.max_stationary_frac,   "too_stationary"),
        (m.gps_dev_p95_m > cfg.max_gps_dev_p95_m,       "gps_ekf_divergence"),
        (m.heading_err_med_rad > cfg.max_heading_err_med_rad, "heading_motion_mismatch"),
    ]
    for bad, reason in gates:
        if bad:
            m.reject_reasons.append(reason)
    m.accepted = not m.reject_reasons
    return m


# ── Route QA ──────────────────────────────────────────────────────────────────

@dataclass
class RouteQA:
    """Quality metrics of an OSM-snapped route against the EKF trajectory."""

    snap_mean_m: float          # mean point→edge distance (trajectory→network)
    route_len_m: float
    len_ratio: float            # route_len / trajectory arc — detects detours
    max_gap_m: float            # largest route segment (informational — long
                                # straight OSM edges are legitimate)
    offroute_gaps: int          # long segments (>gap threshold) whose midpoint
                                # is far from the demo = jump across unmapped space
    monotonic_frac: float       # fraction of samples whose route projection advances
    dir_agree_med_deg: float    # median angle traj tangent ↔ route tangent at projection
    spike_count: int            # direction reversals >150° between consecutive route segs
    n_anchors: int              # via-anchors used by route_by_graph
    dev_p90_m: float            # p90 demo↔route deviation from route_by_graph
    accepted: bool
    reject_reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def route_qa(
    route_xy: np.ndarray,
    traj_xy: np.ndarray,
    snap_mean_m: float,
    max_snap_mean_m: float = 8.0,
    len_ratio_range: tuple[float, float] = (0.6, 1.8),
    max_gap_m: float = 30.0,
    min_monotonic_frac: float = 0.8,
    max_dir_agree_med_deg: float = 45.0,
    n_anchors: int = 0,
    dev_p90_m: float = 0.0,
) -> RouteQA:
    """Gate a snapped route. All inputs in a common local-metric frame.

    ``dir_agree_med_deg`` is the decisive gate against *tangled* routes:
    per-point edge projection can flip between competing OSM edges and
    produce a zigzag that still passes distance/length/monotonicity checks,
    but its local tangents then disagree wildly with the trajectory's.
    """
    seg = np.diff(route_xy, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    route_len = float(seg_len.sum())
    arc = float(np.linalg.norm(np.diff(traj_xy, axis=0), axis=1).sum())
    ratio = route_len / max(arc, 1e-6)
    gap = float(seg_len.max()) if len(seg_len) else float("inf")

    # monotonic progress + tangent agreement, sampled along the trajectory
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    stride = max(1, len(traj_xy) // 100)
    look = max(stride, 4)        # tangent window ≥ ±4 samples — a single
                                 # subsampled step can be <0.5 m on slow demos
    pts = traj_xy[::stride]
    proj, agree = [], []
    for i, p in enumerate(pts):
        d2 = np.sum((route_xy - p) ** 2, axis=1)
        j = int(np.argmin(d2))
        proj.append(cum[j])
        j_seg = min(j, len(seg) - 1)
        gi = min(i * stride, len(traj_xy) - 1)
        t_traj = (traj_xy[min(gi + look, len(traj_xy) - 1)]
                  - traj_xy[max(gi - look, 0)])
        t_route = seg[j_seg]
        n1, n2 = np.linalg.norm(t_traj), np.linalg.norm(t_route)
        if n1 > 0.5 and n2 > 1e-6:
            cosang = np.clip(t_traj @ t_route / (n1 * n2), -1.0, 1.0)
            agree.append(math.degrees(math.acos(cosang)))
    proj = np.array(proj)
    mono = float((np.diff(proj) >= -2.0).mean()) if len(proj) > 1 else 0.0
    dir_agree = float(np.median(agree)) if agree else 180.0

    # spikes: consecutive route segments that reverse direction (>150°)
    spikes = 0
    for j in range(len(seg) - 1):
        n1, n2 = seg_len[j], seg_len[j + 1]
        if n1 > 0.5 and n2 > 0.5:
            cosang = np.clip(seg[j] @ seg[j + 1] / (n1 * n2), -1.0, 1.0)
            if math.degrees(math.acos(cosang)) > 150.0:
                spikes += 1

    # long route segments are fine along straight roads (sparse OSM vertices);
    # they are jumps across unmapped space only when far from the demo
    offroute_gaps = 0
    for j in np.where(seg_len > max_gap_m)[0]:
        mid = (route_xy[j] + route_xy[j + 1]) / 2.0
        d_mid = float(np.sqrt(np.min(np.sum((traj_xy - mid) ** 2, axis=1))))
        if d_mid > 15.0:
            offroute_gaps += 1

    qa = RouteQA(
        snap_mean_m=float(snap_mean_m), route_len_m=route_len,
        len_ratio=float(ratio), max_gap_m=gap, offroute_gaps=offroute_gaps,
        monotonic_frac=mono,
        dir_agree_med_deg=dir_agree, spike_count=spikes,
        n_anchors=int(n_anchors), dev_p90_m=float(dev_p90_m),
        accepted=True, reject_reasons=[],
    )
    gates = [
        (qa.snap_mean_m > max_snap_mean_m,        "snap_distance"),
        (not len_ratio_range[0] <= qa.len_ratio <= len_ratio_range[1], "route_length_ratio"),
        (qa.offroute_gaps > 0,                    "route_gap"),
        (qa.monotonic_frac < min_monotonic_frac,  "route_not_monotonic"),
        (qa.dir_agree_med_deg > max_dir_agree_med_deg, "route_direction_disagrees"),
        (qa.spike_count > 0,                      "route_spike"),
    ]
    for bad, reason in gates:
        if bad:
            qa.reject_reasons.append(reason)
    qa.accepted = not qa.reject_reasons
    return qa
