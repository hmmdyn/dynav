"""GPS episode segmentation for FrodoBots dataset building.

FrodoBots "episodes" contain noisy, irregular GPS logs: the robot may stop
for long periods, loop back to a previous position, or jump discontinuously.
This module splits a raw episode into clean, monotonic sub-segments suitable
for OSRM map matching and model training.

Filtering pipeline (applied in order):
  1. GPS jump detection  — split wherever consecutive points are too far apart.
  2. Stationary removal  — drop contiguous blocks where speed < threshold.
  3. Loop detection      — end a segment when the robot returns near its start.
  4. Length filter       — discard segments shorter than min_length_m.
"""

from __future__ import annotations

import math


# ── Geodesy helper ─────────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2.0 * R * math.asin(math.sqrt(a))


def _arc_length_m(latlons: list[tuple[float, float]]) -> float:
    """Total arc length of a GPS path in metres."""
    total = 0.0
    for i in range(1, len(latlons)):
        total += _haversine_m(*latlons[i - 1], *latlons[i])
    return total


# ── Segmentation ───────────────────────────────────────────────────────────────

def segment_gps_episode(
    latlons: list[tuple[float, float]],
    timestamps: list[float],
    jump_threshold_m: float = 5.0,
    stationary_speed_mps: float = 0.3,
    stationary_window_s: float = 5.0,
    loop_radius_m: float = 15.0,
    min_length_m: float = 10.0,
) -> list[list[tuple[float, float]]]:
    """Split a noisy GPS episode into clean, monotonic segments.

    Args:
        latlons:              Full GPS trajectory [(lat, lon), ...].
        timestamps:           Unix epoch timestamps (seconds) for each point.
                              Must be the same length as *latlons*.
        jump_threshold_m:     Max allowed distance between consecutive GPS
                              points (metres). Exceeding this triggers a split.
                              At 1 Hz, 5 m corresponds to 18 km/h — well above
                              walking speed, so it reliably catches teleports.
        stationary_speed_mps: Speed threshold below which the robot is
                              considered stationary (m/s).
        stationary_window_s:  Minimum contiguous duration (seconds) of
                              sub-threshold speed required to trigger removal.
                              Short pauses (< window) are preserved.
        loop_radius_m:        If the robot returns within this radius of the
                              *current segment's start point*, the segment is
                              closed and a new one begins.
        min_length_m:         Segments shorter than this arc length are
                              discarded entirely.

    Returns:
        List of clean segments, each as [(lat, lon), ...].
        The goal for each segment is its last GPS point.
    """
    if len(latlons) < 2:
        return []
    if len(latlons) != len(timestamps):
        raise ValueError("latlons and timestamps must have the same length.")

    # ── Step 1: split on GPS jumps ─────────────────────────────────────────────
    raw_segments: list[list[tuple[tuple[float, float], float]]] = []  # [(latlon, ts), ...]
    current: list[tuple[tuple[float, float], float]] = [(latlons[0], timestamps[0])]

    for i in range(1, len(latlons)):
        dist = _haversine_m(*latlons[i - 1], *latlons[i])
        if dist > jump_threshold_m:
            if len(current) >= 2:
                raw_segments.append(current)
            current = [(latlons[i], timestamps[i])]
        else:
            current.append((latlons[i], timestamps[i]))
    if len(current) >= 2:
        raw_segments.append(current)

    # ── Step 2: remove stationary blocks ──────────────────────────────────────
    def _remove_stationary(
        seg: list[tuple[tuple[float, float], float]]
    ) -> list[list[tuple[tuple[float, float], float]]]:
        """Remove contiguous stationary blocks, returning sub-segments."""
        result: list[list[tuple[tuple[float, float], float]]] = []
        active: list[tuple[tuple[float, float], float]] = []
        stat_buf: list[tuple[tuple[float, float], float]] = []

        def flush_stat_buf():
            nonlocal stat_buf
            if not stat_buf:
                return
            buf_dur = stat_buf[-1][1] - stat_buf[0][1]
            if buf_dur < stationary_window_s:
                # Short pause — keep as part of active segment
                active.extend(stat_buf)
            else:
                # Long stop — end current segment, discard pause
                if len(active) >= 2:
                    result.append(active[:])
                active.clear()
            stat_buf = []

        for i, (ll, ts) in enumerate(seg):
            if i == 0:
                active.append((ll, ts))
                continue

            prev_ll, prev_ts = seg[i - 1]
            dt = ts - prev_ts
            dist = _haversine_m(*prev_ll, *ll)
            speed = dist / dt if dt > 0 else 0.0

            if speed < stationary_speed_mps:
                if not stat_buf:
                    # Start of potential stationary block
                    stat_buf.append(seg[i - 1] if not active else active[-1])
                stat_buf.append((ll, ts))
            else:
                flush_stat_buf()
                active.append((ll, ts))

        flush_stat_buf()
        if len(active) >= 2:
            result.append(active)
        return result

    after_stat: list[list[tuple[tuple[float, float], float]]] = []
    for seg in raw_segments:
        after_stat.extend(_remove_stationary(seg))

    # ── Step 3: loop detection ─────────────────────────────────────────────────
    def _split_on_loops(
        seg: list[tuple[tuple[float, float], float]]
    ) -> list[list[tuple[tuple[float, float], float]]]:
        result: list[list[tuple[tuple[float, float], float]]] = []
        current: list[tuple[tuple[float, float], float]] = [seg[0]]
        start_ll = seg[0][0]
        has_left = False   # must leave the radius before re-entry counts as a loop

        for item in seg[1:]:
            ll, ts = item
            dist_to_start = _haversine_m(*ll, *start_ll)
            if dist_to_start > loop_radius_m:
                has_left = True
                current.append(item)
            elif has_left and len(current) > 5:
                # Robot returned to start — close segment
                current.append(item)
                result.append(current)
                current = [item]
                start_ll = ll
                has_left = False
            else:
                current.append(item)

        if len(current) >= 2:
            result.append(current)
        return result

    after_loops: list[list[tuple[tuple[float, float], float]]] = []
    for seg in after_stat:
        after_loops.extend(_split_on_loops(seg))

    # ── Step 4: length filter + extract latlons ────────────────────────────────
    results: list[list[tuple[float, float]]] = []
    for seg in after_loops:
        lls = [item[0] for item in seg]
        if _arc_length_m(lls) >= min_length_m:
            results.append(lls)

    return results
