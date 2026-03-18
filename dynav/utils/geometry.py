"""Geometric utilities for robot navigation coordinate transforms.

All functions assume ROS-style body frame: x = forward, y = left.
Global frame is ENU (East-North-Up): x = East, y = North.
Headings are measured from East (x-axis) counter-clockwise in radians.
"""

import math
from typing import Optional

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────

_LAT_M_PER_DEG: float = 111_320.0  # meters per degree of latitude (approx.)


# ── Coordinate helpers ─────────────────────────────────────────────────────────

def _gps_to_local_meters(
    gps_points: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Convert GPS (lat/lon) positions to local ENU meters.

    Uses a flat-earth approximation — accurate to within ~0.1% for distances
    up to several kilometers near the reference point.

    Args:
        gps_points: Array of shape (N, 2) with [latitude, longitude] columns.
        reference: Reference point of shape (2,) as [latitude, longitude].

    Returns:
        Local ENU positions of shape (N, 2) with [East, North] columns (meters).
    """
    lat_ref, lon_ref = reference
    lon_m_per_deg = _LAT_M_PER_DEG * math.cos(math.radians(lat_ref))

    delta_lat = gps_points[:, 0] - lat_ref   # degrees
    delta_lon = gps_points[:, 1] - lon_ref   # degrees

    east  = delta_lon * lon_m_per_deg         # (N,)
    north = delta_lat * _LAT_M_PER_DEG        # (N,)

    return np.stack([east, north], axis=-1)   # (N, 2) ENU meters


# ── Public functions ───────────────────────────────────────────────────────────

def body_frame_transform(
    global_positions: np.ndarray,
    current_position: np.ndarray,
    current_heading: float,
) -> np.ndarray:
    """Transform global ENU positions into the robot body frame.

    The body frame has x pointing forward (in the direction of heading) and
    y pointing left (ROS REP-103 convention).

    Transformation:
        relative  = global_positions - current_position
        body      = R(heading) @ relative
        where R = [[cos(h),  sin(h)],
                   [-sin(h), cos(h)]]

    Args:
        global_positions: Positions in global ENU meters of shape (N, 2)
            with [East, North] columns. Can also be a single position (2,).
        current_position: Robot's current position in global ENU meters,
            shape (2,).
        current_heading: Robot heading in radians measured from East (x-axis)
            counter-clockwise (standard ENU yaw).

    Returns:
        Positions in robot body frame of shape (N, 2) with [forward, left]
        columns, or (2,) if a single position was given.

    Example:
        >>> # Robot at origin facing north (heading = π/2)
        >>> body_frame_transform(np.array([[0.0, 5.0]]), np.zeros(2), math.pi/2)
        array([[5., 0.]])   # 5 m ahead in body frame
    """
    squeeze = global_positions.ndim == 1
    positions = np.atleast_2d(global_positions)          # (N, 2)

    relative = positions - current_position               # (N, 2) ENU delta

    cos_h = math.cos(current_heading)
    sin_h = math.sin(current_heading)

    # Rotation matrix: global ENU → body frame
    # R = [[cos(h),  sin(h)],
    #      [-sin(h), cos(h)]]
    x_body = relative[:, 0] * cos_h + relative[:, 1] * sin_h   # (N,) forward
    y_body = -relative[:, 0] * sin_h + relative[:, 1] * cos_h  # (N,) left

    body = np.stack([x_body, y_body], axis=-1)            # (N, 2)

    return body.squeeze(0) if squeeze else body


def compute_route_direction(
    current_gps: np.ndarray,
    route_points: np.ndarray,
    current_heading: float,
    lookahead_distance: float = 30.0,
) -> float:
    """Compute the route direction in the robot body frame.

    Finds the point on the OSM route that is ``lookahead_distance`` meters
    ahead of the robot (measured along the route polyline) and returns the
    bearing to that point in the body frame.

    Algorithm:
        1. Convert all GPS coordinates to local ENU meters.
        2. Find the route segment closest to the robot's current position.
        3. Walk forward along the route from that segment until
           ``lookahead_distance`` meters of arc-length are accumulated.
        4. Return ``atan2(y_body, x_body)`` of the lookahead target.

    Args:
        current_gps: Current robot GPS position as [latitude, longitude].
        route_points: OSM route waypoints of shape (M, 2) as
            [latitude, longitude] columns.
        current_heading: Robot heading in radians (ENU yaw, from East CCW).
        lookahead_distance: Arc-length to look ahead along the route, meters.
            Defaults to 30 m.

    Returns:
        Direction to the lookahead point in body frame radians. Positive is
        left (counter-clockwise from forward), negative is right.

    Raises:
        ValueError: If ``route_points`` has fewer than 2 rows.
    """
    if len(route_points) < 2:
        raise ValueError("route_points must have at least 2 waypoints.")

    # Convert to local meters (ENU) with current position as reference
    ref = current_gps
    route_m = _gps_to_local_meters(route_points, ref)     # (M, 2) ENU meters
    cur_m   = np.zeros(2)                                  # origin by definition

    # ── Step 1: find the route segment closest to the robot ────────────────────
    # Project robot onto each segment and find the closest projection point.
    seg_starts = route_m[:-1]                              # (M-1, 2)
    seg_ends   = route_m[1:]                               # (M-1, 2)
    seg_vecs   = seg_ends - seg_starts                     # (M-1, 2)
    seg_lens   = np.linalg.norm(seg_vecs, axis=-1)        # (M-1,)
    seg_lens   = np.where(seg_lens < 1e-9, 1e-9, seg_lens)

    to_cur       = cur_m - seg_starts                      # (M-1, 2)
    # Scalar projection of cur onto each segment, clamped to [0, seg_len]
    t = np.clip(
        (to_cur * seg_vecs).sum(axis=-1) / seg_lens,
        0.0,
        seg_lens,
    )                                                      # (M-1,)
    projections  = seg_starts + (t / seg_lens)[:, None] * seg_vecs  # (M-1, 2)
    dists_to_cur = np.linalg.norm(projections - cur_m, axis=-1)     # (M-1,)

    closest_seg_idx = int(np.argmin(dists_to_cur))
    closest_proj    = projections[closest_seg_idx]         # (2,)

    # ── Step 2: walk along the route until lookahead_distance is reached ───────
    remaining = lookahead_distance
    start_pt  = closest_proj

    target_pt = route_m[-1]  # fallback: last point on route
    for seg_idx in range(closest_seg_idx, len(seg_starts)):
        end_pt   = seg_ends[seg_idx]                       # (2,)
        seg_dist = float(np.linalg.norm(end_pt - start_pt))

        if seg_dist >= remaining:
            # Lookahead target lies on this segment
            direction = (end_pt - start_pt) / max(seg_dist, 1e-9)
            target_pt = start_pt + remaining * direction
            break

        remaining -= seg_dist
        start_pt   = end_pt
    # If loop exhausted, target_pt remains the last route point.

    # ── Step 3: convert target to body frame and compute bearing ───────────────
    target_body = body_frame_transform(
        target_pt[None, :],        # (1, 2)
        cur_m,
        current_heading,
    )                              # (1, 2)

    x_fwd, y_left = float(target_body[0, 0]), float(target_body[0, 1])
    return math.atan2(y_left, x_fwd)  # radians, body frame
