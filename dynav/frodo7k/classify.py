"""Per-sample maneuver / environment / difficulty labelling.

Labels drive composition control (sampling.py) and are stored in each
sample's meta.json so dataset composition is auditable after the build.

  maneuver     : net heading change over the waypoint horizon (5 s)
                 straight | slight_left/right | turn_left/right | uturn
  intersection : an OSM graph node of degree ≥ 3 lies within
                 ``intersection_radius_m`` of the route ahead
  env_density  : urban | mid | sparse — OSM way density around the segment
  difficulty   : [0, 1] — weighted mix of turn magnitude, intersection
                 proximity and speed; used for difficulty-aware selection
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .reader import EpisodeData

MANEUVERS = ["straight", "slight_left", "slight_right",
             "turn_left", "turn_right", "uturn"]
SCENES = ["city", "park", "straight_road", "other"]


def scene_label(
    seg_lats: "np.ndarray",
    seg_lons: "np.ndarray",
    green_polys: list[list[tuple[float, float]]],
    building_count: int,
    bbox_area_km2: float,
    way_density: float,
    arc_m: float,
    net_disp_m: float,
    turn_total_deg: float,
    park_inside_frac: float = 0.5,
    city_building_per_km2: float = 800.0,
    urban_way_density: float = 400.0,
    straight_min_ratio: float = 0.9,
    straight_max_turn_deg_per_m: float = 3.0,
) -> str:
    """Segment-level scene class: park > city > straight_road > other.

    park          — ≥ park_inside_frac of trajectory points lie inside OSM
                    green-area polygons (park/garden/forest/grass…).
    city          — building density or road density above urban thresholds.
    straight_road — geometric: nearly straight segment with little heading
                    change (long stretch of open road).
    """
    from dynav.map.osm_snap import point_in_polygon

    if green_polys:
        pts = list(zip(seg_lats[::5], seg_lons[::5]))
        inside = sum(
            any(point_in_polygon(la, lo, poly) for poly in green_polys)
            for la, lo in pts
        )
        if pts and inside / len(pts) >= park_inside_frac:
            return "park"

    bld_density = building_count / max(bbox_area_km2, 1e-6)
    if bld_density >= city_building_per_km2 or way_density >= urban_way_density:
        return "city"

    if (net_disp_m / max(arc_m, 1e-6) >= straight_min_ratio
            and turn_total_deg / max(arc_m, 1e-6) <= straight_max_turn_deg_per_m):
        return "straight_road"
    return "other"


def maneuver_label(dh_deg: float) -> str:
    """Maneuver class from signed net heading change (deg, + = left/CCW)."""
    a = abs(dh_deg)
    if a < 15.0:
        return "straight"
    if a > 120.0:
        return "uturn"
    side = "left" if dh_deg > 0 else "right"
    return f"{'slight' if a < 45.0 else 'turn'}_{side}"


def _intersection_nodes(edges: list) -> np.ndarray:
    """(K, 2) lat/lon of OSM nodes with degree ≥ 3 (treated as intersections)."""
    deg: dict = defaultdict(int)
    for p1, p2 in edges:
        deg[p1] += 1
        deg[p2] += 1
    pts = [p for p, d in deg.items() if d >= 3]
    return np.array(pts) if pts else np.empty((0, 2))


def classify_candidates(
    ep: EpisodeData,
    seg_start: int,
    indices: list[int],
    horizon_frames: int,
    edges: list,
    way_count: int,
    bbox_area_km2: float,
    scene: str = "other",
    intersection_radius_m: float = 15.0,
    lookahead_m: float = 20.0,
    urban_way_density: float = 400.0,
    sparse_way_density: float = 60.0,
) -> list[dict]:
    """Label each candidate frame index (offsets within the episode).

    Args:
        ep:             Episode data.
        seg_start:      Segment start (for bookkeeping only).
        indices:        Candidate frame offsets within the episode.
        horizon_frames: Waypoint horizon (frames @10 fps) used for maneuver.
        edges:          OSM edges [( (lat,lon), (lat,lon) ), …] of the segment area.
        way_count:      Number of OSM ways in the segment bbox.
        bbox_area_km2:  Area of that bbox.
        intersection_radius_m: Node-to-route distance that counts as "at" an
                        intersection.
        lookahead_m:    How far ahead along the trajectory to look for
                        intersections.
        urban_way_density / sparse_way_density: ways/km² thresholds.

    Returns:
        One dict per index: {idx, maneuver, dh_deg, near_intersection,
        env_density, scene, difficulty, speed_ms}.  *scene* is the
        segment-level class from :func:`scene_label`, attached verbatim.
    """
    density = way_count / max(bbox_area_km2, 1e-6)
    env = ("urban" if density >= urban_way_density
           else "sparse" if density < sparse_way_density
           else "mid")

    nodes = _intersection_nodes(edges)
    node_xy = None
    if len(nodes):
        # local metric frame anchored at episode origin
        mlat = 6_371_000.0 * math.pi / 180.0
        mlon = mlat * math.cos(math.radians(ep.lat0))
        node_xy = np.stack([
            (nodes[:, 1] - ep.lon0) * mlon,        # x = East
            (nodes[:, 0] - ep.lat0) * mlat,        # y = North
        ], axis=1) + ep.pos[0]

    speed = ep.speed()
    out = []
    h_un = np.unwrap(ep.heading)
    for idx in indices:
        wi = min(idx + horizon_frames, ep.n - 1)
        dh = math.degrees(h_un[wi] - h_un[idx])
        man = maneuver_label(dh)

        near_int = False
        if node_xy is not None:
            # points along the trajectory ahead of idx, up to lookahead_m
            j, dist = idx, 0.0
            while j < ep.n - 1 and dist < lookahead_m:
                dist += float(np.linalg.norm(ep.pos[j + 1] - ep.pos[j]))
                j += 1
            ahead = ep.pos[idx:j + 1:5]
            if len(ahead):
                d2 = ((node_xy[None, :, :] - ahead[:, None, :]) ** 2).sum(-1)
                near_int = bool((d2.min() ** 0.5) <= intersection_radius_m)

        turn_score = min(abs(dh) / 90.0, 1.0)
        difficulty = float(np.clip(
            0.55 * turn_score + 0.30 * near_int + 0.15 * min(speed[idx] / 1.0, 1.0),
            0.0, 1.0))

        out.append(dict(
            idx=int(idx), maneuver=man, dh_deg=round(float(dh), 1),
            near_intersection=near_int, env_density=env, scene=scene,
            difficulty=round(difficulty, 3), speed_ms=round(float(speed[idx]), 2),
        ))
    return out
