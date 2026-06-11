"""OSM pedestrian network download and graph-aware trajectory snapping.

Offline alternative to OSRM for FrodoBots dataset building.
Requires internet access on first call per area; results are disk-cached.

Typical usage::

    edges, all_pts = fetch_ped_network(lat, lon, radius_m=400,
                                       cache_dir=Path(".../osm_net_cache"))
    _, mean_dist = snap_trajectory(lats, lons, edges, all_pts)
    if mean_dist < 10.0:
        route = snap_trajectory_graph(lats, lons, edges, all_pts)
        # pass route to MapRenderer.render(route_latlons=route, ...)
"""

from __future__ import annotations

import heapq
import json
import math
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import KDTree

R_EARTH: float = 6_371_000.0
_PED_TAGS: str = "footway|path|pedestrian|living_street|steps|track"

# FrodoBots robots ride sidewalks of ordinary streets; the pedestrian-only
# tag set misses those areas entirely (good segments were rejected and snapped
# routes detoured).  frodo7k pipeline uses this wider set.
_DRIVE_PED_TAGS: str = (
    "footway|path|pedestrian|living_street|steps|track"
    "|residential|service|unclassified|cycleway|tertiary|secondary"
)


# ── Geodesy ────────────────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in metres between two lat/lon points."""
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return 2.0 * R_EARTH * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing (0 = North, clockwise) from point 1 to point 2."""
    d_lon = math.radians(lon2 - lon1)
    x = math.sin(d_lon) * math.cos(math.radians(lat2))
    y = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
         - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.cos(d_lon))
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _pt_to_segment(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> tuple[float, float, float]:
    """Distance and projection from point (px,py) to segment (a→b).

    Returns:
        (distance, proj_x, proj_y) — all in the same units as the inputs.
    """
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 < 1e-12:
        return math.hypot(px - ax, py - ay), ax, ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len2))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y), proj_x, proj_y


# ── Local metric coordinate helper ────────────────────────────────────────────

def _make_metric(ref_lat: float, ref_lon: float):
    """Return (to_m, from_m) callables for a flat-earth local coordinate system."""
    mpp_lat = R_EARTH * math.pi / 180.0
    mpp_lon = mpp_lat * math.cos(math.radians(ref_lat))

    def to_m(lat: float, lon: float) -> tuple[float, float]:
        return lat * mpp_lat, lon * mpp_lon

    def from_m(y: float, x: float) -> tuple[float, float]:
        return y / mpp_lat, x / mpp_lon

    return to_m, from_m


# ── Overpass API ───────────────────────────────────────────────────────────────

def _overpass_query(bbox_str: str, tags: str = _PED_TAGS) -> dict:
    query = (
        f'[out:json][timeout:30];\n'
        f'(way[highway~"^({tags})$"]({bbox_str}););\n'
        f'out geom;'
    )
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(
        "https://overpass-api.de/api/interpreter",
        data=data,
        headers={"User-Agent": "dynav-research/1.0"},
    )
    with urllib.request.urlopen(req, timeout=40) as resp:
        return json.loads(resp.read())


def fetch_ped_network(
    center_lat: float,
    center_lon: float,
    radius_m: float = 400.0,
    cache_dir: Optional[Path] = None,
) -> tuple[list, list]:
    """Download (or load from cache) OSM pedestrian network around a point.

    Downloads footway / path / pedestrian / living_street / steps / track
    ways from the Overpass API. Results are saved as JSON files keyed by
    (lat, lon, radius) so repeated calls for the same area are instant.

    Args:
        center_lat: Area centre latitude in degrees.
        center_lon: Area centre longitude in degrees.
        radius_m:   Bounding-box half-width in metres.
        cache_dir:  Directory for JSON cache files. Falls back to
                    ``/tmp/dynav_osm_cache`` if None.

    Returns:
        edges:    List of ``[(lat1, lon1), (lat2, lon2)]`` OSM way segments.
        all_pts:  Flat ``[(lat, lon), ...]`` list of all segment endpoints
                  (used to build a KDTree for fast nearest-point queries).
    """
    if cache_dir is None:
        cache_dir = Path("/tmp/dynav_osm_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    deg_r = radius_m / R_EARTH * (180.0 / math.pi)
    deg_r_lon = deg_r / math.cos(math.radians(center_lat))
    bbox = (
        f"{center_lat - deg_r:.5f},{center_lon - deg_r_lon:.5f},"
        f"{center_lat + deg_r:.5f},{center_lon + deg_r_lon:.5f}"
    )

    cache_key = f"{center_lat:.4f}_{center_lon:.4f}_{radius_m:.0f}.json"
    cache_path = cache_dir / cache_key
    if cache_path.exists():
        result = json.loads(cache_path.read_text())
    else:
        result = _overpass_query(bbox)
        cache_path.write_text(json.dumps(result))
        time.sleep(0.5)  # polite rate limit

    edges: list = []
    for way in result.get("elements", []):
        geom = way.get("geometry", [])
        for i in range(len(geom) - 1):
            p1 = (geom[i]["lat"],     geom[i]["lon"])
            p2 = (geom[i + 1]["lat"], geom[i + 1]["lon"])
            edges.append((p1, p2))

    all_pts = [pt for edge in edges for pt in edge]
    return edges, all_pts


def fetch_network_bbox(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    margin_m: float = 150.0,
    tags: str = _DRIVE_PED_TAGS,
    cache_dir: Optional[Path] = None,
) -> tuple[list, list, int]:
    """Download (or load cached) OSM network covering a full bounding box.

    Unlike :func:`fetch_ped_network` (centre + fixed radius), this covers the
    *whole* segment trajectory, so long segments do not lose network coverage
    at their ends.  The cache key is grid-aligned (0.005° ≈ 550 m) so nearby
    segments share downloads.

    Returns:
        edges:     ``[((lat1, lon1), (lat2, lon2)), …]`` way segments.
        all_pts:   Flat endpoint list (KDTree input).
        way_count: Number of OSM ways in the response (environment-density
                   signal for classification).
    """
    if cache_dir is None:
        cache_dir = Path("/tmp/dynav_osm_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    margin_deg = margin_m / R_EARTH * (180.0 / math.pi)
    grid = 0.005
    lo_lat = math.floor((min_lat - margin_deg) / grid) * grid
    lo_lon = math.floor((min_lon - margin_deg) / grid) * grid
    hi_lat = math.ceil((max_lat + margin_deg) / grid) * grid
    hi_lon = math.ceil((max_lon + margin_deg) / grid) * grid
    bbox = f"{lo_lat:.5f},{lo_lon:.5f},{hi_lat:.5f},{hi_lon:.5f}"

    tag_key = hashlib_md5_8(tags)
    cache_path = cache_dir / f"bbox_{bbox.replace(',', '_')}_{tag_key}.json"
    if cache_path.exists():
        result = json.loads(cache_path.read_text())
    else:
        result = _overpass_query(bbox, tags=tags)
        cache_path.write_text(json.dumps(result))
        time.sleep(0.5)  # polite rate limit

    edges: list = []
    way_count = 0
    for way in result.get("elements", []):
        way_count += 1
        geom = way.get("geometry", [])
        for i in range(len(geom) - 1):
            p1 = (geom[i]["lat"], geom[i]["lon"])
            p2 = (geom[i + 1]["lat"], geom[i + 1]["lon"])
            edges.append((p1, p2))

    all_pts = [pt for edge in edges for pt in edge]
    return edges, all_pts, way_count


def hashlib_md5_8(s: str) -> str:
    """First 8 hex chars of md5 — short stable cache-key component."""
    import hashlib

    return hashlib.md5(s.encode()).hexdigest()[:8]


_GREEN_TAGS_QUERY = (
    'way[leisure~"^(park|garden|playground|nature_reserve)$"]({bbox});'
    'way[landuse~"^(forest|grass|recreation_ground|meadow|village_green)$"]({bbox});'
    'way[natural~"^(wood|grassland|scrub)$"]({bbox});'
)


def fetch_scene_bbox(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    margin_m: float = 150.0,
    cache_dir: Optional[Path] = None,
) -> tuple[list[list[tuple[float, float]]], int]:
    """Scene-classification signals for a bbox (grid-cached like the network).

    Two Overpass queries per new grid cell:
      1. green-area polygons (park / garden / forest / grass …, ``out geom``)
      2. building count only (``out count`` — cheap, no geometry transferred)

    Returns:
        green_polys:    List of polygons, each ``[(lat, lon), …]``.
        building_count: Number of building ways in the bbox.
    """
    if cache_dir is None:
        cache_dir = Path("/tmp/dynav_osm_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    margin_deg = margin_m / R_EARTH * (180.0 / math.pi)
    grid = 0.005
    lo_lat = math.floor((min_lat - margin_deg) / grid) * grid
    lo_lon = math.floor((min_lon - margin_deg) / grid) * grid
    hi_lat = math.ceil((max_lat + margin_deg) / grid) * grid
    hi_lon = math.ceil((max_lon + margin_deg) / grid) * grid
    bbox = f"{lo_lat:.5f},{lo_lon:.5f},{hi_lat:.5f},{hi_lon:.5f}"

    cache_path = cache_dir / f"scene_{bbox.replace(',', '_')}.json"
    if cache_path.exists():
        result = json.loads(cache_path.read_text())
    else:
        q_green = ('[out:json][timeout:30];('
                   + _GREEN_TAGS_QUERY.format(bbox=bbox)
                   + ');out geom;')
        q_bld = f'[out:json][timeout:30];way[building]({bbox});out count;'
        green = _overpass_post(q_green)
        bld = _overpass_post(q_bld)
        n_bld = 0
        for el in bld.get("elements", []):
            if el.get("type") == "count":
                n_bld = int(el.get("tags", {}).get("ways", 0))
        result = {
            "green": [
                [[g["lat"], g["lon"]] for g in way.get("geometry", [])]
                for way in green.get("elements", [])
                if len(way.get("geometry", [])) >= 3
            ],
            "building_count": n_bld,
        }
        cache_path.write_text(json.dumps(result))
        time.sleep(0.5)  # polite rate limit

    polys = [[(p[0], p[1]) for p in poly] for poly in result["green"]]
    return polys, int(result["building_count"])


def _overpass_post(query: str) -> dict:
    """POST a raw Overpass QL query."""
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(
        "https://overpass-api.de/api/interpreter",
        data=data,
        headers={"User-Agent": "dynav-research/1.0"},
    )
    with urllib.request.urlopen(req, timeout=40) as resp:
        return json.loads(resp.read())


def point_in_polygon(lat: float, lon: float,
                     poly: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test; *poly* is [(lat, lon), …]."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        yi, xi = poly[i]
        yj, xj = poly[j]
        if (yi > lat) != (yj > lat):
            x_cross = xi + (lat - yi) * (xj - xi) / (yj - yi)
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


# ── Point-to-edge snapping (quality filter) ───────────────────────────────────

def snap_trajectory(
    frame_lats: list[float],
    frame_lons: list[float],
    edges: list,
    all_pts: list,
) -> tuple[list[tuple[float, float]], float]:
    """Snap GPS points to the nearest OSM pedestrian edge.

    Used as a segment quality filter: if the mean snap distance exceeds a
    threshold (e.g. 10 m), the segment is not on any pedestrian network and
    should be discarded.

    Args:
        frame_lats: GPS latitudes (one per frame).
        frame_lons: GPS longitudes (one per frame).
        edges:      OSM edges from :func:`fetch_ped_network`.
        all_pts:    Flat endpoint list from :func:`fetch_ped_network`.

    Returns:
        snapped_latlons: Projected GPS points on OSM edges.
        mean_dist_m:     Mean snap distance (metres). ``float('inf')`` if
                         the network is empty.
    """
    if not edges or not all_pts:
        return list(zip(frame_lats, frame_lons)), float("inf")

    to_m, from_m = _make_metric(frame_lats[0], frame_lons[0])

    pts_m = np.array([to_m(la, lo) for la, lo in all_pts])
    tree  = KDTree(pts_m)

    edge_pts_m = [(to_m(*p1), to_m(*p2)) for p1, p2 in edges]
    pt2edges: dict = {}
    for ei, (a_m, b_m) in enumerate(edge_pts_m):
        for pt in (a_m, b_m):
            key = (round(pt[0], 3), round(pt[1], 3))
            pt2edges.setdefault(key, []).append(ei)

    snapped: list[tuple[float, float]] = []
    dists: list[float] = []

    for lat, lon in zip(frame_lats, frame_lons):
        pm = to_m(lat, lon)
        k = min(8, len(pts_m))
        _, idxs = tree.query(pm, k=k)
        if not hasattr(idxs, "__iter__"):
            idxs = [idxs]

        best_d, best_proj = float("inf"), pm
        for idx in idxs:
            pt_key = (round(pts_m[idx][0], 3), round(pts_m[idx][1], 3))
            for ei in pt2edges.get(pt_key, []):
                (ax, ay), (bx, by) = edge_pts_m[ei]
                d, px, py = _pt_to_segment(pm[0], pm[1], ax, ay, bx, by)
                if d < best_d:
                    best_d, best_proj = d, (px, py)

        dists.append(best_d)
        snapped.append(from_m(best_proj[0], best_proj[1]))

    return snapped, float(np.mean(dists))


# ── Graph routing (training route = inference route style) ────────────────────

def _project_directional(
    pm: tuple[float, float],
    tangent_m: tuple[float, float],
    edge_pts_m: list,
    pts_m: np.ndarray,
    tree,
    pt2edges: dict,
    max_edge_angle_deg: float = 50.0,
) -> tuple[int, tuple[float, float], float]:
    """Project a point onto the best *direction-compatible* edge.

    Among k-NN candidate edges, edges whose direction disagrees with the
    local demo tangent by more than ``max_edge_angle_deg`` are skipped —
    this prevents snapping onto perpendicular side streets at intersections.
    Falls back to the plain nearest edge if no candidate passes the angle
    test (e.g. the robot is crossing a road).

    Args:
        pm:        Point in local metric coords.
        tangent_m: Local demo tangent at that point (metric, any norm).
        edge_pts_m / pts_m / tree / pt2edges: precomputed edge structures.

    Returns:
        (edge_index, projected_point_m, distance_m)
    """
    k = min(12, len(pts_m))
    _, idxs = tree.query(pm, k=k)
    if not hasattr(idxs, "__iter__"):
        idxs = [idxs]

    t_norm = math.hypot(*tangent_m)
    cos_lim = math.cos(math.radians(max_edge_angle_deg))

    best = (float("inf"), 0, pm)        # (dist, edge_idx, proj)
    best_any = (float("inf"), 0, pm)    # ignoring the angle test (fallback)
    seen: set = set()
    for idx in idxs:
        pt_key = (round(pts_m[idx][0], 3), round(pts_m[idx][1], 3))
        for ei in pt2edges.get(pt_key, []):
            if ei in seen:
                continue
            seen.add(ei)
            (ax, ay), (bx, by) = edge_pts_m[ei]
            d, px, py = _pt_to_segment(pm[0], pm[1], ax, ay, bx, by)
            if d < best_any[0]:
                best_any = (d, ei, (px, py))
            if t_norm > 1e-6:
                ex, ey = bx - ax, by - ay
                e_norm = math.hypot(ex, ey)
                if e_norm > 1e-9:
                    cosang = abs((tangent_m[0] * ex + tangent_m[1] * ey)
                                 / (t_norm * e_norm))
                    if cosang < cos_lim:
                        continue
            if d < best[0]:
                best = (d, ei, (px, py))

    if best[0] == float("inf"):
        best = best_any
    return best[1], best[2], best[0]


def _remove_spurs(route_m: list, max_spur_m: float = 15.0) -> list:
    """Drop vertices that make the polyline double back on itself.

    Projection points lie mid-edge while Dijkstra enters/exits via edge
    corner nodes, which can create short A→corner→A reversals ("spurs").
    A vertex is removed when the polyline turns by more than 150° there and
    at least one adjacent segment is shorter than ``max_spur_m``.
    """
    cos_rev = math.cos(math.radians(150.0))
    pts = list(route_m)
    changed = True
    while changed and len(pts) > 2:
        changed = False
        out = [pts[0]]
        i = 1
        while i < len(pts) - 1:
            a, b, c = out[-1], pts[i], pts[i + 1]
            v1 = (b[0] - a[0], b[1] - a[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            n1, n2 = math.hypot(*v1), math.hypot(*v2)
            if (n1 > 1e-6 and n2 > 1e-6 and min(n1, n2) < max_spur_m
                    and (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2) < cos_rev):
                changed = True       # reversal at b → drop b
                i += 1
                continue
            out.append(pts[i])
            i += 1
        out.append(pts[-1])
        pts = out
    return pts


def route_by_graph(
    frame_lats: list[float],
    frame_lons: list[float],
    edges: list,
    all_pts: list,
    max_dev_m: float = 10.0,
    max_anchors: int = 12,
    max_edge_angle_deg: float = 50.0,
    stitch_radius_m: float = 2.5,
) -> Optional[tuple[list[tuple[float, float]], int, float]]:
    """Build a training route by *graph wayfinding*, not per-point snapping.

    Mirrors inference (`OSRM /route`: one shortest path to the goal):
    project start and end onto direction-compatible edges, connect them with
    a Dijkstra shortest path along road centerlines.  Where the demonstrated
    trajectory diverges from that path by more than ``max_dev_m``, insert a
    via-anchor at the point of maximum deviation and recurse, so the route
    follows the *demonstrated* road sequence with the minimum number of
    anchors.  Per-point projection zigzag cannot occur by construction.

    Args:
        frame_lats / frame_lons: Demo trajectory (subsampled is fine).
        edges / all_pts:         Network from fetch_network_bbox().
        max_dev_m:               Demo↔route deviation that triggers an anchor.
        max_anchors:             Give up beyond this many via-anchors — the
                                 demo does not follow mapped roads.
        max_edge_angle_deg:      Direction-compatibility limit for projection.

    Returns:
        (route_latlons, n_anchors, dev_p90_m), or None if the demo cannot be
        explained by the road graph within the anchor budget.
    """
    n = len(frame_lats)
    if not edges or not all_pts or n < 2:
        return None

    graph = build_osm_graph(edges)
    to_m, from_m = _make_metric(frame_lats[0], frame_lons[0])
    traj_m = np.array([to_m(la, lo) for la, lo in zip(frame_lats, frame_lons)])

    # Stitch near-coincident nodes: OSM footway and street ways frequently
    # cross without sharing a node, fragmenting the graph and forcing
    # Dijkstra into straight-jump fallbacks (visible as huge route gaps).
    nodes = list(graph.keys())
    if nodes:
        node_m = np.array([to_m(*nd) for nd in nodes])
        ntree = KDTree(node_m)
        for i, j in ntree.query_pairs(stitch_radius_m):
            d = float(np.linalg.norm(node_m[i] - node_m[j]))
            graph[nodes[i]].setdefault(nodes[j], d)
            graph[nodes[j]].setdefault(nodes[i], d)

    pts_m = np.array([to_m(la, lo) for la, lo in all_pts])
    tree = KDTree(pts_m)
    edge_pts_m = [(to_m(*p1), to_m(*p2)) for p1, p2 in edges]
    pt2edges: dict = {}
    for ei, (a_m, b_m) in enumerate(edge_pts_m):
        for pt in (a_m, b_m):
            key = (round(pt[0], 3), round(pt[1], 3))
            pt2edges.setdefault(key, []).append(ei)

    def tangent_at(i: int) -> tuple[float, float]:
        i0, i1 = max(0, i - 2), min(n - 1, i + 2)
        d = traj_m[i1] - traj_m[i0]
        return (float(d[0]), float(d[1]))

    def project(i: int) -> tuple[int, tuple[float, float]]:
        ei, proj, _ = _project_directional(
            (float(traj_m[i][0]), float(traj_m[i][1])), tangent_at(i),
            edge_pts_m, pts_m, tree, pt2edges, max_edge_angle_deg)
        return ei, proj

    def anchor_node(i: int) -> tuple:
        """Graph node (lat/lon tuple) for a via-anchor at demo index *i*.

        Via-anchors snap to the nearer corner *node* of the chosen edge —
        mid-edge anchor points force the path to enter and leave through the
        same corner, creating 180° out-and-back spurs at every join.
        """
        ei, proj = project(i)
        n1, n2 = edges[ei]

        def _dm(p_ll):
            pm_ = to_m(*p_ll)
            return math.hypot(pm_[0] - proj[0], pm_[1] - proj[1])

        return n1 if _dm(n1) < _dm(n2) else n2

    def _dm_to(p_ll, q_m) -> float:
        pm_ = to_m(*p_ll)
        return math.hypot(pm_[0] - q_m[0], pm_[1] - q_m[1])

    def connect(a, b) -> list[tuple[float, float]]:
        """Graph path (metric coords) between endpoint specs.

        Spec: ``("pt", demo_idx)`` (mid-edge projection, used for the segment
        start/goal) or ``("node", latlon)`` (via-anchor graph node).
        """
        if a[0] == "pt" and b[0] == "pt":
            ei_a, pa = project(a[1])
            ei_b, pb = project(b[1])
            if ei_a == ei_b:
                return [pa, pb]
            a1, a2 = edges[ei_a]
            b1, b2 = edges[ei_b]
            exit_node = a1 if _dm_to(a1, pb) < _dm_to(a2, pb) else a2
            entry_node = b1 if _dm_to(b1, pa) < _dm_to(b2, pa) else b2
            path_ll = dijkstra_path(graph, exit_node, entry_node)
            return [pa] + [to_m(*p) for p in path_ll] + [pb]

        if a[0] == "pt":                       # pt → node
            nd_b = b[1]
            ei_a, pa = project(a[1])
            a1, a2 = edges[ei_a]
            nb_m = to_m(*nd_b)
            exit_node = a1 if _dm_to(a1, nb_m) < _dm_to(a2, nb_m) else a2
            path_ll = dijkstra_path(graph, exit_node, nd_b)
            return [pa] + [to_m(*p) for p in path_ll]

        if b[0] == "pt":                       # node → pt
            nd_a = a[1]
            ei_b, pb = project(b[1])
            b1, b2 = edges[ei_b]
            na_m = to_m(*nd_a)
            entry_node = b1 if _dm_to(b1, na_m) < _dm_to(b2, na_m) else b2
            path_ll = dijkstra_path(graph, nd_a, entry_node)
            return [to_m(*p) for p in path_ll] + [pb]

        path_ll = dijkstra_path(graph, a[1], b[1])   # node → node
        return [to_m(*p) for p in path_ll]

    def demo_i(spec) -> int:
        return spec[2] if spec[0] == "node" else spec[1]

    def seg_dev(route_m: list, i_a: int, i_b: int) -> tuple[float, int]:
        """(max deviation, demo index of max) of demo[i_a:i_b] from route_m."""
        r = np.array(route_m)
        worst, worst_i = 0.0, i_a
        for i in range(i_a, i_b + 1):
            p = traj_m[i]
            d_best = float("inf")
            for j in range(len(r) - 1):
                d, _, _ = _pt_to_segment(p[0], p[1], r[j][0], r[j][1],
                                         r[j + 1][0], r[j + 1][1])
                if d < d_best:
                    d_best = d
            if d_best > worst:
                worst, worst_i = d_best, i
        return worst, worst_i

    # recursive bisection: anchor where the demo leaves the shortest path
    anchors_used = 0

    def build(a, b, depth: int) -> Optional[list]:
        nonlocal anchors_used
        route_m = connect(a, b)
        i_a, i_b = demo_i(a), demo_i(b)
        dev, i_mid = seg_dev(route_m, i_a, i_b)
        if dev <= max_dev_m or i_b - i_a < 4 or depth > max_anchors:
            return route_m
        if anchors_used >= max_anchors:
            return None
        if i_mid <= i_a or i_mid >= i_b:
            i_mid = (i_a + i_b) // 2
        anchors_used += 1
        mid = ("node", anchor_node(i_mid), i_mid)
        left = build(a, mid, depth + 1)
        right = build(mid, b, depth + 1)
        if left is None or right is None:
            return None
        return left + right[1:]

    route_m = build(("pt", 0), ("pt", n - 1), 0)
    if route_m is None or len(route_m) < 2:
        return None
    route_m = _remove_spurs(route_m)

    # residual deviation stats over the whole demo
    r = np.array(route_m)
    devs = []
    for i in range(0, n, max(1, n // 60)):
        p = traj_m[i]
        d_best = min(
            _pt_to_segment(p[0], p[1], r[j][0], r[j][1], r[j + 1][0], r[j + 1][1])[0]
            for j in range(len(r) - 1)
        )
        devs.append(d_best)
    dev_p90 = float(np.percentile(devs, 90))
    if dev_p90 > max_dev_m * 1.5:
        return None

    route_ll = [from_m(y_x[0], y_x[1]) for y_x in route_m]
    # dedupe consecutive duplicates
    route_ll = [p for i, p in enumerate(route_ll) if i == 0 or p != route_ll[i - 1]]
    return route_ll, anchors_used, dev_p90


# ── Graph-aware route snapping ─────────────────────────────────────────────────

def build_osm_graph(edges: list) -> dict:
    """Build undirected adjacency graph ``{node: {neighbor: dist_m}}`` from edges."""
    graph: dict = defaultdict(dict)
    for p1, p2 in edges:
        d = haversine_m(p1[0], p1[1], p2[0], p2[1])
        graph[p1][p2] = d
        graph[p2][p1] = d
    return graph


def dijkstra_path(graph: dict, start: tuple, end: tuple) -> list[tuple]:
    """Shortest path from *start* to *end* through OSM graph nodes.

    Args:
        graph: Adjacency dict from :func:`build_osm_graph`.
        start: Source node (lat, lon) tuple.
        end:   Target node (lat, lon) tuple.

    Returns:
        List of (lat, lon) nodes from start to end (inclusive).
        Falls back to ``[start, end]`` if no path exists.
    """
    if start == end:
        return [start]
    dist = {start: 0.0}
    prev: dict = {}
    pq = [(0.0, start)]
    seen: set = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == end:
            path = []
            while u in prev:
                path.append(u)
                u = prev[u]
            path.append(start)
            return path[::-1]
        for v, w in graph.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return [start, end]  # fallback: direct segment


def snap_trajectory_graph(
    frame_lats: list[float],
    frame_lons: list[float],
    edges: list,
    all_pts: list,
) -> list[tuple[float, float]]:
    """Edge-aware graph snap: correct resolution on straight roads, no shortcuts at turns.

    Two cases per consecutive GPS pair:
    - **Same edge**: append direct edge projection — preserves sub-metre
      resolution on long straight paths.
    - **Edge transition**: insert Dijkstra path through the corner nodes
      of both edges — eliminates the "hypotenuse" artifact at 90° turns.

    Args:
        frame_lats: GPS latitudes (one per frame).
        frame_lons: GPS longitudes (one per frame).
        edges:      OSM edges from :func:`fetch_ped_network`.
        all_pts:    Flat endpoint list from :func:`fetch_ped_network`.

    Returns:
        Route as ``[(lat, lon), ...]`` suitable for ``MapRenderer.render()``.
    """
    if not edges or not all_pts:
        return list(zip(frame_lats, frame_lons))

    graph = build_osm_graph(edges)
    to_m, from_m = _make_metric(frame_lats[0], frame_lons[0])

    edge_pts_m = [(to_m(*p1), to_m(*p2)) for p1, p2 in edges]
    pts_m = np.array([to_m(la, lo) for la, lo in all_pts])
    tree  = KDTree(pts_m)

    pt2edges: dict = {}
    for ei, (a_m, b_m) in enumerate(edge_pts_m):
        for pt in (a_m, b_m):
            key = (round(pt[0], 3), round(pt[1], 3))
            pt2edges.setdefault(key, []).append(ei)

    # Project each GPS point onto its best edge
    projs: list[tuple[int, float, float]] = []
    for lat, lon in zip(frame_lats, frame_lons):
        pm = to_m(lat, lon)
        k = min(8, len(pts_m))
        _, idxs = tree.query(pm, k=k)
        if not hasattr(idxs, "__iter__"):
            idxs = [idxs]

        best_d, best_ei, best_proj = float("inf"), 0, pm
        for idx in idxs:
            pt_key = (round(pts_m[idx][0], 3), round(pts_m[idx][1], 3))
            for ei in pt2edges.get(pt_key, []):
                (ax, ay), (bx, by) = edge_pts_m[ei]
                d, px, py = _pt_to_segment(pm[0], pm[1], ax, ay, bx, by)
                if d < best_d:
                    best_d, best_ei, best_proj = d, ei, (px, py)

        projs.append((best_ei, *from_m(best_proj[0], best_proj[1])))

    # Build route
    result: list[tuple[float, float]] = [(projs[0][1], projs[0][2])]
    for i in range(1, len(projs)):
        prev_ei, prev_lat, prev_lon = projs[i - 1]
        curr_ei, curr_lat, curr_lon = projs[i]

        if prev_ei == curr_ei:
            result.append((curr_lat, curr_lon))
        else:
            p1p, p2p = edges[prev_ei]
            p1c, p2c = edges[curr_ei]

            curr_m = to_m(curr_lat, curr_lon)

            def _d_to_curr(p: tuple) -> float:
                a, b = to_m(*p)
                return math.hypot(a - curr_m[0], b - curr_m[1])

            exit_node = p1p if _d_to_curr(p1p) < _d_to_curr(p2p) else p2p

            prev_m = to_m(prev_lat, prev_lon)

            def _d_to_prev(p: tuple) -> float:
                a, b = to_m(*p)
                return math.hypot(a - prev_m[0], b - prev_m[1])

            entry_node = p1c if _d_to_prev(p1c) < _d_to_prev(p2c) else p2c

            path = dijkstra_path(graph, exit_node, entry_node)
            result.extend(path)
            result.append((curr_lat, curr_lon))

    return result
