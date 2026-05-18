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

def _overpass_query(bbox_str: str) -> dict:
    query = (
        f'[out:json][timeout:30];\n'
        f'(way[highway~"^({_PED_TAGS})$"]({bbox_str}););\n'
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
