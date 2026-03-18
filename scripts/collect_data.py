#!/usr/bin/env python3
"""dynav offline data collection pipeline.

Generates training samples from OSRM routes — no real robot required.
Camera observations are blank placeholders; replace with rosbag extraction
once real data is available.

Usage
-----
# Single route:
python scripts/collect_data.py \\
    --start 37.557 126.936 \\
    --end   37.562 126.941 \\
    --output data/ --split train

# Batch from JSON file:
python scripts/collect_data.py \\
    --routes-json routes.json \\
    --output data/ --split train

routes.json format:
    [{"start": [lat, lon], "end": [lat, lon]}, ...]
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

# ── osmnav imports ─────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "osmnav/src/osmnav"))
sys.path.insert(0, str(_REPO / "osmnav/src/nomad_map_context"))

from osmnav.osrm_client import OSRMClient
from nomad_map_context.tile_cache import TileCache
from nomad_map_context.tile_renderer import TileRenderer
from nomad_map_context.route_overlay import (
    draw_goal_marker,
    draw_position_marker,
    draw_route,
)
from nomad_map_context.image_processor import crop_ego_view, rotate_north_up

# ── Constants ──────────────────────────────────────────────────────────────────
HORIZON: int = 5                    # number of GT waypoints  (H)
MAX_WP_DIST_M: float = 2.5         # normalisation cap (metres)
MAP_RENDER_SIZE: int = 512          # TileRenderer internal canvas
MAP_OUTPUT_SIZE: int = 224          # final map.png side length
OBS_SIZE: int = 224                 # camera image side length
TILE_ZOOM: int = 17
M_PER_DEG_LAT: float = 111_320.0


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * 6_371_000.0 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing from point 1 → 2, degrees [0, 360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _latlon_to_body(
    robot_lat: float, robot_lon: float, heading_deg: float,
    target_lat: float, target_lon: float,
) -> Tuple[float, float]:
    """Convert (target_lat, target_lon) to robot body frame (x=forward, y=left).

    heading_deg: compass bearing the robot faces (0=North, 90=East, CW).
    Returns (x, y) in metres — NOT yet normalised.
    """
    cos_lat = math.cos(math.radians(robot_lat))
    m_per_deg_lon = M_PER_DEG_LAT * cos_lat

    dE = (target_lon - robot_lon) * m_per_deg_lon   # East metres
    dN = (target_lat - robot_lat) * M_PER_DEG_LAT   # North metres

    h = math.radians(heading_deg)
    x =  dE * math.sin(h) + dN * math.cos(h)   # forward  (positive = ahead)
    y = -dE * math.cos(h) + dN * math.sin(h)   # left     (positive = left)
    return x, y


def _route_direction_rad(
    robot_lat: float, robot_lon: float, heading_deg: float,
    ahead_wps: List[Tuple[float, float]],
) -> float:
    """Route direction angle in body frame (radians).

    Looks at the next 3 waypoints ahead and returns the mean bearing
    relative to the robot's body frame.  A value near 0 means the route
    goes straight ahead; ±π means the route goes directly behind.
    """
    if not ahead_wps:
        return 0.0
    look = ahead_wps[:3]
    angles = []
    for wlat, wlon in look:
        b_world = _bearing(robot_lat, robot_lon, wlat, wlon)
        b_body = math.radians(b_world - heading_deg)
        # wrap to [-π, π]
        b_body = (b_body + math.pi) % (2 * math.pi) - math.pi
        angles.append(b_body)
    return sum(angles) / len(angles)


# ── Route sampling ─────────────────────────────────────────────────────────────

def _parse_osrm_waypoints(osrm_data: dict) -> List[Tuple[float, float]]:
    """Extract (lat, lon) list from OSRM GeoJSON response."""
    coords = osrm_data["routes"][0]["geometry"]["coordinates"]  # [lon, lat]
    return [(lat, lon) for lon, lat in coords]


def _sample_along_route(
    route_wps: List[Tuple[float, float]],
    step_m: float,
    heading_jitter_deg: float = 8.0,
) -> List[dict]:
    """Walk the route and yield sample dicts every *step_m* metres.

    Each dict contains:
        lat, lon        – robot position
        heading_deg     – robot compass heading (base + small jitter)
        gt_waypoints    – List[Tuple[float,float]] next HORIZON lat/lon positions
        route_wps_ahead – remaining waypoints after this position (for map overlay)
    """
    samples = []
    # Interpolate route_wps to finer resolution first
    fine: List[Tuple[float, float]] = [route_wps[0]]
    for i in range(1, len(route_wps)):
        d = _haversine(*route_wps[i - 1], *route_wps[i])
        if d < 0.1:
            continue
        steps = max(1, int(d / 0.5))
        lat0, lon0 = route_wps[i - 1]
        lat1, lon1 = route_wps[i]
        for k in range(1, steps + 1):
            t = k / steps
            fine.append((lat0 + t * (lat1 - lat0), lon0 + t * (lon1 - lon0)))

    accumulated = 0.0
    for i in range(len(fine) - 1):
        d = _haversine(*fine[i], *fine[i + 1])
        accumulated += d
        if accumulated < step_m:
            continue
        accumulated = 0.0

        lat, lon = fine[i]
        # heading = bearing to next fine point
        base_heading = _bearing(lat, lon, *fine[i + 1])
        jitter = random.gauss(0.0, heading_jitter_deg)
        heading_deg = (base_heading + jitter) % 360.0

        # GT waypoints: next HORIZON points spaced ~MAX_WP_DIST_M apart
        gt_latlon: List[Tuple[float, float]] = []
        j = i + 1
        last_lat, last_lon = lat, lon
        while len(gt_latlon) < HORIZON and j < len(fine):
            if _haversine(last_lat, last_lon, *fine[j]) >= MAX_WP_DIST_M / HORIZON:
                gt_latlon.append(fine[j])
                last_lat, last_lon = fine[j]
            j += 1
        # Pad with last point if route is too short
        while len(gt_latlon) < HORIZON:
            gt_latlon.append(gt_latlon[-1] if gt_latlon else (lat, lon))

        samples.append({
            "lat": lat,
            "lon": lon,
            "heading_deg": heading_deg,
            "gt_latlon": gt_latlon,
            "route_wps_ahead": fine[i:],
        })

    return samples


# ── Map image generation ───────────────────────────────────────────────────────

def _render_map_image(
    renderer: TileRenderer,
    lat: float,
    lon: float,
    heading_deg: float,
    route_wps: List[Tuple[float, float]],
    goal_lat: float,
    goal_lon: float,
    output_size: int = MAP_OUTPUT_SIZE,
) -> Image.Image:
    """Generate a heading-up map image centred on the robot.

    Pipeline:
        TileRenderer (north-up, 512×512)
        → draw_route / draw_position / draw_goal overlays
        → rotate_north_up (heading-up)
        → crop_ego_view   (robot at bottom-third, output_size×output_size)
    """
    result = renderer.render(lat, lon)
    if result is None:
        # Tile download failed — return grey placeholder
        return Image.new("RGB", (output_size, output_size), (180, 180, 180))

    raw_map, geo_tf = result

    # Overlays (all in-place on raw_map)
    draw_route(raw_map, route_wps, geo_tf)
    robot_px = TileRenderer.latlon_to_pixel(lat, lon, geo_tf)
    draw_position_marker(raw_map, robot_px[0], robot_px[1], heading_deg)
    goal_px = TileRenderer.latlon_to_pixel(goal_lat, goal_lon, geo_tf)
    draw_goal_marker(raw_map, goal_px[0], goal_px[1])

    # Rotate heading-up (robot forward = top)
    rotated, new_robot_px = rotate_north_up(raw_map, heading_deg, robot_px)

    # Crop: robot at vertical_bias=2/3 from top → ~1/3 from bottom
    cropped, _ = crop_ego_view(
        rotated, new_robot_px,
        output_size=output_size,
        vertical_bias=2.0 / 3.0,
    )

    if cropped.size != (output_size, output_size):
        cropped = cropped.resize((output_size, output_size), Image.LANCZOS)

    return cropped


# ── Placeholder camera observations ───────────────────────────────────────────

def _make_placeholder_obs(n: int = 4, size: int = OBS_SIZE) -> List[Image.Image]:
    """Return n grey placeholder camera images.

    Replace this function with real rosbag frame extraction when available.
    Grey (128, 128, 128) makes the model train on map information only,
    which is fine for initial pipeline validation.
    """
    return [Image.new("RGB", (size, size), (128, 128, 128)) for _ in range(n)]


# ── Sample writer ──────────────────────────────────────────────────────────────

def _write_sample(
    sample_dir: Path,
    obs_images: List[Image.Image],
    map_image: Image.Image,
    gt_latlon: List[Tuple[float, float]],
    robot_lat: float,
    robot_lon: float,
    heading_deg: float,
) -> None:
    """Save one sample to disk in dynavDataset format."""
    sample_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(obs_images):
        img.save(sample_dir / f"obs_{i}.png")

    map_image.save(sample_dir / "map.png")

    # Convert GT lat/lon to body-frame (x, y) and normalise
    gt_waypoints: List[List[float]] = []
    for wlat, wlon in gt_latlon:
        x, y = _latlon_to_body(robot_lat, robot_lon, heading_deg, wlat, wlon)
        x_norm = max(-1.0, min(1.0, x / MAX_WP_DIST_M))
        y_norm = max(-1.0, min(1.0, y / MAX_WP_DIST_M))
        gt_waypoints.append([x_norm, y_norm])

    # Route direction: angle of the route in body frame (radians)
    route_direction = _route_direction_rad(
        robot_lat, robot_lon, heading_deg, gt_latlon
    )

    meta = {
        "gt_waypoints": gt_waypoints,           # [[x, y], ...] × H, in [-1, 1]
        "route_direction": route_direction,      # float, radians, body frame
        # Debug fields (not used by dynavDataset)
        "robot_lat": robot_lat,
        "robot_lon": robot_lon,
        "heading_deg": heading_deg,
        "gt_latlon": [[float(la), float(lo)] for la, lo in gt_latlon],
    }
    with open(sample_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ── Pipeline orchestration ─────────────────────────────────────────────────────

def collect_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    output_dir: Path,
    split: str,
    step_m: float,
    start_idx: int,
    renderer: TileRenderer,
    osrm: OSRMClient,
) -> int:
    """Collect samples for one route.  Returns number of samples written."""
    print(f"  OSRM route ({start_lat:.4f},{start_lon:.4f}) → ({end_lat:.4f},{end_lon:.4f})")
    data = osrm.get_route(start_lat, start_lon, end_lat, end_lon, alternatives=False)
    if data is None or data.get("code") != "Ok":
        print(f"  [WARN] OSRM failed: {data}")
        return 0

    route_wps = _parse_osrm_waypoints(data)
    total_dist = data["routes"][0]["distance"]
    print(f"  Route: {len(route_wps)} waypoints, {total_dist:.0f} m")

    samples = _sample_along_route(route_wps, step_m=step_m)
    print(f"  Sampling: {len(samples)} positions at step={step_m} m")

    split_dir = output_dir / split
    count = 0
    for sample in samples:
        idx = start_idx + count
        sample_dir = split_dir / f"sample_{idx:06d}"

        map_img = _render_map_image(
            renderer,
            sample["lat"], sample["lon"], sample["heading_deg"],
            sample["route_wps_ahead"],
            end_lat, end_lon,
        )
        obs_imgs = _make_placeholder_obs()

        _write_sample(
            sample_dir, obs_imgs, map_img,
            sample["gt_latlon"],
            sample["lat"], sample["lon"], sample["heading_deg"],
        )
        count += 1
        if count % 10 == 0:
            print(f"    {count}/{len(samples)} samples written...")

    return count


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="dynav data collection pipeline")
    p.add_argument("--output", default="data", help="Root dataset directory")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--step", type=float, default=1.0,
                   help="Sampling step along route in metres (default: 1.0)")
    p.add_argument("--tile-cache", default="/tmp/nomad_tile_cache",
                   help="Directory for OSM tile cache")
    p.add_argument("--zoom", type=int, default=TILE_ZOOM)
    p.add_argument("--seed", type=int, default=42)

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--routes-json", metavar="FILE",
                       help="JSON file with list of {start:[lat,lon], end:[lat,lon]}")
    group.add_argument("--start", nargs=2, type=float, metavar=("LAT", "LON"))

    p.add_argument("--end", nargs=2, type=float, metavar=("LAT", "LON"),
                   help="Required when --start is used")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    if args.start and not args.end:
        print("ERROR: --end is required when --start is specified", file=sys.stderr)
        sys.exit(1)

    # Load route list
    if args.routes_json:
        with open(args.routes_json) as f:
            route_list = json.load(f)
        routes = [
            (r["start"][0], r["start"][1], r["end"][0], r["end"][1])
            for r in route_list
        ]
    else:
        routes = [(args.start[0], args.start[1], args.end[0], args.end[1])]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / args.split).mkdir(parents=True, exist_ok=True)

    # Initialise shared objects
    cache = TileCache(
        cache_dir=args.tile_cache,
        tile_url_template="https://a.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
    )
    renderer = TileRenderer(cache, zoom=args.zoom, output_size=MAP_RENDER_SIZE)
    osrm = OSRMClient()

    # Count existing samples to continue numbering
    existing = sorted((output_dir / args.split).glob("sample_*"))
    next_idx = len(existing)

    total = 0
    for i, (s_lat, s_lon, e_lat, e_lon) in enumerate(routes):
        print(f"\n[Route {i + 1}/{len(routes)}]")
        n = collect_route(
            s_lat, s_lon, e_lat, e_lon,
            output_dir, args.split, args.step,
            start_idx=next_idx + total,
            renderer=renderer,
            osrm=osrm,
        )
        total += n

    print(f"\nDone. {total} samples written to {output_dir / args.split}/")


if __name__ == "__main__":
    main()
