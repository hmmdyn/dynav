"""Build dynav training dataset from FrodoBots ride data.

Replaces build_frodobots_dataset.py (v1) and build_frodobots_dataset_v2.py.
Uses dynav.map for all map rendering — identical output to extract_rosbag.py.

Data layout expected on disk (configure via --frodo-root):
    <frodo_root>/
        output_rides_0/   output_rides_2/   output_rides_23/
            <ride_id>/
                recordings/          ← .ts video files
                camera_timestamps.csv
                gps_data.csv
        valid_segments_rides0.json
        valid_segments_rides2.json
        valid_segments_v2.json

Route generation (per segment):
    Raw GPS → segment_gps_episode() → OSRM /match → OSM-snapped route
    Segments where mean GPS↔route deviation > 10 m are skipped.

Usage::

    python scripts/build_frodobots_dataset.py \\
        --frodo-root ~/data/frodobots \\
        --output data/ \\
        [--map-config configs/map.yaml]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from dynav.map import MapRenderer, OSRMRouter, is_route_valid  # noqa: E402
from dynav.map.routing import OSRMMatchError  # noqa: E402
from dynav.utils.geometry import compute_route_direction  # noqa: E402

try:
    from tqdm import tqdm as _tqdm
    def _progress(it, **kw):
        return _tqdm(it, **kw)
except ImportError:
    def _progress(it, **kw):
        return it

# ---------------------------------------------------------------------------
# Constants (FrodoBots-specific)
# ---------------------------------------------------------------------------
FPS            = 20
OBS_STRIDE     = 10   # frames between obs_0/1/2  (0.5 s @ 20 Hz)
SAMPLE_STRIDE  = 20   # frames between consecutive samples (1 s)
N_WAYPOINTS    = 5
WAYPOINT_FRAMES = [FPS * (i + 1) for i in range(N_WAYPOINTS)]  # [20,40,60,80,100]
MAX_WP_DIST_M  = 2.5
IMAGE_SIZE     = 224

OBS_PAD = OBS_STRIDE * 2        # minimum lead frames for obs history
WP_PAD  = WAYPOINT_FRAMES[-1]   # minimum trailing frames for waypoints

EARTH_R = 6_371_000.0


# ---------------------------------------------------------------------------
# Geodesy helpers
# ---------------------------------------------------------------------------
def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return 2.0 * EARTH_R * math.asin(math.sqrt(a))


def _gps_to_body(
    cur_lat: float, cur_lon: float,
    fut_lat: float, fut_lon: float,
    hdg_deg: float,
) -> tuple[float, float]:
    """Convert future GPS position to robot body frame (x=forward, y=left)."""
    d_lat = math.radians(fut_lat - cur_lat)
    d_lon = math.radians(fut_lon - cur_lon)
    north_m = d_lat * EARTH_R
    east_m  = d_lon * EARTH_R * math.cos(math.radians(cur_lat))
    h = math.radians(hdg_deg)
    dx = north_m * math.cos(h) + east_m * math.sin(h)
    dy = -north_m * math.sin(h) + east_m * math.cos(h)
    return dx, dy


def _compute_heading(lats: list[float], lons: list[float], idx: int) -> float:
    """Estimate compass heading at *idx* from surrounding GPS points."""
    w = 40  # ±2 s at 20 Hz — wide enough to span multiple 1-Hz GPS samples
    i0 = max(0, idx - w)
    i1 = min(len(lats) - 1, idx + w)
    if i0 == i1:
        return 0.0
    d_lat = lats[i1] - lats[i0]
    d_lon = lons[i1] - lons[i0]
    north_m = d_lat * 111_000.0
    east_m  = d_lon * math.cos(math.radians(lats[idx])) * 111_000.0
    if north_m == 0.0 and east_m == 0.0:
        return 0.0
    return math.degrees(math.atan2(east_m, north_m)) % 360.0


# ---------------------------------------------------------------------------
# Timestamp / frame index helpers
# ---------------------------------------------------------------------------
def _parse_ts_ms(stem: str) -> int:
    """Extract Unix-ms timestamp from a .ts filename stem."""
    ts_str = stem.split("_")[-1][:17]
    from datetime import datetime, timezone
    s = ts_str
    dt = datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]),
                  int(s[8:10]), int(s[10:12]), int(s[12:14]),
                  int(s[14:17]) * 1000, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _build_ts_index(recordings_dir: Path) -> list[tuple[float, Path]]:
    """Return [(start_ts_s, ts_path), ...] sorted by start time."""
    files = sorted(recordings_dir.glob("*uid_s_1000*uid_e_video*.ts"))
    return [(_parse_ts_ms(p.stem) / 1000.0, p) for p in files]


def _load_camera_csv(cam_csv: Path) -> list[tuple[int, float]]:
    with open(cam_csv) as f:
        return [(int(r["frame_id"]), float(r["timestamp"]))
                for r in csv.DictReader(f)]


def _frame_to_ts_local(
    ts_index: list[tuple[float, Path]],
    frame_ts_s: float,
) -> Optional[tuple[Path, int]]:
    """Map a camera timestamp to (ts_file, local_frame_idx)."""
    lo, hi = 0, len(ts_index) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ts_index[mid][0] <= frame_ts_s:
            lo = mid
        else:
            hi = mid - 1  # pylint: disable=unused-variable
    ts_start, ts_path = ts_index[lo]
    local_idx = round((frame_ts_s - ts_start) * FPS)
    if local_idx < 0:
        return None
    return ts_path, local_idx


def _extract_frames(
    cam_list: list[tuple[int, float]],
    ts_index: list[tuple[float, Path]],
    needed_fids: set[int],
    frames_dir: Path,
) -> None:
    """Extract only the needed frames from .ts files via ffmpeg."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    missing = {fid for fid in needed_fids
               if not (frames_dir / f"{fid:06d}.jpg").exists()}
    if not missing:
        return

    fid_to_ts = {fid: ts for fid, ts in cam_list if fid in missing}
    ts_groups: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for fid, ts_s in fid_to_ts.items():
        result = _frame_to_ts_local(ts_index, ts_s)
        if result is None:
            continue
        ts_path, local_idx = result
        ts_groups[ts_path].append((fid, local_idx))

    for ts_path, frame_list in ts_groups.items():
        if not ts_path.exists():
            continue
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(ts_path),
                 "-q:v", "3", "-vf", f"scale={IMAGE_SIZE}:{IMAGE_SIZE}",
                 f"{tmpdir}/%06d.jpg"],
                capture_output=True,
            )
            for global_fid, local_idx in frame_list:
                src = Path(tmpdir) / f"{local_idx + 1:06d}.jpg"
                dst = frames_dir / f"{global_fid:06d}.jpg"
                if src.exists() and not dst.exists():
                    shutil.move(str(src), str(dst))


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------
class FrodoBotsBuilder:
    """Build a dynav dataset from FrodoBots source data.

    Args:
        frodo_root: Directory containing ``output_rides_*`` and JSON files.
        output_dir: Root output directory for the dataset.
        renderer:   Shared MapRenderer instance.
        router:     Shared OSRMRouter instance.
        threshold_m: Max GPS↔route deviation to accept a segment.
    """

    SOURCES = [
        ("output_rides_0",  "valid_segments_rides0.json"),
        ("output_rides_2",  "valid_segments_rides2.json"),
        ("output_rides_23", "valid_segments_v2.json"),
    ]

    def __init__(
        self,
        frodo_root: Path,
        output_dir: Path,
        renderer: MapRenderer,
        router: OSRMRouter,
        threshold_m: float = 10.0,
    ) -> None:
        self.frodo_root  = frodo_root
        self.output_dir  = output_dir
        self.renderer    = renderer
        self.router      = router
        self.threshold_m = threshold_m
        self.frames_root = output_dir / "frames"

    # ------------------------------------------------------------------

    def build(self) -> dict[str, int]:
        """Run the full build pipeline.

        Returns:
            {"train": n_train, "val": n_val} sample counts.
        """
        (self.output_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "val").mkdir(parents=True, exist_ok=True)

        counts: dict[str, int] = {"train": 0, "val": 0}

        for rides_name, seg_json_name in self.SOURCES:
            rides_dir = self.frodo_root / rides_name
            seg_path  = self.frodo_root / seg_json_name
            if not rides_dir.exists() or not seg_path.exists():
                print(f"[skip] {rides_name}: directory or segment JSON not found")
                continue

            with open(seg_path) as f:
                segments: list[dict] = json.load(f)

            print(f"[build] {rides_name}: {len(segments)} segments")
            for seg in _progress(segments, desc=rides_name, leave=False):
                n = self._build_segment(seg, rides_dir)
                counts["train"] += n.get("train", 0)
                counts["val"]   += n.get("val", 0)

        print(f"[build] done — train={counts['train']}, val={counts['val']}")
        return counts

    # ------------------------------------------------------------------

    def _build_segment(
        self,
        seg: dict,
        rides_dir: Path,
    ) -> dict[str, int]:
        """Process one segment: OSRM match, frame extraction, sample writing."""
        ride_id = seg["ride_id"]
        fids    = seg["frame_ids"]
        lats    = seg["frame_lat"]
        lons    = seg["frame_lon"]
        n       = len(fids)

        if n < OBS_PAD + WP_PAD + 1:
            return {}

        split = "val" if int(ride_id) % 10 == 0 else "train"
        split_dir = self.output_dir / split

        # Unique GPS points (1-Hz, many frames share the same GPS)
        unique_latlons = list(dict.fromkeys(zip(lats, lons)))
        if len(unique_latlons) < 2:
            return {}

        # --- OSRM map matching (once per segment) ---
        try:
            matched_route, avg_dev = self.router.match(
                unique_latlons, radius_m=50.0
            )
        except OSRMMatchError as exc:
            print(f"[skip] ride={ride_id}: OSRM match failed: {exc}")
            return {}

        if not is_route_valid(matched_route, unique_latlons, self.threshold_m):
            print(f"[skip] ride={ride_id}: avg_dev={avg_dev:.1f}m > {self.threshold_m}m")
            return {}

        goal_lat, goal_lon = unique_latlons[-1]

        # --- Determine which frames we need ---
        i_start = OBS_PAD
        i_end   = n - 1 - WP_PAD
        if i_end <= i_start:
            return {}

        sample_indices = list(range(i_start, i_end + 1, SAMPLE_STRIDE))
        needed_fids: set[int] = set()
        for local_i in sample_indices:
            needed_fids.add(fids[local_i])
            needed_fids.add(fids[local_i - OBS_STRIDE])
            needed_fids.add(fids[local_i - 2 * OBS_STRIDE])
            needed_fids.add(fids[min(n - 1, local_i + OBS_STRIDE)])  # rear ≈ forward shifted

        # --- Extract frames from .ts files ---
        ride_dir    = rides_dir / ride_id
        frames_dir  = self.frames_root / f"ride_{ride_id}"
        recordings  = ride_dir / "recordings"
        cam_csv     = ride_dir / "camera_timestamps.csv"

        if not recordings.exists() or not cam_csv.exists():
            return {}

        ts_index = _build_ts_index(recordings)
        cam_list = _load_camera_csv(cam_csv)
        _extract_frames(cam_list, ts_index, needed_fids, frames_dir)

        # --- Write samples ---
        written: dict[str, int] = {"train": 0, "val": 0}

        for local_i in sample_indices:
            fid_cur   = fids[local_i]
            fid_past1 = fids[local_i - OBS_STRIDE]
            fid_past2 = fids[local_i - 2 * OBS_STRIDE]
            fid_rear  = fids[min(n - 1, local_i + OBS_STRIDE)]

            # Observation frames must exist on disk
            def _img(fid: int) -> Optional[Image.Image]:
                p = frames_dir / f"{fid:06d}.jpg"
                if not p.exists():
                    return None
                return Image.open(p).convert("RGB")

            obs0 = _img(fid_cur)
            obs1 = _img(fid_past1)
            obs2 = _img(fid_past2)
            obs3 = _img(fid_rear)
            if any(o is None for o in (obs0, obs1, obs2, obs3)):
                continue

            cur_lat = lats[local_i]
            cur_lon = lons[local_i]
            hdg_deg = _compute_heading(lats, lons, local_i)

            # Map image
            map_img = self.renderer.render(
                lat=cur_lat,
                lon=cur_lon,
                heading_deg=hdg_deg,
                route_latlons=matched_route,
                goal_lat=goal_lat,
                goal_lon=goal_lon,
            )
            if map_img is None:
                continue

            # GT waypoints
            waypoints = []
            for wp_frames in WAYPOINT_FRAMES:
                wi = min(local_i + wp_frames, n - 1)
                dx, dy = _gps_to_body(cur_lat, cur_lon, lats[wi], lons[wi], hdg_deg)
                x_norm = max(-1.0, min(1.0, dx / MAX_WP_DIST_M))
                y_norm = max(-1.0, min(1.0, dy / MAX_WP_DIST_M))
                waypoints.append([round(x_norm, 5), round(y_norm, 5)])

            # route_direction: radians, body frame
            enu_heading = math.radians(90.0 - hdg_deg)
            route_dir_rad = compute_route_direction(
                np.array([cur_lat, cur_lon]),
                np.array(matched_route),
                enu_heading,
                lookahead_distance=10.0,
            )

            # Write sample directory
            sample_name = f"sample_{ride_id}_{fid_cur:07d}"
            sample_dir  = split_dir / sample_name
            sample_dir.mkdir(exist_ok=True)

            obs0.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).save(sample_dir / "obs_0.png")
            obs1.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).save(sample_dir / "obs_1.png")
            obs2.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).save(sample_dir / "obs_2.png")
            obs3.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).save(sample_dir / "obs_3.png")
            map_img.save(sample_dir / "map.png")

            meta = {
                "ride_id":         ride_id,
                "frame_id":        fid_cur,
                "lat":             cur_lat,
                "lon":             cur_lon,
                "heading_deg":     hdg_deg,
                "goal_lat":        goal_lat,
                "goal_lon":        goal_lon,
                "gt_waypoints":    waypoints,
                "route_direction": float(route_dir_rad),
            }
            with open(sample_dir / "meta.json", "w") as fh:
                json.dump(meta, fh, indent=2)

            written[split] += 1

        return written


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def _load_map_config(map_config_path: Path):
    from omegaconf import OmegaConf
    return OmegaConf.load(map_config_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dynav dataset from FrodoBots ride data."
    )
    parser.add_argument("--frodo-root", type=Path, required=True, metavar="DIR",
                        help="FrodoBots data root (contains output_rides_* and JSON files).")
    parser.add_argument("--output", type=Path, required=True, metavar="DIR",
                        help="Dataset output root directory.")
    parser.add_argument("--map-config", type=Path,
                        default=_REPO / "configs" / "map.yaml", metavar="YAML",
                        help="Path to map.yaml (default: configs/map.yaml).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    map_cfg_path = args.map_config.resolve()
    if not map_cfg_path.exists():
        print(f"[error] map config not found: {map_cfg_path}", file=sys.stderr)
        sys.exit(1)

    # Wrap map config in a namespace expected by MapRenderer.from_config
    from omegaconf import OmegaConf
    raw = OmegaConf.load(map_cfg_path)
    cfg = OmegaConf.create({"map": raw})

    renderer = MapRenderer.from_config(cfg)
    router   = OSRMRouter.from_config(cfg)

    builder = FrodoBotsBuilder(
        frodo_root=args.frodo_root.resolve(),
        output_dir=args.output.resolve(),
        renderer=renderer,
        router=router,
        threshold_m=float(raw.get("osrm_valid_threshold_m", 10.0)),
    )
    counts = builder.build()
    sys.exit(0 if sum(counts.values()) > 0 else 1)


if __name__ == "__main__":
    main()
