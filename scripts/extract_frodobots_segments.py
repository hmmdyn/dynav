#!/usr/bin/env python3
"""
extract_frodobots_segments.py — FrodoBots raw ride 데이터에서 세그먼트 추출

입력:  <frodo_root>/output_rides_*/  (GPS, camera timestamps CSV)
출력:  <frodo_root>/valid_segments_{group}.json
       형식: { ride_id: { "segments": [ {seg_idx, frame_ids, frame_lat, frame_lon,
                                         n_frames, net_disp_m, avg_speed_ms, straightness} ] } }

세그먼트 분할 기준:
  - GPS 속도 < STOP_SPEED_MS 가 STOP_WINDOW_S 이상 지속 → 세그먼트 경계
  - GPS 타임스탬프 갭 > GPS_GAP_S → 강제 분할
  - 최소 세그먼트 길이 MIN_SEG_FRAMES 미만 → 제거

이미 처리된 ride는 건너뜀 (incremental 실행 지원).

환경변수로 파라미터 오버라이드 가능 (GUI 연동):
  DYNAV_SEG_STOP_SPEED   (default 0.4 m/s)
  DYNAV_SEG_STOP_WINDOW  (default 3.0 s)
  DYNAV_SEG_GPS_GAP      (default 5.0 s)
  DYNAV_SEG_MIN_FRAMES   (default 50, stride-10 기준)
  DYNAV_SEG_GROUPS       (default "rides0,rides2,rides22,rides23")
  DYNAV_FRODO_ROOT       FrodoBots 데이터 루트 (configs/paths.yaml 오버라이드)

Usage::

    python scripts/extract_frodobots_segments.py
    python scripts/extract_frodobots_segments.py --paths-config configs/paths.yaml
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

OBS_STRIDE = 10  # extract every 10th camera frame

STOP_SPEED_MS  = float(os.environ.get("DYNAV_SEG_STOP_SPEED",  "0.4"))
STOP_WINDOW_S  = float(os.environ.get("DYNAV_SEG_STOP_WINDOW", "3.0"))
GPS_GAP_S      = float(os.environ.get("DYNAV_SEG_GPS_GAP",     "5.0"))
MIN_SEG_FRAMES = int(  os.environ.get("DYNAV_SEG_MIN_FRAMES",  "50"))

ALL_GROUP_DIRS = {
    "rides0":  "output_rides_0",
    "rides2":  "output_rides_2",
    "rides22": "output_rides_22",
    "rides23": "output_rides_23",
}
SEG_GROUPS = [
    g.strip()
    for g in os.environ.get("DYNAV_SEG_GROUPS",
                             "rides0,rides2,rides22,rides23").split(",")
    if g.strip()
]


# ── path loading ──────────────────────────────────────────────────────────────

def _load_frodo_root(paths_config: Path) -> Path:
    try:
        import yaml
        with open(paths_config) as f:
            cfg = yaml.safe_load(f) or {}
        default = cfg.get("frodo_root", "/data/frodobots")
    except Exception:
        default = "/data/frodobots"
    return Path(os.environ.get("DYNAV_FRODO_ROOT", default)).expanduser()


# ── geometry ──────────────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p)
         * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


# ── GPS interpolation ─────────────────────────────────────────────────────────

def _bisect(lst: list, val: float) -> int:
    lo, hi = 0, len(lst) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if lst[mid] <= val:
            lo = mid
        else:
            hi = mid
    return lo


def build_gps_timeline(gps_rows: list) -> tuple:
    """Returns (gps_t, gps_lat, gps_lon). gps_t in seconds (ms → s)."""
    gps_t   = [int(r["timestamp"]) / 1000.0 for r in gps_rows]
    gps_lat = [float(r["latitude"])          for r in gps_rows]
    gps_lon = [float(r["longitude"])         for r in gps_rows]
    return gps_t, gps_lat, gps_lon


def interp_gps(t: float, gps_t: list, gps_lat: list, gps_lon: list) -> tuple:
    """Linearly interpolate GPS lat/lon at time t (seconds)."""
    if t <= gps_t[0]:
        return gps_lat[0], gps_lon[0]
    if t >= gps_t[-1]:
        return gps_lat[-1], gps_lon[-1]
    lo = _bisect(gps_t, t)
    hi = lo + 1
    span = gps_t[hi] - gps_t[lo]
    frac = (t - gps_t[lo]) / span if span > 0 else 0.0
    return (gps_lat[lo] + frac * (gps_lat[hi] - gps_lat[lo]),
            gps_lon[lo] + frac * (gps_lon[hi] - gps_lon[lo]))


def smoothed_speed(t: float, gps_t: list, gps_lat: list, gps_lon: list,
                   window_s: float = 2.0) -> float:
    """Speed estimate at t from GPS points within [t-w, t+w]. More robust than point speed."""
    pts = [(gps_t[i], gps_lat[i], gps_lon[i])
           for i in range(len(gps_t))
           if abs(gps_t[i] - t) <= window_s]
    if len(pts) < 2:
        # Fallback: nearest-neighbour segment speed
        idx = min(range(len(gps_t)), key=lambda i: abs(gps_t[i] - t))
        if idx == 0:
            return 0.0
        d  = haversine(gps_lat[idx-1], gps_lon[idx-1], gps_lat[idx], gps_lon[idx])
        dt = gps_t[idx] - gps_t[idx-1]
        return d / dt if dt > 0 else 0.0
    total_d = sum(haversine(pts[i-1][1], pts[i-1][2], pts[i][1], pts[i][2])
                  for i in range(1, len(pts)))
    total_t = pts[-1][0] - pts[0][0]
    return total_d / total_t if total_t > 0 else 0.0


# ── segmentation ──────────────────────────────────────────────────────────────

def extract_ride_segments(ride_dir: Path, ride_id: str) -> list:
    """Parse one ride directory and return list of segment dicts ([] on invalid data)."""
    gps_file = ride_dir / f"gps_data_{ride_id}.csv"
    cam_file = ride_dir / f"front_camera_timestamps_{ride_id}.csv"

    if not gps_file.exists() or not cam_file.exists():
        return []

    with open(gps_file) as f:
        gps_rows = list(csv.DictReader(f))
    with open(cam_file) as f:
        cam_rows = list(csv.DictReader(f))

    if len(gps_rows) < 2 or len(cam_rows) < OBS_STRIDE * 5:
        return []

    gps_t, gps_lat, gps_lon = build_gps_timeline(gps_rows)

    # stride-10 frames only, sorted by frame_id
    stride_frames = sorted(
        [(int(r["frame_id"]), float(r["timestamp"]))
         for r in cam_rows if int(r["frame_id"]) % OBS_STRIDE == 0],
        key=lambda x: x[0],
    )
    if len(stride_frames) < MIN_SEG_FRAMES:
        return []

    # per-frame GPS interpolation + smoothed speed
    fdata = []  # (frame_id, t, lat, lon, spd)
    for fid, t in stride_frames:
        lat, lon = interp_gps(t, gps_t, gps_lat, gps_lon)
        spd = smoothed_speed(t, gps_t, gps_lat, gps_lon, window_s=2.0)
        fdata.append((fid, t, lat, lon, spd))

    # ── detect segment boundaries ──
    # Boundary types:
    #   1. Sustained stop: speed < STOP_SPEED_MS for >= STOP_WINDOW_S
    #   2. GPS time gap > GPS_GAP_S (data loss / recording break)
    boundaries = []
    n = len(fdata)
    i = 0
    while i < n:
        if i > 0 and fdata[i][1] - fdata[i-1][1] > GPS_GAP_S:
            boundaries.append(i)
            i += 1
            continue
        if fdata[i][4] < STOP_SPEED_MS:
            j = i
            while j < n and fdata[j][4] < STOP_SPEED_MS:
                j += 1
            stop_dur = (fdata[j-1][1] - fdata[i][1]) if j > i else 0.0
            if stop_dur >= STOP_WINDOW_S:
                if i > 0:
                    boundaries.append(i)
                boundaries.append(j)
                i = j
                continue
        i += 1

    # build segment index ranges
    splits = sorted(set(boundaries))
    ranges, prev = [], 0
    for b in splits:
        if b > prev:
            ranges.append((prev, b))
        prev = b
    if prev < n:
        ranges.append((prev, n))

    # ── compute per-segment stats ──
    result = []
    out_idx = 0
    for start, end in ranges:
        seg = fdata[start:end]
        if len(seg) < MIN_SEG_FRAMES:
            continue

        fids = [f[0] for f in seg]
        lats = [f[2] for f in seg]
        lons = [f[3] for f in seg]

        total_dist = sum(
            haversine(lats[k-1], lons[k-1], lats[k], lons[k])
            for k in range(1, len(seg))
        )
        net_disp  = haversine(lats[0], lons[0], lats[-1], lons[-1])
        duration  = seg[-1][1] - seg[0][1]
        avg_speed = total_dist / duration if duration > 0 else 0.0
        straight  = net_disp / total_dist if total_dist > 0 else 0.0

        result.append({
            "seg_idx":      out_idx,
            "frame_ids":    fids,
            "frame_lat":    [round(x, 8) for x in lats],
            "frame_lon":    [round(x, 8) for x in lons],
            "n_frames":     len(fids),
            "net_disp_m":   round(net_disp, 2),
            "avg_speed_ms": round(avg_speed, 3),
            "straightness": round(straight, 3),
        })
        out_idx += 1

    return result


def process_group(group_name: str, rides_dir: Path, existing: dict) -> dict:
    result = dict(existing)
    ride_dirs = sorted([d for d in rides_dir.iterdir()
                        if d.is_dir() and d.name.startswith("ride_")])
    total = len(ride_dirs)
    t0 = time.time()

    for ri, ride_path in enumerate(ride_dirs):
        parts = ride_path.name.split("_")
        ride_id = parts[1] if len(parts) >= 2 else None
        if not ride_id or ride_id in result:
            continue

        segs = extract_ride_segments(ride_path, ride_id)
        if segs:
            result[ride_id] = {"segments": segs}
            elapsed = time.time() - t0
            print(f"[{ri+1:4d}/{total}] ride {ride_id}: {len(segs)} segments  "
                  f"({elapsed:.0f}s)", flush=True)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FrodoBots ride segments")
    parser.add_argument("--paths-config", type=Path,
                        default=_REPO / "configs" / "paths.yaml",
                        help="Path to paths.yaml")
    args = parser.parse_args()

    frodo_root = _load_frodo_root(args.paths_config)

    print(f"파라미터: stop_speed={STOP_SPEED_MS} m/s  stop_window={STOP_WINDOW_S}s  "
          f"gps_gap={GPS_GAP_S}s  min_frames={MIN_SEG_FRAMES}")
    print(f"FrodoBots root: {frodo_root}")
    print(f"처리 그룹: {SEG_GROUPS}\n")

    for group in SEG_GROUPS:
        subdir = ALL_GROUP_DIRS.get(group)
        if subdir is None:
            print(f"[skip] 알 수 없는 그룹: {group}")
            continue
        rides_dir = frodo_root / subdir
        if not rides_dir.exists():
            print(f"[skip] 디렉토리 없음: {rides_dir}")
            continue

        out_path = frodo_root / f"valid_segments_{group}.json"

        existing: dict = {}
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text())
                print(f"[{group}] 기존 {len(existing)} rides 로드, 신규 ride만 처리")
            except Exception:
                pass

        print(f"\n[{group}] {rides_dir} 처리 중...")
        result = process_group(group, rides_dir, existing)

        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        total_segs = sum(len(v["segments"]) for v in result.values())
        print(f"[{group}] 완료: {len(result)} rides, {total_segs} segments → {out_path.name}")

    print("\n=== extract_frodobots_segments 완료 ===")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
