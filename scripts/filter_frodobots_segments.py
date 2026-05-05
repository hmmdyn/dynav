"""
FrodoBots ride에서 유효한 주행 세그먼트를 추출한다.

유효 세그먼트 조건:
  1. GPS 10초 윈도우 내 누적 이동거리 > MIN_WINDOW_DIST_M
  2. 세그먼트 길이 >= SEGMENT_MIN_FRAMES (카메라 프레임 수)
  3. GPS jump 없음 (연속 두 GPS점 사이 > MAX_GPS_JUMP_M)

출력: JSON — 세그먼트별 (start_frame, end_frame, gps_lat/lon per frame)
"""

import csv
import glob
import json
import math
import os
from dataclasses import dataclass, field

# ── 파라미터 ──────────────────────────────────────────────────────────────────
RIDES = ["40268", "40272"]
DATA_ROOT = os.path.expanduser("~/data/frodobots/output_rides_23")
OUTPUT_PATH = os.path.expanduser("~/data/frodobots/valid_segments.json")

GPS_WINDOW_SEC = 10.0    # GPS 이동 판정 윈도우 (초)
MIN_WINDOW_DIST_M = 3.0  # 윈도우 내 최소 누적 이동거리 (m)
MAX_GPS_JUMP_M = 15.0    # 이 이상은 GPS 노이즈로 간주, 해당 구간 세그먼트 차단
SEGMENT_MIN_FRAMES = 80  # 세그먼트 최소 프레임 수 (~4s @ 20Hz)
SEGMENT_GAP_FRAMES = 10  # 이 이하 간격의 정지 구간은 merge
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GpsPoint:
    lat: float
    lon: float
    ts_s: float  # Unix timestamp (초)


@dataclass
class Segment:
    start_frame: int
    end_frame: int
    frame_ids: list = field(default_factory=list)      # front_camera frame_id
    frame_ts: list = field(default_factory=list)       # timestamp (s)
    frame_lat: list = field(default_factory=list)      # 보간된 위도
    frame_lon: list = field(default_factory=list)      # 보간된 경도


def haversine_m(lat1, lon1, lat2, lon2):
    """두 위경도 사이 거리 (m), 단거리용 평면 근사."""
    dlat = (lat2 - lat1) * 111_000
    dlon = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111_000
    return math.sqrt(dlat**2 + dlon**2)


def load_gps(path) -> list[GpsPoint]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    pts = []
    for r in rows:
        pts.append(GpsPoint(
            lat=float(r["latitude"]),
            lon=float(r["longitude"]),
            ts_s=int(r["timestamp"]) / 1000.0,
        ))
    return pts


def load_camera_timestamps(path) -> list[tuple[int, float]]:
    """(frame_id, timestamp_s) 리스트 반환."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [(int(r["frame_id"]), float(r["timestamp"])) for r in rows]


def remove_gps_jumps(gps: list[GpsPoint]) -> list[GpsPoint]:
    """GPS jump 제거: 직전 점과 MAX_GPS_JUMP_M 이상 떨어진 점을 drop."""
    if not gps:
        return gps
    cleaned = [gps[0]]
    for pt in gps[1:]:
        prev = cleaned[-1]
        if haversine_m(prev.lat, prev.lon, pt.lat, pt.lon) <= MAX_GPS_JUMP_M:
            cleaned.append(pt)
        else:
            print(f"  [GPS jump 제거] {prev.ts_s:.1f}s → {pt.ts_s:.1f}s")
    return cleaned


def interpolate_gps(gps: list[GpsPoint], ts_s: float) -> tuple[float, float] | None:
    """카메라 timestamp에 GPS 선형 보간. GPS 범위 밖이면 None."""
    if ts_s < gps[0].ts_s or ts_s > gps[-1].ts_s:
        return None
    # 이진 탐색으로 앞뒤 점 찾기
    lo, hi = 0, len(gps) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if gps[mid].ts_s <= ts_s:
            lo = mid
        else:
            hi = mid
    a, b = gps[lo], gps[hi]
    if b.ts_s == a.ts_s:
        return a.lat, a.lon
    t = (ts_s - a.ts_s) / (b.ts_s - a.ts_s)
    return a.lat + t * (b.lat - a.lat), a.lon + t * (b.lon - a.lon)


def compute_moving_mask(
    frame_ts: list[float],
    frame_lat: list[float],
    frame_lon: list[float],
) -> list[bool]:
    """
    각 프레임에 대해 [ts, ts+GPS_WINDOW_SEC] 구간의 누적 GPS 이동거리를
    계산하고, MIN_WINDOW_DIST_M 이상이면 True.
    """
    n = len(frame_ts)
    moving = [False] * n
    j = 0  # 윈도우 우단 포인터
    for i in range(n):
        # 윈도우 우단을 GPS_WINDOW_SEC 이후까지 확장
        while j < n - 1 and frame_ts[j + 1] - frame_ts[i] <= GPS_WINDOW_SEC:
            j += 1
        # i..j 구간 누적 거리
        cum = 0.0
        for k in range(i, j):
            cum += haversine_m(
                frame_lat[k], frame_lon[k],
                frame_lat[k + 1], frame_lon[k + 1],
            )
        moving[i] = cum >= MIN_WINDOW_DIST_M
    return moving


def merge_segments(moving: list[bool]) -> list[tuple[int, int]]:
    """
    연속 True 구간을 세그먼트로 묶는다.
    SEGMENT_GAP_FRAMES 이하의 False 구간은 bridge로 merge.
    """
    segments = []
    in_seg = False
    start = 0
    gap_count = 0

    for i, m in enumerate(moving):
        if m:
            if not in_seg:
                in_seg = True
                start = i
            gap_count = 0
        else:
            if in_seg:
                gap_count += 1
                if gap_count > SEGMENT_GAP_FRAMES:
                    segments.append((start, i - gap_count))
                    in_seg = False
                    gap_count = 0

    if in_seg:
        segments.append((start, len(moving) - 1 - gap_count))

    # 최소 길이 필터
    segments = [(s, e) for s, e in segments if e - s + 1 >= SEGMENT_MIN_FRAMES]
    return segments


def process_ride(ride_id: str) -> list[dict]:
    dirs = glob.glob(os.path.join(DATA_ROOT, f"ride_{ride_id}_*"))
    if not dirs:
        print(f"ride {ride_id}: 디렉토리 없음")
        return []
    r = dirs[0]
    print(f"\n{'='*60}")
    print(f"ride {ride_id}: {r}")

    gps_raw = load_gps(os.path.join(r, f"gps_data_{ride_id}.csv"))
    gps = remove_gps_jumps(gps_raw)
    cam = load_camera_timestamps(
        os.path.join(r, f"front_camera_timestamps_{ride_id}.csv")
    )
    print(f"  GPS: {len(gps_raw)} → {len(gps)}점 (jump 제거 후)")
    print(f"  Camera: {len(cam)} 프레임")

    # GPS 보간
    valid_cam = []
    interp_lat, interp_lon = [], []
    for fid, ts in cam:
        result = interpolate_gps(gps, ts)
        if result is not None:
            interp_lat.append(result[0])
            interp_lon.append(result[1])
            valid_cam.append((fid, ts))

    print(f"  GPS 보간 가능 프레임: {len(valid_cam)} / {len(cam)}")

    if not valid_cam:
        return []

    frame_ids = [fid for fid, _ in valid_cam]
    frame_ts  = [ts  for _, ts  in valid_cam]

    # Moving mask 계산
    moving = compute_moving_mask(frame_ts, interp_lat, interp_lon)
    moving_count = sum(moving)
    print(f"  Moving 프레임: {moving_count} / {len(moving)} "
          f"({moving_count / len(moving):.0%})")

    # 세그먼트 추출
    raw_segs = merge_segments(moving)
    print(f"  세그먼트: {len(raw_segs)}개 (최소 {SEGMENT_MIN_FRAMES}프레임)")

    results = []
    total_frames = 0
    for i, (s, e) in enumerate(raw_segs):
        seg_frames = e - s + 1
        total_frames += seg_frames
        seg_dist = sum(
            haversine_m(interp_lat[k], interp_lon[k],
                        interp_lat[k+1], interp_lon[k+1])
            for k in range(s, e)
        )
        dur = frame_ts[e] - frame_ts[s]
        avg_spd = seg_dist / dur if dur > 0 else 0
        print(f"    seg {i:02d}: frame {frame_ids[s]}~{frame_ids[e]} "
              f"({seg_frames}f, {dur:.1f}s, {seg_dist:.1f}m, {avg_spd:.2f}m/s)")
        results.append({
            "ride_id": ride_id,
            "seg_idx": i,
            "start_frame": frame_ids[s],
            "end_frame":   frame_ids[e],
            "n_frames":    seg_frames,
            "duration_s":  round(dur, 2),
            "dist_m":      round(seg_dist, 2),
            "avg_speed_ms": round(avg_spd, 3),
            "frame_ids":   frame_ids[s:e+1],
            "frame_ts":    [round(t, 3) for t in frame_ts[s:e+1]],
            "frame_lat":   [round(v, 8) for v in interp_lat[s:e+1]],
            "frame_lon":   [round(v, 8) for v in interp_lon[s:e+1]],
        })

    print(f"  총 유효 프레임: {total_frames} / {len(valid_cam)} "
          f"({total_frames / len(valid_cam):.0%})")
    return results


def main():
    all_segments = {}
    summary = []

    for ride_id in RIDES:
        segs = process_ride(ride_id)
        all_segments[ride_id] = segs
        total_f = sum(s["n_frames"] for s in segs)
        total_d = sum(s["dist_m"] for s in segs)
        summary.append(
            f"  ride {ride_id}: {len(segs)}개 세그먼트, "
            f"{total_f}프레임, {total_d:.1f}m"
        )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_segments, f, indent=2)

    print(f"\n{'='*60}")
    print("요약:")
    for s in summary:
        print(s)
    print(f"\n저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
