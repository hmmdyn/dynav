"""
FrodoBots 유효 샘플 필터링 v2 — 순변위(net displacement) 기반

기존 v1의 문제: GPS 누적거리 기준 → 왔다갔다 구간도 통과
v2 개선: 프레임 t에서 t+HORIZON_S 사이의 직선 순변위가 MIN_NET_DISP_M 이상인
구간만 유효 샘플로 인정.

유효 샘플 조건 (프레임 단위):
  1. GPS 순변위: dist(GPS[t], GPS[t+HORIZON_S]) >= MIN_NET_DISP_M
  2. GPS jump 없음: 인접 GPS점 간격 <= MAX_GPS_JUMP_M
  3. 방향 일관성: 전반부/후반부 진행 방향 차이 <= MAX_TURN_DEG (선택)

출력: JSON — ride별 유효 세그먼트 목록 + 전체 통계
"""

import csv
import glob
import json
import math
import os
from pathlib import Path

# ── 파라미터 ──────────────────────────────────────────────────────────────────
DATA_ROOT       = Path.home() / "data/frodobots/output_rides_23"
OUTPUT_PATH     = Path.home() / "data/frodobots/valid_segments_v2.json"

MIN_NET_DISP_M  = 2.0    # 순변위 최소값 (m) — "실제로 2m 이상 전진"
HORIZON_S       = 5.0    # 순변위 측정 시간 범위 (초)
MAX_GPS_JUMP_M  = 10.0   # 이 이상 GPS 점프는 노이즈로 간주 → 세그먼트 차단
MAX_TURN_DEG    = 120.0  # 전반/후반 방향 변화 허용 최대값 (U턴 제거)
SEGMENT_MIN_FRAMES = 40  # 세그먼트 최소 프레임 수 (~2s @ 20Hz)
SEGMENT_GAP_FRAMES = 10  # 이 이하 간격의 무효 프레임은 bridge로 merge
FPS             = 20
HORIZON_FRAMES  = round(HORIZON_S * FPS)  # = 100
# ─────────────────────────────────────────────────────────────────────────────


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = (lat2 - lat1) * 111_000
    dlon = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111_000
    return math.sqrt(dlat ** 2 + dlon ** 2)


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """두 점 사이 방위각 (North=0, CW+, 도 단위)."""
    dlat = (lat2 - lat1) * 111_000
    dlon = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111_000
    return math.degrees(math.atan2(dlon, dlat))


def angle_diff_deg(a: float, b: float) -> float:
    """두 방위각 차이 절댓값 (0~180)."""
    d = abs((a - b + 180) % 360 - 180)
    return d


def load_gps(path: Path) -> list[tuple[float, float, float]]:
    """[(lat, lon, ts_s), ...] — GPS jump 제거 포함."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["latitude"]), float(r["longitude"]),
                         int(r["timestamp"]) / 1000.0))

    # GPS jump 제거
    cleaned = [rows[0]]
    n_jump = 0
    for i in range(1, len(rows)):
        dist = haversine_m(cleaned[-1][0], cleaned[-1][1], rows[i][0], rows[i][1])
        if dist <= MAX_GPS_JUMP_M:
            cleaned.append(rows[i])
        else:
            n_jump += 1
    return cleaned, n_jump


def load_camera_ts(path: Path) -> list[tuple[int, float]]:
    """[(frame_id, ts_s), ...]"""
    with open(path) as f:
        return [(int(r["frame_id"]), float(r["timestamp"])) for r in csv.DictReader(f)]


def interpolate_gps(gps: list, ts_s: float) -> tuple[float, float] | None:
    """카메라 timestamp에 GPS 선형 보간. 범위 밖이면 None."""
    if ts_s < gps[0][2] or ts_s > gps[-1][2]:
        return None
    lo, hi = 0, len(gps) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if gps[mid][2] <= ts_s:
            lo = mid
        else:
            hi = mid
    a, b = gps[lo], gps[hi]
    if b[2] == a[2]:
        return a[0], a[1]
    t = (ts_s - a[2]) / (b[2] - a[2])
    return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])


def compute_valid_mask(
    frame_lat: list[float],
    frame_lon: list[float],
) -> list[bool]:
    """
    각 프레임 i에 대해:
      - 순변위(i → i+HORIZON_FRAMES) >= MIN_NET_DISP_M  AND
      - 방향 일관성: 전반부/후반부 bearing 차이 <= MAX_TURN_DEG
    """
    n = len(frame_lat)
    valid = [False] * n

    for i in range(n - HORIZON_FRAMES):
        j = i + HORIZON_FRAMES

        # 조건 1: 순변위
        net = haversine_m(frame_lat[i], frame_lon[i],
                          frame_lat[j], frame_lon[j])
        if net < MIN_NET_DISP_M:
            continue

        # 조건 2: 방향 일관성 (전반/후반 bearing 차이)
        mid = (i + j) // 2
        b1 = bearing_deg(frame_lat[i],   frame_lon[i],
                         frame_lat[mid],  frame_lon[mid])
        b2 = bearing_deg(frame_lat[mid],  frame_lon[mid],
                         frame_lat[j],    frame_lon[j])
        if angle_diff_deg(b1, b2) > MAX_TURN_DEG:
            continue

        valid[i] = True

    return valid


def merge_segments(valid: list[bool]) -> list[tuple[int, int]]:
    segments = []
    in_seg = False
    start = 0
    gap = 0

    for i, v in enumerate(valid):
        if v:
            if not in_seg:
                in_seg = True
                start = i
            gap = 0
        else:
            if in_seg:
                gap += 1
                if gap > SEGMENT_GAP_FRAMES:
                    segments.append((start, i - gap))
                    in_seg = False
                    gap = 0

    if in_seg:
        segments.append((start, len(valid) - 1 - gap))

    return [(s, e) for s, e in segments if e - s + 1 >= SEGMENT_MIN_FRAMES]


def process_ride(ride_dir: Path) -> dict | None:
    rid = ride_dir.name.split("_")[1]

    gps_file = ride_dir / f"gps_data_{rid}.csv"
    cam_file_candidates = list(ride_dir.glob(f"front_camera_timestamps_{rid}.csv"))
    if not gps_file.exists() or not cam_file_candidates:
        return None

    gps_raw, n_jump = load_gps(gps_file)
    cam = load_camera_ts(cam_file_candidates[0])

    if len(gps_raw) < 2:
        return None

    # GPS → 프레임 보간
    interp = []
    valid_cam = []
    for fid, ts in cam:
        pt = interpolate_gps(gps_raw, ts)
        if pt is not None:
            interp.append(pt)
            valid_cam.append((fid, ts))

    if len(valid_cam) < HORIZON_FRAMES + 1:
        return None

    frame_lat = [p[0] for p in interp]
    frame_lon = [p[1] for p in interp]
    frame_ids = [fid for fid, _ in valid_cam]
    frame_ts  = [ts  for _, ts  in valid_cam]

    valid_mask = compute_valid_mask(frame_lat, frame_lon)
    segments   = merge_segments(valid_mask)

    result_segs = []
    for s, e in segments:
        seg_lat = frame_lat[s:e + 1]
        seg_lon = frame_lon[s:e + 1]

        # 누적거리
        cum_dist = sum(
            haversine_m(seg_lat[k], seg_lon[k], seg_lat[k+1], seg_lon[k+1])
            for k in range(len(seg_lat) - 1)
        )
        # 순변위
        net_disp = haversine_m(seg_lat[0], seg_lon[0], seg_lat[-1], seg_lon[-1])
        duration = frame_ts[e] - frame_ts[s]

        result_segs.append({
            "ride_id":      rid,
            "seg_idx":      len(result_segs),
            "start_frame":  frame_ids[s],
            "end_frame":    frame_ids[e],
            "n_frames":     e - s + 1,
            "duration_s":   round(duration, 2),
            "cum_dist_m":   round(cum_dist, 2),
            "net_disp_m":   round(net_disp, 2),
            "straightness": round(net_disp / cum_dist, 3) if cum_dist > 0 else 0,
            "avg_speed_ms": round(cum_dist / duration, 3) if duration > 0 else 0,
            "frame_ids":    frame_ids[s:e + 1],
            "frame_ts":     [round(t, 3) for t in frame_ts[s:e + 1]],
            "frame_lat":    [round(v, 8) for v in seg_lat],
            "frame_lon":    [round(v, 8) for v in seg_lon],
        })

    return {
        "ride_id":    rid,
        "ride_dir":   str(ride_dir),
        "n_cam_frames": len(cam),
        "n_gps_raw":  len(gps_raw) + n_jump,
        "n_gps_jumps": n_jump,
        "segments":   result_segs,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT,
                        help="ride_* 디렉토리들이 있는 루트 경로")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="결과 JSON 저장 경로")
    args = parser.parse_args()

    data_root   = args.data_root
    output_path = args.output

    ride_dirs = sorted(data_root.glob("ride_*"))
    print(f"\n데이터 루트: {data_root}")
    print(f"스캔 대상 ride: {len(ride_dirs)}개")
    print(f"필터 기준: 순변위 >= {MIN_NET_DISP_M}m / {HORIZON_S}s 이내, "
          f"방향변화 <= {MAX_TURN_DEG}°, 세그먼트 >= {SEGMENT_MIN_FRAMES}프레임\n")

    all_results = {}
    summary_rows = []

    for ride_dir in ride_dirs:
        result = process_ride(ride_dir)
        rid = ride_dir.name.split("_")[1]
        print(f"\r  처리 중: {rid}    ", end="", flush=True)

        if result is None:
            summary_rows.append((rid, 0, 0, 0, 0.0, "no camera"))
            continue

        segs = result["segments"]
        n_segs = len(segs)
        total_frames = sum(s["n_frames"] for s in segs)
        total_net    = sum(s["net_disp_m"] for s in segs)
        best_net     = max((s["net_disp_m"] for s in segs), default=0)

        flag = "✅" if n_segs > 0 else "—"
        summary_rows.append((rid, n_segs, total_frames, round(total_net, 1), best_net, flag))

        if n_segs > 0:
            all_results[rid] = result

    # 결과 출력
    print(f"{'ride':>8}  {'segs':>5}  {'frames':>7}  {'net_m':>7}  {'best_seg':>9}  {'판정':>4}")
    print("-" * 55)
    for rid, n_segs, frames, net, best, flag in summary_rows:
        print(f"{rid:>8}  {n_segs:>5}  {frames:>7}  {net:>7.1f}  {best:>9.1f}m  {flag}")

    total_segs   = sum(len(v["segments"]) for v in all_results.values())
    total_frames = sum(s["n_frames"] for v in all_results.values() for s in v["segments"])
    total_net    = sum(s["net_disp_m"] for v in all_results.values() for s in v["segments"])
    print("-" * 55)
    print(f"{'합계':>8}  {total_segs:>5}  {total_frames:>7}  {total_net:>7.1f}")

    # 세그먼트 상세 (유효 ride만)
    if all_results:
        print(f"\n=== 세그먼트 상세 (net_disp >= {MIN_NET_DISP_M}m) ===")
        for rid, result in all_results.items():
            print(f"\nride {rid}:")
            for s in result["segments"]:
                print(f"  seg{s['seg_idx']:02d}: {s['n_frames']:4d}f  "
                      f"{s['duration_s']:5.1f}s  "
                      f"cum={s['cum_dist_m']:6.1f}m  "
                      f"net={s['net_disp_m']:5.1f}m  "
                      f"직진율={s['straightness']:.2f}  "
                      f"{s['avg_speed_ms']:.2f}m/s")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n저장: {output_path}")


if __name__ == "__main__":
    main()
