"""
FrodoBots 전체 유효 ride → DyNavDataset 빌드 (v2)

설계:
  - 3개 source (output_rides_0/2/23)의 valid_segments JSON을 합산
  - 필요한 프레임만 .ts에서 선택 추출 (모든 프레임 X)
  - 샘플명: {ride_id}_{frame_id:07d}  (중복 없음, 전역 카운터 불필요)
  - train/val: int(ride_id) % 10 == 0 → val, 나머지 → train

파라미터:
  OBS_STRIDE     = 10  프레임 (0.5s) — obs_0/1/2 간격 및 프레임 추출 간격
  SAMPLE_STRIDE  = 20  프레임 (1s)   — 세그먼트 내 샘플 간격
  HORIZON_FRAMES = [20,40,60,80,100] — 웨이포인트 미래 프레임 (1~5s)
  IMAGE_SIZE     = 224
  MAP_ZOOM       = 18

출력:
  OUT_DIR/
    train/  val/          ← sample_RRRRRR_FFFFFFF/{obs_0..3.png, map.png, meta.json}
    frames/ride_{id}/     ← {frame_id:06d}.jpg  (공유, 재사용)
    osm_cache/            ← {z}_{x}_{y}.png
"""

import argparse
import csv
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image, ImageDraw

# ── 경로 ──────────────────────────────────────────────────────────────────────
FRODO_ROOT = Path.home() / "data/frodobots"
OUT_DIR    = FRODO_ROOT / "dataset"

SOURCES = [
    (FRODO_ROOT / "output_rides_0",  FRODO_ROOT / "valid_segments_rides0.json"),
    (FRODO_ROOT / "output_rides_2",  FRODO_ROOT / "valid_segments_rides2.json"),
    (FRODO_ROOT / "output_rides_23", FRODO_ROOT / "valid_segments_v2.json"),
]

# ── 파라미터 ──────────────────────────────────────────────────────────────────
OBS_STRIDE     = 10     # obs_0/1/2 사이 프레임 간격 (= 추출 간격, 0.5s @ 20Hz)
SAMPLE_STRIDE  = 20     # 세그먼트 내 샘플 간격 (1s)
FPS            = 20
N_WAYPOINTS    = 5
WAYPOINT_FRAMES = [FPS * (i + 1) for i in range(N_WAYPOINTS)]  # [20,40,60,80,100]
MAX_WP_DIST_M  = 2.5
IMAGE_SIZE     = 224
MAP_ZOOM       = 18
MAP_SIZE       = 224
OSM_URL        = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_HEADERS    = {"User-Agent": "dynav-research/1.0 (non-commercial)"}
OSM_CACHE      = OUT_DIR / "osm_cache"
FRAMES_DIR     = OUT_DIR / "frames"

# obs history에 필요한 최소 리드 프레임
OBS_PAD = OBS_STRIDE * 2      # = 20
# 미래 웨이포인트에 필요한 최소 후행 프레임
WP_PAD  = WAYPOINT_FRAMES[-1]  # = 100
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  유틸
# ══════════════════════════════════════════════════════════════════════════════

def haversine_m(lat1, lon1, lat2, lon2):
    dlat = (lat2 - lat1) * 111_000
    dlon = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111_000
    return math.sqrt(dlat**2 + dlon**2)


def _parse_ts_ms(stem: str) -> int:
    """파일명 끝 17자리 타임스탬프 → Unix ms."""
    ts_str = stem.split("_")[-1][:17]
    from datetime import datetime, timezone
    s = ts_str
    dt = datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]),
                  int(s[8:10]), int(s[10:12]), int(s[12:14]),
                  int(s[14:17]) * 1000, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def build_ts_index(recordings_dir: Path) -> list[tuple[float, Path]]:
    """front camera .ts 파일 목록 → [(start_ts_s, path), ...] 오름차순."""
    files = sorted(recordings_dir.glob("*uid_s_1000*uid_e_video*.ts"))
    return [(_parse_ts_ms(p.stem) / 1000.0, p) for p in files]


def load_camera_ts(cam_csv: Path) -> list[tuple[int, float]]:
    with open(cam_csv) as f:
        return [(int(r["frame_id"]), float(r["timestamp"]))
                for r in csv.DictReader(f)]


def frame_to_ts_local(ts_index, frame_ts_s) -> tuple[Path, int] | None:
    """카메라 timestamp → (ts_file_path, local_frame_idx).

    local_frame_idx = round((frame_ts - ts_start) * FPS)
    """
    lo, hi = 0, len(ts_index) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ts_index[mid][0] <= frame_ts_s:
            lo = mid
        else:
            hi = mid - 1
    ts_start, ts_path = ts_index[lo]
    local_idx = round((frame_ts_s - ts_start) * FPS)
    if local_idx < 0:
        return None
    return ts_path, local_idx


# ══════════════════════════════════════════════════════════════════════════════
#  프레임 추출
# ══════════════════════════════════════════════════════════════════════════════

def extract_needed_frames(
    ride_id: str,
    cam_list: list[tuple[int, float]],
    ts_index: list[tuple[float, Path]],
    needed_fids: set[int],
    frames_dir: Path,
) -> int:
    """needed_fids에 해당하는 프레임만 .ts에서 추출해 frames_dir에 저장.

    Returns: 실제 추출된 프레임 수
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 이미 있는 프레임은 스킵
    missing = {fid for fid in needed_fids
               if not (frames_dir / f"{fid:06d}.jpg").exists()}
    if not missing:
        return 0

    # frame_id → (cam_ts) 매핑
    fid_to_ts = {fid: ts for fid, ts in cam_list if fid in missing}

    # ts_file 별로 그룹화: {ts_path: [(global_fid, local_idx), ...]}
    ts_groups: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for fid, ts_s in fid_to_ts.items():
        result = frame_to_ts_local(ts_index, ts_s)
        if result is None:
            continue
        ts_path, local_idx = result
        ts_groups[ts_path].append((fid, local_idx))

    n_extracted = 0
    for ts_path, frame_list in ts_groups.items():
        if not ts_path.exists():
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            # .ts 전체 프레임 추출 (ffmpeg는 1-indexed)
            cmd = [
                "ffmpeg", "-y", "-i", str(ts_path),
                "-q:v", "3",
                "-vf", f"scale={IMAGE_SIZE}:{IMAGE_SIZE}",
                f"{tmpdir}/%06d.jpg",
            ]
            subprocess.run(cmd, capture_output=True)

            # 필요한 프레임만 영구 디렉토리로 이동
            for global_fid, local_idx in frame_list:
                src = Path(tmpdir) / f"{local_idx + 1:06d}.jpg"  # 1-indexed
                dst = frames_dir / f"{global_fid:06d}.jpg"
                if src.exists() and not dst.exists():
                    shutil.move(str(src), str(dst))
                    n_extracted += 1

    return n_extracted


# ══════════════════════════════════════════════════════════════════════════════
#  OSM 지도
# ══════════════════════════════════════════════════════════════════════════════

_tile_cache: dict[tuple, Image.Image] = {}  # 메모리 캐시 (프로세스 내)

def _tile_coord(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    lr = math.radians(lat)
    y = int((1 - math.log(math.tan(lr) + 1 / math.cos(lr)) / math.pi) / 2 * n)
    return x, y

def _lat_lon_to_px(lat, lon, zoom):
    n = 2 ** zoom
    px = (lon + 180) / 360 * n * 256
    lr = math.radians(lat)
    py = (1 - math.log(math.tan(lr) + 1 / math.cos(lr)) / math.pi) / 2 * n * 256
    return px, py

def fetch_tile(z, x, y) -> Image.Image:
    key = (z, x, y)
    if key in _tile_cache:
        return _tile_cache[key]
    cache_path = OSM_CACHE / f"{z}_{x}_{y}.png"
    if cache_path.exists():
        img = Image.open(cache_path).convert("RGB")
        _tile_cache[key] = img
        return img
    for _ in range(3):
        try:
            r = requests.get(OSM_URL.format(z=z, x=x, y=y),
                             headers=OSM_HEADERS, timeout=10)
            if r.status_code == 200:
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                img.save(cache_path)
                _tile_cache[key] = img
                time.sleep(0.05)
                return img
        except Exception:
            time.sleep(0.5)
    img = Image.new("RGB", (256, 256), (200, 200, 200))
    _tile_cache[key] = img
    return img

def generate_map(center_lat, center_lon, heading_deg,
                 gps_track, future_wps) -> Image.Image:
    cx_g, cy_g = _lat_lon_to_px(center_lat, center_lon, MAP_ZOOM)
    half = MAP_SIZE // 2
    tx_min = int((cx_g - half) // 256)
    tx_max = int((cx_g + half) // 256)
    ty_min = int((cy_g - half) // 256)
    ty_max = int((cy_g + half) // 256)

    cols, rows = tx_max - tx_min + 1, ty_max - ty_min + 1
    canvas = Image.new("RGB", (cols * 256, rows * 256))
    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            canvas.paste(fetch_tile(MAP_ZOOM, tx, ty),
                         ((tx - tx_min) * 256, (ty - ty_min) * 256))

    def to_c(lat, lon):
        gx, gy = _lat_lon_to_px(lat, lon, MAP_ZOOM)
        return (gx - tx_min * 256, gy - ty_min * 256)

    draw = ImageDraw.Draw(canvas, "RGBA")

    # 경로 (과거: 회색, 미래: 빨강)
    ci = min(range(len(gps_track)),
             key=lambda i: abs(gps_track[i][0]-center_lat)+abs(gps_track[i][1]-center_lon))
    pts_past   = [to_c(*p) for p in gps_track[:ci+1]]
    pts_future = [to_c(*p) for p in gps_track[ci:]]
    if len(pts_past)   >= 2: draw.line(pts_past,   fill=(128,128,128,170), width=3)
    if len(pts_future) >= 2: draw.line(pts_future, fill=(220,50,50,200),   width=3)
    for lat, lon in future_wps:
        cx, cy = to_c(lat, lon)
        draw.ellipse([cx-5, cy-5, cx+5, cy+5], fill=(220,50,50,220))

    # 로봇 화살표
    cx_c, cy_c = to_c(center_lat, center_lon)
    rad = math.radians(heading_deg)
    ex, ey = cx_c + math.sin(rad)*18, cy_c - math.cos(rad)*18
    draw.line([(cx_c, cy_c), (ex, ey)], fill=(30,100,220,230), width=4)
    hl = 7
    for dθ in [150, 210]:
        hr = math.radians(heading_deg + dθ)
        draw.polygon([(ex, ey),
                       (ex+math.sin(hr)*hl, ey-math.cos(hr)*hl),
                       (ex+math.sin(hr+math.radians(60))*hl,
                        ey-math.cos(hr+math.radians(60))*hl)],
                      fill=(30,100,220,230))
    draw.ellipse([cx_c-6, cy_c-6, cx_c+6, cy_c+6], fill=(30,100,220,230))

    # 크롭
    cx_c2 = cx_g - tx_min * 256
    cy_c2 = cy_g - ty_min * 256
    l, t = int(cx_c2)-half, int(cy_c2)-half
    r2, b = l+MAP_SIZE, t+MAP_SIZE
    cw, ch = canvas.size
    if l < 0 or t < 0 or r2 > cw or b > ch:
        pad = Image.new("RGB", (cw+MAP_SIZE, ch+MAP_SIZE), (200,200,200))
        pad.paste(canvas, (half, half))
        l += half; t += half; r2 += half; b += half
        canvas = pad
    return canvas.crop((l, t, r2, b))


# ══════════════════════════════════════════════════════════════════════════════
#  웨이포인트
# ══════════════════════════════════════════════════════════════════════════════

def compute_heading(lats, lons, idx) -> float:
    # w=40 (±2s at 20Hz) — GPS 1Hz이므로 ±0.5s(w=10)는 동일 GPS 구간에 걸려
    # atan2(0,0)=0 오류 발생. ±2s로 GPS 최소 2개 구간을 확보.
    w = 40
    i0, i1 = max(0, idx-w), min(len(lats)-1, idx+w)
    if i0 == i1:
        return 0.0
    dlat = lats[i1] - lats[i0]
    dlon = lons[i1] - lons[i0]
    dN = dlat * 111_000
    dE = dlon * math.cos(math.radians(lats[idx])) * 111_000
    if dN == 0.0 and dE == 0.0:
        return 0.0
    return math.degrees(math.atan2(dE, dN))

def gps_to_body(cur_lat, cur_lon, fut_lat, fut_lon, hdg_deg):
    dlat = fut_lat - cur_lat
    dlon = fut_lon - cur_lon
    dN = dlat * 111_000
    dE = dlon * math.cos(math.radians(cur_lat)) * 111_000
    h = math.radians(hdg_deg)
    dx = dN*math.cos(h) + dE*math.sin(h)
    dy = -dN*math.sin(h) + dE*math.cos(h)
    return dx, dy

def normalize_wp(dx, dy):
    return [round(max(-1., min(1., dx/MAX_WP_DIST_M)), 5),
            round(max(-1., min(1., dy/MAX_WP_DIST_M)), 5)]


# ══════════════════════════════════════════════════════════════════════════════
#  샘플 빌드
# ══════════════════════════════════════════════════════════════════════════════

def build_samples_for_segment(
    seg: dict,
    frames_dir: Path,
    split_dir: Path,
    gps_track: list[tuple[float,float]],
) -> int:
    fids  = seg["frame_ids"]
    lats  = seg["frame_lat"]
    lons  = seg["frame_lon"]
    n     = len(fids)
    rid   = seg["ride_id"]
    n_created = 0

    i_start = OBS_PAD
    i_end   = n - 1 - WP_PAD
    if i_end <= i_start:
        return 0

    for local_i in range(i_start, i_end + 1, SAMPLE_STRIDE):
        fid = fids[local_i]

        # obs 프레임 파일 확인
        obs_src = []
        ok = True
        for k in [0, OBS_STRIDE, OBS_STRIDE * 2]:
            f_id = fids[local_i - k]
            p = frames_dir / f"{f_id:06d}.jpg"
            if not p.exists():
                ok = False; break
            obs_src.append(p)
        if not ok:
            continue

        # 웨이포인트
        heading = compute_heading(lats, lons, local_i)
        waypoints, future_gps = [], []
        for wp_off in WAYPOINT_FRAMES:
            wi = local_i + wp_off
            if wi >= n:
                ok = False; break
            fl, flo = lats[wi], lons[wi]
            future_gps.append((fl, flo))
            dx, dy = gps_to_body(lats[local_i], lons[local_i], fl, flo, heading)
            waypoints.append(normalize_wp(dx, dy))
        if not ok:
            continue

        route_dir = math.atan2(sum(w[1] for w in waypoints),
                                sum(w[0] for w in waypoints))

        sample_name = f"sample_{rid}_{fid:07d}"
        sample_dir  = split_dir / sample_name
        if sample_dir.exists():
            n_created += 1
            continue
        sample_dir.mkdir()

        # obs 이미지 (JPEG → PNG 변환)
        for k, src in enumerate(obs_src):
            _save_as_png(src, sample_dir / f"obs_{k}.png")
        shutil.copy(sample_dir / "obs_0.png", sample_dir / "obs_3.png")

        # map
        try:
            img = generate_map(lats[local_i], lons[local_i],
                               heading, gps_track, future_gps)
            img.save(sample_dir / "map.png")
        except Exception as e:
            shutil.rmtree(sample_dir, ignore_errors=True)
            continue

        with open(sample_dir / "meta.json", "w") as f:
            json.dump({
                "gt_waypoints":    waypoints,
                "route_direction": round(route_dir, 5),
                "ride_id": rid,
                "seg_idx": seg["seg_idx"],
                "frame_id": fid,
                "lat": round(lats[local_i], 8),
                "lon": round(lons[local_i], 8),
                "heading_deg": round(heading, 2),
            }, f, indent=2)

        n_created += 1

    return n_created


def _save_as_png(src: Path, dst: Path):
    img = Image.open(src).convert("RGB")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    img.save(dst)


# ══════════════════════════════════════════════════════════════════════════════
#  ride 처리
# ══════════════════════════════════════════════════════════════════════════════

def process_ride(ride_id: str, ride_data: dict, data_root: Path) -> tuple[int, int]:
    """한 ride 전체 처리. Returns: (n_frames_extracted, n_samples_created)."""
    ride_dirs = sorted(data_root.glob(f"ride_{ride_id}_*"))
    if not ride_dirs:
        return 0, 0
    ride_dir = ride_dirs[0]

    segments = ride_data["segments"]
    if not segments:
        return 0, 0

    split = "val" if int(ride_id) % 10 == 0 else "train"
    split_dir = OUT_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # 필요한 frame_id 집합 계산
    fids_in_segs = {fid for seg in segments for fid in seg["frame_ids"]}
    needed_fids: set[int] = set()
    for seg in segments:
        fids = seg["frame_ids"]
        for local_i in range(OBS_PAD, len(fids) - WP_PAD, OBS_STRIDE):
            for k in [0, OBS_STRIDE, OBS_STRIDE * 2]:
                if local_i - k >= 0:
                    needed_fids.add(fids[local_i - k])

    frames_dir = FRAMES_DIR / f"ride_{ride_id}"

    # ts index 구축
    recordings_dir = ride_dir / "recordings"
    if not recordings_dir.exists():
        return 0, 0
    ts_index = build_ts_index(recordings_dir)
    if not ts_index:
        return 0, 0

    cam_csv = ride_dir / f"front_camera_timestamps_{ride_id}.csv"
    if not cam_csv.exists():
        return 0, 0
    cam_list = load_camera_ts(cam_csv)

    # 프레임 추출
    n_frames = extract_needed_frames(
        ride_id, cam_list, ts_index, needed_fids, frames_dir
    )

    # GPS track (지도 오버레이용 — 전체 세그먼트 합)
    all_lat = [lat for seg in segments for lat in seg["frame_lat"]]
    all_lon = [lon for seg in segments for lon in seg["frame_lon"]]
    gps_track = list(zip(all_lat, all_lon))

    # 샘플 생성 — gps_track은 ride 전체(세그먼트 합)로 넘겨 지도에 넓은 경로 표시
    n_samples = 0
    for seg in segments:
        n_samples += build_samples_for_segment(seg, frames_dir, split_dir, gps_track)

    return n_frames, n_samples


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="프레임 추출/샘플 생성 없이 통계만 출력")
    parser.add_argument("--rides-filter", nargs="*",
                        help="처리할 ride_id 목록 (미지정 시 전체)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OSM_CACHE.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # 모든 source JSON 병합
    all_rides: list[tuple[str, dict, Path]] = []  # (ride_id, ride_data, data_root)
    for data_root, seg_json in SOURCES:
        if not seg_json.exists():
            print(f"[경고] {seg_json} 없음, 스킵")
            continue
        with open(seg_json) as f:
            data = json.load(f)
        for rid, ride_data in data.items():
            if args.rides_filter and rid not in args.rides_filter:
                continue
            all_rides.append((rid, ride_data, data_root))

    # 통계 요약
    total_segs    = sum(len(v["segments"]) for _, v, _ in all_rides)
    total_frames  = sum(s["n_frames"] for _, v, _ in all_rides for s in v["segments"])
    n_train_rides = sum(1 for rid, _, _ in all_rides if int(rid) % 10 != 0)
    n_val_rides   = sum(1 for rid, _, _ in all_rides if int(rid) % 10 == 0)
    est_samples   = total_frames // SAMPLE_STRIDE

    print(f"\n{'='*60}")
    print(f"대상 ride:     {len(all_rides)}개  (train {n_train_rides} / val {n_val_rides})")
    print(f"유효 세그먼트: {total_segs}개")
    print(f"유효 프레임:   {total_frames:,}개")
    print(f"예상 샘플:     ~{est_samples:,}개")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("--dry-run 모드: 실제 빌드 없이 종료합니다.")
        return

    total_samples = 0
    total_extracted = 0

    for i, (rid, ride_data, data_root) in enumerate(all_rides):
        n_segs = len(ride_data["segments"])
        split = "val" if int(rid) % 10 == 0 else "train"
        print(f"[{i+1:4d}/{len(all_rides)}] ride {rid} ({split}, {n_segs}segs)", end=" ... ", flush=True)
        n_frames, n_samples = process_ride(rid, ride_data, data_root)
        total_extracted += n_frames
        total_samples   += n_samples
        print(f"frames={n_frames}  samples={n_samples}  (누계 {total_samples:,})")

    print(f"\n{'='*60}")
    print("데이터셋 빌드 완료")
    for split in ["train", "val"]:
        d = OUT_DIR / split
        if d.exists():
            n = sum(1 for p in d.iterdir() if p.is_dir())
            print(f"  {split}: {n:,}개 샘플")
    print(f"  추출 프레임: {total_extracted:,}장")
    print(f"  OSM 타일 캐시: {len(list(OSM_CACHE.glob('*.png')))}장")


if __name__ == "__main__":
    main()
