"""
FrodoBots → DyNavDataset 빌드 파이프라인

Phase 1: .ts 비디오 → 프레임 JPEG 추출
Phase 2: 샘플 디렉토리 빌드 (obs_0..3.png, map.png, meta.json)
  - obs_0: 현재 프레임 (front cam)
  - obs_1: t - OBS_STRIDE 프레임
  - obs_2: t - 2*OBS_STRIDE 프레임
  - obs_3: obs_0 복사 (rear cam 없음)
  - map.png: OSM 타일 + 경로 오버레이 (224×224)
  - meta.json: gt_waypoints (5×2, body frame, [-1,1] 정규화) + route_direction

출력 구조:
  OUT_DIR/
    frames/ride_XXXXX/{frame_id:06d}.jpg   (임시, 재사용 가능)
    train/sample_XXXXXX/{obs_0..3.png, map.png, meta.json}
    val/sample_XXXXXX/...
    osm_cache/{z}_{x}_{y}.png              (타일 캐시)
"""

import csv
import glob
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw

# ── 파라미터 ──────────────────────────────────────────────────────────────────
RIDES = ["40268", "40272"]
DATA_ROOT   = Path.home() / "data/frodobots/output_rides_23"
SEGMENTS_JSON = Path.home() / "data/frodobots/valid_segments.json"
OUT_DIR     = Path.home() / "data/frodobots/dataset"

# 관측 이미지
OBS_STRIDE_FRAMES = 10   # obs_0, obs_1, obs_2 사이 프레임 간격 (0.5s @ 20Hz)
IMAGE_SIZE        = 224  # 출력 이미지 크기

# 웨이포인트
WAYPOINT_STRIDE_S = 1.0  # 웨이포인트 간격 (초)
N_WAYPOINTS       = 5
FPS               = 20
WAYPOINT_FRAMES   = [round(WAYPOINT_STRIDE_S * FPS * (i + 1)) for i in range(N_WAYPOINTS)]
MAX_WP_DIST_M     = 2.5  # 정규화 기준 (m)

# 샘플링
SAMPLE_STRIDE = 4        # 세그먼트 내 샘플 간격 (프레임)
VAL_RIDE      = "40272"  # val split에 사용할 ride

# OSM 지도
MAP_ZOOM   = 18          # zoom 18 ≈ 0.6 m/pixel → 224px ≈ 134m 커버리지
MAP_SIZE   = 224
OSM_URL    = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_CACHE  = OUT_DIR / "osm_cache"
OSM_HEADERS = {"User-Agent": "dynav-research/1.0 (lab project, non-commercial)"}
OSM_RETRY   = 3
OSM_DELAY_S = 0.05       # 타일 요청 간 최소 대기

# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 1: 프레임 추출
# ══════════════════════════════════════════════════════════════════════════════

def build_ts_index(recordings_dir: Path, uid: str) -> list[tuple[float, Path]]:
    """uid에 해당하는 .ts 파일들을 타임스탬프 순으로 정렬한 목록 반환.

    Returns: [(start_ts_s, path), ...]  — start_ts_s = Unix timestamp (초)
    """
    pattern = str(recordings_dir / f"*uid_s_{uid}*uid_e_video*.ts")
    files = sorted(glob.glob(pattern))
    result = []
    for f in files:
        # 파일명 끝 숫자 = ms timestamp: ..._video_20240504072852381.ts
        stem = Path(f).stem
        ts_str = stem.split("_")[-1]   # "20240504072852381" (17자리 = YYYYMMDDHHMMSSmmm)
        ts_ms = _parse_ts_filename(ts_str)
        result.append((ts_ms / 1000.0, Path(f)))
    return result


def _parse_ts_filename(ts_str: str) -> int:
    """'20240504072852381' → Unix milliseconds.

    Format: YYYYMMDDHHMMSSmmm (17 chars, last 3 = milliseconds not microseconds).
    """
    from datetime import datetime, timezone
    s = ts_str[:17]
    dt = datetime(
        int(s[0:4]), int(s[4:6]),  int(s[6:8]),   # year, month, day
        int(s[8:10]), int(s[10:12]), int(s[12:14]),  # hour, minute, second
        int(s[14:17]) * 1000,                         # ms → μs
        tzinfo=timezone.utc,
    )
    return int(dt.timestamp() * 1000)


def extract_frames(ride_id: str, force: bool = False) -> Path:
    """ride의 front camera .ts들을 모두 디코딩해 JPEG로 저장.

    Returns: frames_dir — {OUT_DIR}/frames/ride_{ride_id}/
    """
    ride_dirs = glob.glob(str(DATA_ROOT / f"ride_{ride_id}_*"))
    if not ride_dirs:
        raise FileNotFoundError(f"ride {ride_id} 디렉토리 없음")
    ride_dir = Path(ride_dirs[0])
    recordings_dir = ride_dir / "recordings"

    frames_dir = OUT_DIR / "frames" / f"ride_{ride_id}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 이미 추출됐으면 스킵 (frame count 체크)
    existing = sorted(frames_dir.glob("*.jpg"))
    cam_csv = ride_dir / f"front_camera_timestamps_{ride_id}.csv"
    with open(cam_csv) as f:
        total_frames = sum(1 for _ in f) - 1  # header 제외
    if not force and len(existing) == total_frames:
        print(f"  ride {ride_id}: 프레임 이미 추출됨 ({total_frames}장), 스킵")
        return frames_dir

    print(f"  ride {ride_id}: {total_frames}프레임 추출 중 → {frames_dir}")

    # concat 리스트 파일 생성
    ts_index = build_ts_index(recordings_dir, uid="1000")  # front cam
    if not ts_index:
        raise FileNotFoundError(f"ride {ride_id}: front camera .ts 파일 없음")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_file = f.name
        for _, ts_path in ts_index:
            f.write(f"file '{ts_path}'\n")

    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-q:v", "3",          # JPEG quality (1=best, 31=worst)
            "-vf", f"scale={IMAGE_SIZE}:{IMAGE_SIZE}",
            str(frames_dir / "%06d.jpg"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 실패:\n{result.stderr[-500:]}")

        # ffmpeg는 1-indexed로 저장 → 0-indexed로 rename
        jpegs = sorted(frames_dir.glob("*.jpg"))
        for i, p in enumerate(jpegs):
            target = frames_dir / f"{i:06d}.jpg"
            if p != target:
                p.rename(target)

        print(f"    → {len(jpegs)}장 저장 완료")
    finally:
        os.unlink(concat_file)

    return frames_dir


# ══════════════════════════════════════════════════════════════════════════════
#  OSM 타일 + 지도 생성
# ══════════════════════════════════════════════════════════════════════════════

def _tile_coord(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    lat_r = math.radians(lat)
    y = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return x, y


def _lat_lon_to_pixel_global(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """위경도 → zoom 레벨의 전역 픽셀 좌표 (타일 크기 256 기준)."""
    n = 2 ** zoom
    px = (lon + 180) / 360 * n * 256
    lat_r = math.radians(lat)
    py = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n * 256
    return px, py


def fetch_tile(z: int, x: int, y: int) -> Image.Image:
    """OSM 타일 다운로드 (캐시 사용)."""
    OSM_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = OSM_CACHE / f"{z}_{x}_{y}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    url = OSM_URL.format(z=z, x=x, y=y)
    for attempt in range(OSM_RETRY):
        try:
            resp = requests.get(url, headers=OSM_HEADERS, timeout=10)
            if resp.status_code == 200:
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                img.save(cache_path)
                time.sleep(OSM_DELAY_S)
                return img
        except requests.RequestException:
            pass
        time.sleep(0.5 * (attempt + 1))
    # 다운로드 실패 시 회색 타일
    return Image.new("RGB", (256, 256), (200, 200, 200))


def generate_map(
    center_lat: float,
    center_lon: float,
    heading_deg: float,
    gps_track: list[tuple[float, float]],  # (lat, lon) 리스트 (세그먼트 전체)
    future_waypoints: list[tuple[float, float]],  # (lat, lon) 5개 미래 지점
    zoom: int = MAP_ZOOM,
    size: int = MAP_SIZE,
) -> Image.Image:
    """OSM 타일을 다운받아 경로 오버레이가 있는 지도 이미지 생성."""
    # 센터 픽셀 (전역 좌표계)
    cx_g, cy_g = _lat_lon_to_pixel_global(center_lat, center_lon, zoom)

    # 필요한 타일 범위: 센터 ±(size/2) 픽셀
    half = size // 2
    x_min_g = cx_g - half
    x_max_g = cx_g + half
    y_min_g = cy_g - half
    y_max_g = cy_g + half

    tx_min = int(x_min_g // 256)
    tx_max = int(x_max_g // 256)
    ty_min = int(y_min_g // 256)
    ty_max = int(y_max_g // 256)

    # 타일 그리드 스티칭
    tile_cols = tx_max - tx_min + 1
    tile_rows = ty_max - ty_min + 1
    canvas_w = tile_cols * 256
    canvas_h = tile_rows * 256
    canvas = Image.new("RGB", (canvas_w, canvas_h))

    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            tile = fetch_tile(zoom, tx, ty)
            px = (tx - tx_min) * 256
            py = (ty - ty_min) * 256
            canvas.paste(tile, (px, py))

    # 전역 픽셀 → 캔버스 픽셀 변환 함수
    def to_canvas(lat, lon):
        gx, gy = _lat_lon_to_pixel_global(lat, lon, zoom)
        return (gx - tx_min * 256, gy - ty_min * 256)

    draw = ImageDraw.Draw(canvas, "RGBA")

    # 과거 경로 (현재까지): 회색 반투명
    center_idx = min(
        range(len(gps_track)),
        key=lambda i: abs(gps_track[i][0] - center_lat) + abs(gps_track[i][1] - center_lon),
    )
    past_pts = [to_canvas(lat, lon) for lat, lon in gps_track[: center_idx + 1]]
    if len(past_pts) >= 2:
        draw.line(past_pts, fill=(128, 128, 128, 180), width=3)

    # 미래 경로 (현재 이후): 빨간 점선
    future_track = gps_track[center_idx:]
    future_pts = [to_canvas(lat, lon) for lat, lon in future_track]
    if len(future_pts) >= 2:
        draw.line(future_pts, fill=(220, 50, 50, 200), width=3)

    # 미래 웨이포인트: 빨간 원
    for lat, lon in future_waypoints:
        cx, cy = to_canvas(lat, lon)
        r = 5
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 50, 50, 220))

    # 현재 위치: 파란 화살표
    cx_c, cy_c = to_canvas(center_lat, center_lon)
    _draw_arrow(draw, cx_c, cy_c, heading_deg, length=18, color=(30, 100, 220, 230))

    # 캔버스에서 size×size 크롭 (센터 기준)
    cx_canvas = cx_g - tx_min * 256
    cy_canvas = cy_g - ty_min * 256
    left   = int(cx_canvas) - half
    top    = int(cy_canvas) - half
    right  = left + size
    bottom = top + size

    # 경계 초과 시 패딩
    if left < 0 or top < 0 or right > canvas_w or bottom > canvas_h:
        padded = Image.new("RGB", (canvas_w + size, canvas_h + size), (200, 200, 200))
        padded.paste(canvas, (half, half))
        left += half; top += half; right += half; bottom += half
        canvas = padded

    return canvas.crop((left, top, right, bottom))


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    cx: float, cy: float,
    heading_deg: float,
    length: int = 16,
    color: tuple = (30, 100, 220, 230),
):
    """heading_deg(North=0, CW+) 방향의 화살표를 (cx, cy)에 그린다."""
    rad = math.radians(heading_deg)
    dx = math.sin(rad) * length
    dy = -math.cos(rad) * length   # 이미지는 y-아래가 양수

    ex, ey = cx + dx, cy + dy  # 화살표 끝

    # 몸통
    draw.line([(cx, cy), (ex, ey)], fill=color, width=4)

    # 화살촉 (equilateral triangle)
    head_len = length * 0.4
    head_rad = math.radians(heading_deg + 150)
    draw.polygon([
        (ex, ey),
        (ex + math.sin(head_rad) * head_len, ey - math.cos(head_rad) * head_len),
        (ex + math.sin(head_rad + math.radians(60)) * head_len,
         ey - math.cos(head_rad + math.radians(60)) * head_len),
    ], fill=color)

    # 현재 위치 점
    r = 6
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)


# ══════════════════════════════════════════════════════════════════════════════
#  웨이포인트 계산 (GPS → body frame)
# ══════════════════════════════════════════════════════════════════════════════

def compute_heading(frame_lats: list[float], frame_lons: list[float], idx: int) -> float:
    """인덱스 idx에서의 heading (도, North=0, CW+)을 GPS 이동 방향으로 추정."""
    window = 10  # ±window 프레임 범위에서 평균 방향
    i0 = max(0, idx - window)
    i1 = min(len(frame_lats) - 1, idx + window)
    if i1 == i0:
        return 0.0
    dlat = frame_lats[i1] - frame_lats[i0]
    dlon = frame_lons[i1] - frame_lons[i0]
    dN = dlat * 111_000
    dE = dlon * math.cos(math.radians(frame_lats[idx])) * 111_000
    return math.degrees(math.atan2(dE, dN))  # North=0, CW+


def gps_to_body(
    cur_lat: float, cur_lon: float,
    fut_lat: float, fut_lon: float,
    heading_deg: float,
) -> tuple[float, float]:
    """미래 GPS 점을 현재 body frame 상대 좌표 (dx, dy)로 변환.

    body frame: x = forward (heading 방향), y = left
    """
    dlat = fut_lat - cur_lat
    dlon = fut_lon - cur_lon
    dN = dlat * 111_000
    dE = dlon * math.cos(math.radians(cur_lat)) * 111_000

    hdg_rad = math.radians(heading_deg)
    dx =  dN * math.cos(hdg_rad) + dE * math.sin(hdg_rad)   # forward
    dy = -dN * math.sin(hdg_rad) + dE * math.cos(hdg_rad)   # left

    return dx, dy


def normalize_waypoint(dx: float, dy: float, max_dist: float = MAX_WP_DIST_M) -> list[float]:
    dx_n = max(-1.0, min(1.0, dx / max_dist))
    dy_n = max(-1.0, min(1.0, dy / max_dist))
    return [round(dx_n, 5), round(dy_n, 5)]


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2: 샘플 빌드
# ══════════════════════════════════════════════════════════════════════════════

def build_samples_for_ride(
    ride_id: str,
    segments: list[dict],
    frames_dir: Path,
    split: str,
    sample_counter: list,   # [current_count]  mutable
) -> int:
    """한 ride의 세그먼트들에서 샘플을 생성.

    Returns: 생성된 샘플 수
    """
    split_dir = OUT_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)
    n_created = 0

    for seg in segments:
        seg_idx   = seg["seg_idx"]
        frame_ids = seg["frame_ids"]   # 세그먼트 내 frame_id 리스트
        lats      = seg["frame_lat"]
        lons      = seg["frame_lon"]
        n_frames  = len(frame_ids)

        # GPS track (전체 세그먼트) — 지도 오버레이용
        gps_track = list(zip(lats, lons))

        # 샘플 가능한 인덱스 범위 (obs history + waypoint future 여유)
        obs_past_need = OBS_STRIDE_FRAMES * 2       # obs_1, obs_2 위해 필요한 과거
        wp_future_need = WAYPOINT_FRAMES[-1]         # 마지막 웨이포인트까지
        i_start = obs_past_need
        i_end   = n_frames - 1 - wp_future_need

        if i_end <= i_start:
            print(f"    ride {ride_id} seg{seg_idx:02d}: 샘플 범위 부족, 스킵")
            continue

        print(f"    ride {ride_id} seg{seg_idx:02d}: "
              f"{i_start}~{i_end} 범위에서 stride={SAMPLE_STRIDE} 샘플링")

        for local_i in range(i_start, i_end + 1, SAMPLE_STRIDE):
            fid  = frame_ids[local_i]
            lat  = lats[local_i]
            lon  = lons[local_i]

            # obs 프레임 경로 확인
            obs_paths = []
            valid = True
            for past_k in [0, OBS_STRIDE_FRAMES, OBS_STRIDE_FRAMES * 2]:
                src = frames_dir / f"{frame_ids[local_i - past_k]:06d}.jpg"
                if not src.exists():
                    valid = False
                    break
                obs_paths.append(src)
            if not valid:
                continue

            # 웨이포인트 GPS 지점
            heading = compute_heading(lats, lons, local_i)
            waypoints_norm = []
            future_gps = []
            for wp_offset in WAYPOINT_FRAMES:
                wi = local_i + wp_offset
                if wi >= n_frames:
                    valid = False
                    break
                fut_lat = lats[wi]
                fut_lon = lons[wi]
                future_gps.append((fut_lat, fut_lon))
                dx, dy = gps_to_body(lat, lon, fut_lat, fut_lon, heading)
                waypoints_norm.append(normalize_waypoint(dx, dy))
            if not valid:
                continue

            # route_direction: 웨이포인트 평균 방향 (body frame 라디안)
            sum_dx = sum(w[0] for w in waypoints_norm)
            sum_dy = sum(w[1] for w in waypoints_norm)
            route_dir = math.atan2(sum_dy, sum_dx)

            # 샘플 디렉토리 생성
            sample_name = f"sample_{sample_counter[0]:07d}"
            sample_dir  = split_dir / sample_name
            sample_dir.mkdir()

            # obs 이미지 복사 (symlink)
            for k, src in enumerate(obs_paths):
                dst = sample_dir / f"obs_{k}.png"
                _copy_as_png(src, dst, IMAGE_SIZE)
            # obs_3 = obs_0 복사 (rear cam 없음)
            shutil.copy(sample_dir / "obs_0.png", sample_dir / "obs_3.png")

            # OSM 지도 생성
            try:
                map_img = generate_map(
                    center_lat=lat,
                    center_lon=lon,
                    heading_deg=heading,
                    gps_track=gps_track,
                    future_waypoints=future_gps,
                )
                map_img.save(sample_dir / "map.png")
            except Exception as e:
                print(f"      지도 생성 실패 (sample {sample_counter[0]}): {e}")
                shutil.rmtree(sample_dir, ignore_errors=True)
                continue

            # meta.json
            meta = {
                "gt_waypoints": waypoints_norm,
                "route_direction": round(route_dir, 5),
                "ride_id": ride_id,
                "seg_idx": seg_idx,
                "frame_id": fid,
                "lat": round(lat, 8),
                "lon": round(lon, 8),
                "heading_deg": round(heading, 2),
            }
            with open(sample_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            sample_counter[0] += 1
            n_created += 1

    return n_created


def _copy_as_png(src: Path, dst: Path, size: int):
    """JPEG를 PNG로 변환하며 저장. 이미 size이면 그대로."""
    img = Image.open(src).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    img.save(dst)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    with open(SEGMENTS_JSON) as f:
        all_segments = json.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OSM_CACHE.mkdir(parents=True, exist_ok=True)

    sample_counter = [0]   # mutable 카운터 (train + val 합산)

    for ride_id in RIDES:
        segments = all_segments.get(ride_id, [])
        if not segments:
            print(f"ride {ride_id}: 세그먼트 없음, 스킵")
            continue

        split = "val" if ride_id == VAL_RIDE else "train"
        print(f"\n{'='*60}")
        print(f"ride {ride_id} → {split} split ({len(segments)}개 세그먼트)")

        # Phase 1: 프레임 추출
        print(f"\n[Phase 1] 프레임 추출")
        frames_dir = extract_frames(ride_id)

        # Phase 2: 샘플 빌드
        print(f"\n[Phase 2] 샘플 빌드")
        n = build_samples_for_ride(
            ride_id, segments, frames_dir, split, sample_counter
        )
        print(f"  → {n}개 샘플 생성 완료")

    # 최종 요약
    print(f"\n{'='*60}")
    print("데이터셋 빌드 완료")
    for split in ["train", "val"]:
        d = OUT_DIR / split
        if d.exists():
            n = len([p for p in d.iterdir() if p.is_dir()])
            print(f"  {split}: {n}개 샘플  ({d})")
    print(f"  OSM 타일 캐시: {len(list(OSM_CACHE.glob('*.png')))}장")


if __name__ == "__main__":
    main()
