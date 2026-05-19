#!/usr/bin/env python3
"""
extract_frodobots_frames.py — valid_segments 기반 FrodoBots 프레임 추출

valid_segments_{group}.json에 정의된 segment frame_ids에 해당하는
stride-10 JPEG 프레임을 영상(.ts / m3u8)에서 추출해
<dataset_root>/frames/ride_{id}/{frame_id:06d}.jpg 로 저장.

ffmpeg로 m3u8 플레이리스트에서 stride-10 전체 프레임을 추출한 뒤,
segment에 필요한 frame_id만 남기고 나머지를 삭제.

이미 필요한 프레임이 모두 있는 ride는 건너뜀.

valid_segments_*.json 파일을 frodo_root에서 자동 탐색 (하드코딩 없음).

환경변수:
  DYNAV_FRODO_ROOT      FrodoBots 데이터 루트
  DYNAV_DATASET_ROOT    dataset 출력 루트
  DYNAV_FRAME_QUALITY   JPEG 품질 (-q:v, 2=최고~31=최저, default 3)
  DYNAV_FRAME_FORCE     "1" 이면 이미 추출된 ride도 재추출

Usage::

    python scripts/extract_frodobots_frames.py
    python scripts/extract_frodobots_frames.py --paths-config configs/paths.yaml
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

OBS_STRIDE   = 10
JPEG_QUALITY = int(os.environ.get("DYNAV_FRAME_QUALITY", "3"))
FORCE        = os.environ.get("DYNAV_FRAME_FORCE", "0") == "1"


# ── path loading ──────────────────────────────────────────────────────────────

def _load_paths(paths_config: Path) -> tuple:
    try:
        import yaml
        with open(paths_config) as f:
            cfg = yaml.safe_load(f) or {}
        frodo_default   = cfg.get("frodo_root",   "/data/frodobots")
        dataset_default = cfg.get("dataset_root", str(Path(frodo_default) / "dataset"))
    except Exception:
        frodo_default   = "/data/frodobots"
        dataset_default = str(Path(frodo_default) / "dataset")
    frodo_root   = Path(os.environ.get("DYNAV_FRODO_ROOT",   frodo_default)).expanduser()
    dataset_root = Path(os.environ.get("DYNAV_DATASET_ROOT", dataset_default)).expanduser()
    return frodo_root, dataset_root


# ── helpers ───────────────────────────────────────────────────────────────────

def _path_exists(p: Path) -> bool:
    try:
        return p.exists()
    except OSError:
        return False


def find_ride_dir(group_dir: Path, ride_id: str) -> Path | None:
    matches = list(group_dir.glob(f"ride_{ride_id}_*"))
    return matches[0] if matches else None


def find_m3u8(ride_dir: Path, uid: str = "1000") -> Path | None:
    """Return front-camera m3u8 (uid_s_1000) from recordings/."""
    recordings = ride_dir / "recordings"
    if not _path_exists(recordings):
        return None
    matches = sorted(recordings.glob(f"*__uid_s_{uid}__uid_e_video.m3u8"))
    return matches[0] if matches else None


def needed_frame_ids(segments: list) -> set:
    """
    Frame IDs required by all segments, including obs lookback
    (OBS_STRIDE * 3 frames before each segment start).
    """
    needed: set = set()
    for seg in segments:
        fids = seg["frame_ids"]
        needed.update(fids)
        if fids:
            first = fids[0]
            for k in range(1, 4):
                lb = first - OBS_STRIDE * k
                if lb >= 0:
                    needed.add(lb)
    return needed


def extract_all_stride10(m3u8_path: Path, out_dir: Path, quality: int = 3) -> None:
    """Extract every OBS_STRIDE-th frame from m3u8 into out_dir (ffmpeg 1-indexed names)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(m3u8_path),
        "-vf", f"select='not(mod(n\\,{OBS_STRIDE}))'",
        "-vsync", "0",
        "-q:v", str(quality),
        "-loglevel", "error",
        str(out_dir / "%06d.jpg"),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg를 찾을 수 없습니다. 설치 후 재시도하세요.\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  conda:         conda install -c conda-forge ffmpeg"
        )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 오류:\n{result.stderr[:500]}")


def rename_and_filter(tmp_dir: Path, frame_dir: Path, needed_ids: set) -> int:
    """
    Rename ffmpeg 1-indexed outputs → camera frame_id filenames,
    move only needed_ids into frame_dir, delete the rest.
    Returns number of frames kept.
    """
    frame_dir.mkdir(parents=True, exist_ok=True)
    tmp_files = sorted(tmp_dir.glob("*.jpg"))
    kept = 0
    for seq_idx, tmp_file in enumerate(tmp_files):
        frame_id = seq_idx * OBS_STRIDE  # 1st ffmpeg file = frame_id 0
        if frame_id in needed_ids:
            shutil.move(str(tmp_file), str(frame_dir / f"{frame_id:06d}.jpg"))
            kept += 1
        else:
            tmp_file.unlink()
    return kept


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FrodoBots video frames")
    parser.add_argument("--paths-config", type=Path,
                        default=_REPO / "configs" / "paths.yaml",
                        help="Path to paths.yaml")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("[error] ffmpeg를 찾을 수 없습니다. 설치 후 재시도하세요.")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  conda:         conda install -c conda-forge ffmpeg")
        sys.exit(1)

    frodo_root, dataset_root = _load_paths(args.paths_config)
    frame_base = dataset_root / "frames"
    frame_base.mkdir(parents=True, exist_ok=True)

    # load all valid_segments (auto-discover from frodo_root)
    seg_files = sorted(frodo_root.glob("valid_segments_*.json"))
    if not seg_files:
        print(f"[error] {frodo_root} 에서 valid_segments_*.json 없음")
        print("        extract_frodobots_segments.py 먼저 실행하세요")
        return

    all_segs: dict = {}  # ride_id → {segments, rides_dir}
    for seg_file in seg_files:
        # valid_segments_rides0.json → group "rides0" → output_rides_0
        group     = seg_file.stem[len("valid_segments_"):]   # e.g. "rides0"
        # "rides0" → "rides_0", "rides22" → "rides_22"
        suffix    = group[:5] + "_" + group[5:]              # "rides" + "_" + "0"
        group_dir = frodo_root / f"output_{suffix}"
        if not _path_exists(group_dir):
            print(f"[skip] {group}: rides 디렉토리 없음 ({group_dir})")
            continue
        for ride_id, v in json.loads(seg_file.read_text()).items():
            if ride_id not in all_segs:
                all_segs[ride_id] = {"segments": v["segments"], "rides_dir": group_dir}

    total_rides = len(all_segs)
    print(f"총 {total_rides} rides (valid_segments 기준)")
    print(f"FrodoBots root : {frodo_root}")
    print(f"Dataset root   : {dataset_root}")
    print(f"JPEG 품질: q={JPEG_QUALITY}  강제재추출: {FORCE}\n")

    t0 = time.time()
    done = skipped = total_frames = 0

    for ri, (ride_id, info) in enumerate(sorted(all_segs.items())):
        frame_dir = frame_base / f"ride_{ride_id}"
        needed    = needed_frame_ids(info["segments"])

        if not FORCE and frame_dir.exists():
            existing = {int(f.stem) for f in frame_dir.glob("*.jpg")}
            if needed.issubset(existing):
                skipped += 1
                continue

        ride_dir = find_ride_dir(info["rides_dir"], ride_id)
        if ride_dir is None:
            print(f"[{ri+1}/{total_rides}] ride {ride_id}: 디렉토리 없음, skip")
            continue

        m3u8 = find_m3u8(ride_dir)
        if m3u8 is None:
            print(f"[{ri+1}/{total_rides}] ride {ride_id}: m3u8 없음, skip")
            continue

        with tempfile.TemporaryDirectory(prefix=f"dynav_frames_{ride_id}_") as tmp:
            try:
                extract_all_stride10(m3u8, Path(tmp), quality=JPEG_QUALITY)
            except RuntimeError as e:
                print(f"[{ri+1}/{total_rides}] ride {ride_id}: {e}")
                continue
            kept = rename_and_filter(Path(tmp), frame_dir, needed)

        done += 1
        total_frames += kept
        elapsed = time.time() - t0
        rate = done / elapsed * 60 if elapsed > 0 else 0
        print(f"[{ri+1:4d}/{total_rides}] ride {ride_id}: {kept} frames 저장  "
              f"({done}완료/{skipped}skip, {rate:.1f} rides/min)", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== extract_frodobots_frames 완료 ({elapsed/60:.1f}분) ===")
    print(f"  추출: {done} rides, {total_frames} frames")
    print(f"  skip: {skipped} rides (이미 추출됨)")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
