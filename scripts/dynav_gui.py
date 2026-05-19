#!/usr/bin/env python3
"""
dynav_gui.py — FrodoBots 데이터 파이프라인 GUI

Tab 1: 세그먼트 리뷰  — GIF 확인 후 Keep/Reject, 완료 시 Reject 데이터 삭제
Tab 2: 파이프라인     — Step1 세그먼트 추출 / Step2 프레임 추출 / Step3 빌드
Tab 3: 샘플 뷰어      — 생성된 샘플 빠른 확인 (GIF + map + obs_0)

경로 설정: configs/paths.yaml 또는 환경 변수(DYNAV_FRODO_ROOT, DYNAV_DATASET_ROOT)
"""

import json, os, shutil, subprocess, sys, threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk

try:
    import yaml as _yaml
    def _load_yaml(path):
        with open(path) as f:
            return _yaml.safe_load(f) or {}
except ImportError:
    def _load_yaml(path):
        return {}

_REPO = Path(__file__).resolve().parent.parent


def _load_paths():
    """Load data paths from configs/paths.yaml, overridden by env vars."""
    cfg_path = _REPO / "configs" / "paths.yaml"
    cfg = _load_yaml(cfg_path) if cfg_path.exists() else {}
    frodo_root   = Path(os.environ.get("DYNAV_FRODO_ROOT",
                                        cfg.get("frodo_root",   "/data/frodobots"))).expanduser()
    dataset_root = Path(os.environ.get("DYNAV_DATASET_ROOT",
                                        cfg.get("dataset_root", str(frodo_root / "dataset")))).expanduser()
    return frodo_root, dataset_root


FROB_DIR, DATASET_DIR = _load_paths()
GIF_DIR     = DATASET_DIR / "gifs"
TRAIN_DIR   = DATASET_DIR / "train"
VAL_DIR     = DATASET_DIR / "val"
MANIFEST    = GIF_DIR / "manifest.json"
BUILD_SCRIPT  = _REPO / "scripts" / "build_frodobots_dataset.py"
SEG_SCRIPT    = _REPO / "scripts" / "extract_frodobots_segments.py"
FRAMES_SCRIPT = _REPO / "scripts" / "extract_frodobots_frames.py"


# ── helpers ────────────────────────────────────────────────────────────────────

def load_gif_frames(path, display_size=(300, 300)):
    frames, delays = [], []
    try:
        gif = Image.open(path)
        while True:
            frames.append(ImageTk.PhotoImage(
                gif.copy().convert("RGB").resize(display_size, Image.LANCZOS)))
            delays.append(gif.info.get("duration", 83))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames, delays


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def gif_items_from_manifest(manifest):
    items = []
    for key, meta in manifest.items():
        p = DATASET_DIR / key
        if p.exists():
            items.append((p, meta))
    return items


def gif_items_from_scan():
    if not GIF_DIR.exists():
        return []
    return [(p, None) for p in sorted(GIF_DIR.glob("*.gif"))]


# ── main window ────────────────────────────────────────────────────────────────

class DyNavGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DyNav Data Pipeline")
        self.geometry("980x720")
        self.resizable(True, True)

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.review_tab   = ReviewTab(nb)
        self.pipeline_tab = PipelineTab(nb, on_build_done=self.review_tab.reload)
        self.viewer_tab   = ViewerTab(nb)

        nb.add(self.review_tab,   text="  세그먼트 리뷰  ")
        nb.add(self.pipeline_tab, text="  파이프라인  ")
        nb.add(self.viewer_tab,   text="  샘플 뷰어  ")

        self.review_tab.reload()


# ── Tab 1: Review ──────────────────────────────────────────────────────────────

class ReviewTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._items     = []       # [(gif_path, meta|None)]
        self._decisions = {}       # str(path) → 'keep'|'reject'
        self._idx       = 0
        self._frames    = []
        self._delays    = []
        self._fidx      = 0
        self._anim_id   = None
        self._build_ui()
        self._bind_keys()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        gif_outer = ttk.LabelFrame(top, text="세그먼트 GIF")
        gif_outer.pack(side=tk.LEFT, padx=(0, 6))
        self._gif_lbl = tk.Label(gif_outer, bg="#1a1a2e", width=300, height=300)
        self._gif_lbl.pack(padx=4, pady=4)

        obs_outer = ttk.LabelFrame(top, text="obs_0.png")
        obs_outer.pack(side=tk.LEFT, padx=(0, 6))
        self._obs_lbl = tk.Label(obs_outer, bg="#111", width=224, height=224)
        self._obs_lbl.pack(padx=4, pady=4)
        self._obs_ph = None

        info_outer = ttk.LabelFrame(top, text="세그먼트 정보")
        info_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._info = {}
        rows = [
            ("파일",       "file"),
            ("ride_id",   "ride_id"),
            ("seg_idx",   "seg_idx"),
            ("split",     "split"),
            ("snap (m)",  "snap"),
            ("net_disp",  "net_disp"),
            ("speed",     "speed"),
            ("samples",   "samples"),
        ]
        for r, (label, key) in enumerate(rows):
            ttk.Label(info_outer, text=label + ":", anchor="e").grid(
                row=r, column=0, sticky="e", padx=(8, 4), pady=3)
            var = tk.StringVar(value="—")
            ttk.Label(info_outer, textvariable=var, anchor="w").grid(
                row=r, column=1, sticky="w", padx=(0, 8), pady=3)
            self._info[key] = var

        self._status_var = tk.StringVar(value="— 미결정 —")
        self._badge = tk.Label(info_outer, textvariable=self._status_var,
                               font=("Arial", 13, "bold"), width=14,
                               bg="#7f8c8d", fg="white", relief="flat")
        self._badge.grid(row=len(rows), column=0, columnspan=2, pady=(10, 4))

        nav = ttk.Frame(info_outer)
        nav.grid(row=len(rows)+1, column=0, columnspan=2, pady=4)
        ttk.Button(nav, text="◀ 이전 (←)", width=12,
                   command=self._prev).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="다음 (→) ▶", width=12,
                   command=self._next).pack(side=tk.LEFT, padx=3)

        self._prog_var = tk.StringVar(value="0 / 0")
        ttk.Label(info_outer, textvariable=self._prog_var).grid(
            row=len(rows)+2, column=0, columnspan=2, pady=2)
        self._pbar = ttk.Progressbar(info_outer, length=220, mode="determinate")
        self._pbar.grid(row=len(rows)+3, column=0, columnspan=2, pady=4)

        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, padx=8, pady=4)

        tk.Button(btn_row, text="✓  KEEP  [ K ]",
                  bg="#27ae60", fg="white", font=("Arial", 12, "bold"),
                  width=20, activebackground="#2ecc71",
                  command=lambda: self._decide("keep")).pack(side=tk.LEFT, padx=6)

        tk.Button(btn_row, text="✗  REJECT  [ R ]",
                  bg="#c0392b", fg="white", font=("Arial", 12, "bold"),
                  width=20, activebackground="#e74c3c",
                  command=lambda: self._decide("reject")).pack(side=tk.LEFT, padx=6)

        tk.Button(btn_row, text="↩ 미결정으로",
                  font=("Arial", 10), width=12,
                  command=lambda: self._decide(None)).pack(side=tk.LEFT, padx=6)

        self._summary_var = tk.StringVar(value="Keep: 0  |  Reject: 0  |  미결정: 0  |  총: 0")
        ttk.Label(self, textvariable=self._summary_var,
                  font=("Arial", 10), anchor="center").pack(fill=tk.X, padx=8)

        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(self, text="🗑  검토 완료 — Reject된 데이터 삭제 실행",
                   command=self._apply).pack(pady=6)

    def _bind_keys(self):
        root = self.winfo_toplevel()
        root.bind("k", lambda _: self._decide("keep"))
        root.bind("K", lambda _: self._decide("keep"))
        root.bind("r", lambda _: self._decide("reject"))
        root.bind("R", lambda _: self._decide("reject"))
        root.bind("<Right>", lambda _: self._next())
        root.bind("<Left>",  lambda _: self._prev())
        root.bind("<space>", lambda _: self._next())

    def reload(self):
        manifest = load_manifest()
        self._items = gif_items_from_manifest(manifest) if manifest else gif_items_from_scan()
        n = len(self._items)
        self._pbar["maximum"] = max(1, n)
        self._idx = 0
        self._refresh()

    def _refresh(self):
        if not self._items:
            self._gif_lbl.config(image="")
            for v in self._info.values():
                v.set("—")
            self._prog_var.set("GIF 없음")
            self._update_summary()
            return

        gif_path, meta = self._items[self._idx]
        self._info["file"].set(gif_path.name)
        if meta:
            self._info["ride_id"].set(meta.get("ride_id", "—"))
            self._info["seg_idx"].set(str(meta.get("seg_idx", "—")))
            self._info["split"].set(meta.get("split", "—"))
            self._info["snap"].set(f'{meta.get("osm_snap_mean_m", "—")} m')
            self._info["net_disp"].set(f'{meta.get("net_disp_m", "—")} m')
            self._info["speed"].set(f'{meta.get("avg_speed_ms", "—")} m/s')
            self._info["samples"].set(str(meta.get("n_samples", "—")))
        else:
            for k in ["ride_id", "seg_idx", "split", "snap", "net_disp", "speed", "samples"]:
                self._info[k].set("—")

        self._set_badge(self._decisions.get(str(gif_path)))

        n = len(self._items)
        reviewed = sum(1 for v in self._decisions.values() if v in ("keep", "reject"))
        self._prog_var.set(f"{self._idx + 1} / {n}   (검토 완료: {reviewed})")
        self._pbar["value"] = self._idx + 1
        self._update_summary()

        self._stop_anim()
        frames, delays = load_gif_frames(gif_path)
        self._frames = frames
        self._delays = delays
        self._fidx   = 0
        self._play()

        self._obs_ph = None
        self._obs_lbl.config(image="")
        if meta and meta.get("samples"):
            split    = meta.get("split", "train")
            obs_path = DATASET_DIR / split / meta["samples"][0] / "obs_0.png"
            if obs_path.exists():
                img = Image.open(str(obs_path)).convert("RGB").resize((224, 224), Image.LANCZOS)
                self._obs_ph = ImageTk.PhotoImage(img)
                self._obs_lbl.config(image=self._obs_ph)

    def _play(self):
        if not self._frames:
            return
        self._gif_lbl.config(image=self._frames[self._fidx])
        delay = self._delays[self._fidx] if self._delays else 83
        self._fidx = (self._fidx + 1) % len(self._frames)
        self._anim_id = self.after(delay, self._play)

    def _stop_anim(self):
        if self._anim_id:
            self.after_cancel(self._anim_id)
            self._anim_id = None
        self._frames = []

    def _set_badge(self, status):
        cfg = {
            "keep":   ("#27ae60", "white", "✓  KEEP"),
            "reject": ("#c0392b", "white", "✗  REJECT"),
            None:     ("#7f8c8d", "white", "— 미결정 —"),
        }
        bg, fg, text = cfg.get(status, cfg[None])
        self._badge.config(bg=bg, fg=fg)
        self._status_var.set(text)

    def _update_summary(self):
        n      = len(self._items)
        keep   = sum(1 for v in self._decisions.values() if v == "keep")
        reject = sum(1 for v in self._decisions.values() if v == "reject")
        self._summary_var.set(
            f"Keep: {keep}  |  Reject: {reject}  |  미결정: {n-keep-reject}  |  총: {n}")

    def _decide(self, status):
        if not self._items:
            return
        key = str(self._items[self._idx][0])
        if status is None:
            self._decisions.pop(key, None)
        else:
            self._decisions[key] = status
        self._set_badge(status)
        self._update_summary()
        if status in ("keep", "reject"):
            self._next()

    def _next(self):
        if self._items and self._idx < len(self._items) - 1:
            self._idx += 1
            self._refresh()

    def _prev(self):
        if self._items and self._idx > 0:
            self._idx -= 1
            self._refresh()

    def _apply(self):
        rejected = [p for p, v in self._decisions.items() if v == "reject"]
        if not rejected:
            messagebox.showinfo("알림", "Reject된 세그먼트가 없습니다.")
            return

        manifest = load_manifest()
        samples_to_del, gifs_to_del = [], []
        for path_str in rejected:
            gif_path = Path(path_str)
            key  = f"gifs/{gif_path.name}"
            meta = manifest.get(key, {})
            split = meta.get("split", "train")
            for sname in meta.get("samples", []):
                sdir = DATASET_DIR / split / sname
                if sdir.exists():
                    samples_to_del.append(sdir)
            gifs_to_del.append((gif_path, key))

        msg = (f"Reject 세그먼트: {len(rejected)}개\n"
               f"삭제될 샘플: {len(samples_to_del)}개\n\n"
               "정말 삭제하시겠습니까?")
        if not messagebox.askyesno("삭제 확인", msg, icon="warning"):
            return

        for sdir in samples_to_del:
            shutil.rmtree(sdir, ignore_errors=True)
        for gif_path, key in gifs_to_del:
            gif_path.unlink(missing_ok=True)
            manifest.pop(key, None)

        if MANIFEST.exists():
            MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

        for path_str in rejected:
            self._decisions.pop(path_str, None)

        messagebox.showinfo("완료",
            f"{len(samples_to_del)}개 샘플, {len(gifs_to_del)}개 GIF 삭제 완료.")
        self.reload()


# ── Tab 2: Pipeline ────────────────────────────────────────────────────────────

class PipelineTab(ttk.Frame):
    """
    3단계 파이프라인 (각 단계 독립 실행 가능):
      Step 1: 세그먼트 추출 (extract_frodobots_segments.py)
      Step 2: 프레임 추출  (extract_frodobots_frames.py)
      Step 3: 데이터셋 빌드 (build_frodobots_dataset.py)
    """

    def __init__(self, parent, on_build_done=None):
        super().__init__(parent)
        self._on_build_done = on_build_done
        self._proc           = None
        self._params_seg     = {}
        self._params_build   = {}
        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        # path display
        path_frame = ttk.LabelFrame(self, text="데이터 경로 (configs/paths.yaml)")
        path_frame.pack(fill=tk.X, padx=10, pady=(8, 0))
        ttk.Label(path_frame, text=f"FrodoBots root : {FROB_DIR}", anchor="w").pack(
            anchor="w", padx=8, pady=2)
        ttk.Label(path_frame, text=f"Dataset output : {DATASET_DIR}", anchor="w").pack(
            anchor="w", padx=8, pady=2)

        # ── Step 1: 세그먼트 추출 ──
        sf = ttk.LabelFrame(self, text="Step 1 — 세그먼트 추출 (extract_frodobots_segments.py)")
        sf.pack(fill=tk.X, padx=10, pady=(8, 4))

        seg_defs = [
            ("정지 속도 임계값 (m/s)", "DYNAV_SEG_STOP_SPEED",  "0.4",
             "이 속도 이하가 지속되면 정지로 판단"),
            ("정지 지속 시간 (s)",     "DYNAV_SEG_STOP_WINDOW", "3.0",
             "최소 정지 지속 시간 → 세그먼트 경계"),
            ("GPS 갭 임계값 (s)",      "DYNAV_SEG_GPS_GAP",     "5.0",
             "GPS 타임스탬프 갭 → 강제 분할"),
            ("최소 세그먼트 길이",     "DYNAV_SEG_MIN_FRAMES",  "50",
             "stride-10 프레임 기준 최소 세그먼트 크기"),
        ]
        self._build_param_grid(sf, seg_defs, self._params_seg)
        seg_btn_row = ttk.Frame(sf)
        seg_btn_row.grid(row=len(seg_defs), column=0, columnspan=3,
                         sticky="w", padx=10, pady=6)
        self._seg_btn = ttk.Button(seg_btn_row, text="▶  세그먼트 추출 실행",
                                   command=self._run_seg)
        self._seg_btn.pack(side=tk.LEFT, padx=4)

        # ── Step 2: 프레임 추출 ──
        ff = ttk.LabelFrame(self, text="Step 2 — 프레임 추출 (extract_frodobots_frames.py)")
        ff.pack(fill=tk.X, padx=10, pady=4)

        frame_inner = ttk.Frame(ff)
        frame_inner.pack(fill=tk.X, padx=10, pady=6)
        ttk.Label(frame_inner, text="JPEG 품질 (2=최고~31):", anchor="e").grid(
            row=0, column=0, sticky="e", padx=(0, 4), pady=4)
        self._frame_quality = tk.StringVar(value="3")
        ttk.Entry(frame_inner, textvariable=self._frame_quality, width=5).grid(
            row=0, column=1, sticky="w")
        ttk.Label(frame_inner, text="낮을수록 파일 크기 증가",
                  foreground="#555", font=("Arial", 9)).grid(
            row=0, column=2, sticky="w", padx=(8, 0))

        frame_btn_row = ttk.Frame(ff)
        frame_btn_row.pack(fill=tk.X, padx=10, pady=(0, 6))
        self._frame_btn = ttk.Button(frame_btn_row, text="▶  프레임 추출 실행",
                                     command=self._run_frames)
        self._frame_btn.pack(side=tk.LEFT, padx=4)

        # ── Step 3: 데이터셋 빌드 ──
        bf = ttk.LabelFrame(self, text="Step 3 — 데이터셋 빌드 (build_frodobots_dataset.py)")
        bf.pack(fill=tk.X, padx=10, pady=4)

        build_defs = [
            ("OSM Snap 임계값 (m)",  "DYNAV_OSM_SNAP_THRESH", "10.0",
             "GPS → OSM 보행자 도로 평균 거리 허용 상한"),
            ("최소 순변위 (m)",       "DYNAV_NET_DISP",         "10.0",
             "세그먼트 직선 전진 최소 거리 (왕복 제거)"),
            ("최소 속도 (m/s)",       "DYNAV_SPEED",            "0.7",
             "평균 이동 속도 하한"),
            ("최소 직선성",           "DYNAV_STRAIGHTNESS",     "0.75",
             "net_disp / 누적거리 비율 (1.0 = 완전 직선)"),
            ("샘플 stride (frames)", "DYNAV_SAMPLE_STRIDE",    "20",
             "세그먼트 내 샘플 간격 (20 = 1 s @ 20 fps)"),
        ]
        self._build_param_grid(bf, build_defs, self._params_build)
        build_btn_row = ttk.Frame(bf)
        build_btn_row.grid(row=len(build_defs), column=0, columnspan=3,
                           sticky="w", padx=10, pady=6)
        self._build_btn = ttk.Button(build_btn_row, text="▶  데이터셋 빌드 실행",
                                     command=self._run_build)
        self._build_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(build_btn_row, text="🗑  샘플·GIF 초기화",
                   command=self._clear_dataset).pack(side=tk.LEFT, padx=8)

        # ── 공통 진행바 + 중단 ──
        ctrl_row = ttk.Frame(self)
        ctrl_row.pack(fill=tk.X, padx=10, pady=2)
        self._stop_btn = ttk.Button(ctrl_row, text="■  중단",
                                    command=self._stop, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT, padx=4)
        self._pbar = ttk.Progressbar(ctrl_row, mode="indeterminate", length=400)
        self._pbar.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)

        # ── 공유 로그 ──
        lf = ttk.LabelFrame(self, text="로그")
        lf.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self._log = scrolledtext.ScrolledText(
            lf, height=10, font=("Courier", 9), state=tk.DISABLED)
        self._log.pack(fill=tk.BOTH, expand=True)

    def _build_param_grid(self, parent, defs, store):
        for r, (label, env_key, default, tip) in enumerate(defs):
            ttk.Label(parent, text=label + ":", anchor="e").grid(
                row=r, column=0, sticky="e", padx=(10, 4), pady=3)
            var = tk.StringVar(value=default)
            ttk.Entry(parent, textvariable=var, width=8).grid(
                row=r, column=1, sticky="w", padx=4)
            ttk.Label(parent, text=tip, foreground="#555",
                      font=("Arial", 9)).grid(
                row=r, column=2, sticky="w", padx=(6, 10))
            store[env_key] = var

    # ── step runners ─────────────────────────────────────────────────────

    def _run_seg(self):
        env = os.environ.copy()
        for k, v in self._params_seg.items():
            env[k] = v.get()
        self._log_clear()
        self._log_append("=== Step 1: 세그먼트 추출 ===\n")
        self._run_script(SEG_SCRIPT, env, label="세그먼트 추출")

    def _run_frames(self):
        env = os.environ.copy()
        env["DYNAV_FRAME_QUALITY"] = self._frame_quality.get()
        self._log_clear()
        self._log_append("=== Step 2: 프레임 추출 ===\n")
        self._run_script(FRAMES_SCRIPT, env, label="프레임 추출")

    def _run_build(self):
        env = os.environ.copy()
        for k, v in self._params_build.items():
            env[k] = v.get()
        self._log_clear()
        self._log_append(
            "=== Step 3: 데이터셋 빌드 ===\n" +
            "\n".join(f"  {k}={v.get()}" for k, v in self._params_build.items()) + "\n\n")
        self._run_script(BUILD_SCRIPT, env, label="빌드",
                         on_done=self._on_build_done)

    def _run_script(self, script, env, label="작업", on_done=None):
        self._set_running(True)
        self._pbar.start(12)

        def _worker():
            self._proc = subprocess.Popen(
                [sys.executable, str(script)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
            for line in self._proc.stdout:
                self.after(0, self._log_append, line)
            self._proc.wait()
            self.after(0, self._step_finished, self._proc.returncode, label, on_done)

        threading.Thread(target=_worker, daemon=True).start()

    # ── state helpers ────────────────────────────────────────────────────

    def _set_running(self, running: bool):
        s_run  = tk.DISABLED if running else tk.NORMAL
        s_stop = tk.NORMAL   if running else tk.DISABLED
        for btn in (self._seg_btn, self._frame_btn, self._build_btn):
            btn.config(state=s_run)
        self._stop_btn.config(state=s_stop)

    def _stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._log_append("\n⚠ 중단됨\n")
        self._step_finished(-1, "", None)

    def _step_finished(self, rc, label, on_done):
        self._pbar.stop()
        self._set_running(False)
        if rc == 0:
            self._log_append(f"\n✓ {label} 완료!\n")
            if on_done:
                on_done()
        elif rc != -1:
            self._log_append(f"\n✗ {label} 오류 (returncode={rc})\n")

    # ── dataset clear ────────────────────────────────────────────────────

    def _clear_dataset(self):
        msg = "train/, val/ 샘플과 gifs/ 폴더를 모두 삭제합니다.\n계속하시겠습니까?"
        if not messagebox.askyesno("초기화 확인", msg, icon="warning"):
            return
        self._log_append("초기화 중...\n")

        def _do():
            for d in [TRAIN_DIR, VAL_DIR, GIF_DIR]:
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)
            self.after(0, self._log_append, "초기화 완료\n")

        threading.Thread(target=_do, daemon=True).start()

    # ── log helpers ──────────────────────────────────────────────────────

    def _log_append(self, text):
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, text)
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)

    def _log_clear(self):
        self._log.config(state=tk.NORMAL)
        self._log.delete(1.0, tk.END)
        self._log.config(state=tk.DISABLED)


# ── Tab 3: Sample Viewer ───────────────────────────────────────────────────────

DISPLAY   = 224
GIF_SIZE  = 280


class ViewerTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._root_var  = tk.StringVar(value=str(DATASET_DIR / "train"))
        self._samples   = []
        self._idx       = 0
        self._frames    = []
        self._delays    = []
        self._fidx      = 0
        self._anim_id   = None
        self._build_ui()
        self._bind_keys()
        self._scan()

    def _build_ui(self):
        # Path bar
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(top, text="경로:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self._root_var, width=55).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="스캔", command=self._scan).pack(side=tk.LEFT)

        # Nav
        nav = ttk.Frame(self)
        nav.pack(fill=tk.X, padx=8, pady=2)
        ttk.Button(nav, text="◀ 이전 (A/←)", command=lambda: self._go(-1)).pack(side=tk.LEFT)
        self._title = tk.StringVar()
        ttk.Label(nav, textvariable=self._title,
                  font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=12)
        ttk.Button(nav, text="다음 (D/→) ▶", command=lambda: self._go(+1)).pack(side=tk.LEFT)

        # Images
        img_row = ttk.Frame(self)
        img_row.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        def panel(parent, label, w, h):
            f = ttk.LabelFrame(parent, text=label)
            f.pack(side=tk.LEFT, padx=4)
            lbl = tk.Label(f, bg="#111", width=w, height=h)
            lbl.pack(padx=3, pady=3)
            return lbl

        self._gif_lbl = panel(img_row, "segment.gif", GIF_SIZE, GIF_SIZE)
        self._map_lbl = panel(img_row, "map.png", DISPLAY, DISPLAY)
        self._obs_lbl = panel(img_row, "obs_0.png", DISPLAY, DISPLAY)

        self._meta_var = tk.StringVar()
        ttk.Label(self, textvariable=self._meta_var,
                  font=("Courier", 9), justify=tk.LEFT).pack(anchor=tk.W, padx=12, pady=4)

    def _bind_keys(self):
        root = self.winfo_toplevel()
        root.bind("a", lambda _: self._go(-1))
        root.bind("d", lambda _: self._go(+1))

    def _scan(self):
        p = Path(self._root_var.get())
        self._samples = sorted(p.iterdir()) if p.exists() else []
        self._idx = 0
        self._load()

    def _load(self):
        if not self._samples:
            self._title.set("샘플 없음")
            return
        s = self._samples[self._idx]
        n = len(self._samples)
        self._title.set(f"{s.name}  ({self._idx+1}/{n})")

        self._stop_anim()
        gif_path = s / "segment.gif"
        if gif_path.exists():
            frames, delays = [], []
            try:
                gif = Image.open(gif_path)
                while True:
                    frames.append(ImageTk.PhotoImage(
                        gif.copy().convert("RGB").resize((GIF_SIZE, GIF_SIZE), Image.LANCZOS)))
                    delays.append(gif.info.get("duration", 83))
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass
            self._frames, self._delays, self._fidx = frames, delays, 0
            self._play()
        else:
            self._gif_lbl.config(image="")

        self._show_static(self._map_lbl, s / "map.png", DISPLAY)
        self._show_static(self._obs_lbl, s / "obs_0.png", DISPLAY)

        meta_path = s / "meta.json"
        if meta_path.exists():
            m = json.loads(meta_path.read_text())
            lines = [
                f"ride={m.get('ride_id')}  seg={m.get('seg_idx')}  "
                f"frame={m.get('frame_id')}  split={m.get('split','?')}",
                f"lat={m.get('lat',0):.5f}  lon={m.get('lon',0):.5f}  "
                f"hdg={m.get('heading_deg',0):.1f}°",
                f"snap={m.get('osm_snap_mean_m','?')}m  "
                f"net_disp={m.get('net_disp_m','?')}m  "
                f"speed={m.get('avg_speed_ms','?')}m/s",
                f"wp[0]={m.get('gt_waypoints',[[]])[0]}",
            ]
            self._meta_var.set("\n".join(lines))

    def _show_static(self, label, path, size):
        if not path.exists():
            label.config(image="")
            return
        img = Image.open(str(path)).convert("RGB").resize((size, size), Image.LANCZOS)
        ph  = ImageTk.PhotoImage(img)
        label.config(image=ph)
        label._ph = ph

    def _play(self):
        if not self._frames:
            return
        self._gif_lbl.config(image=self._frames[self._fidx])
        delay = self._delays[self._fidx] if self._delays else 83
        self._fidx = (self._fidx + 1) % len(self._frames)
        self._anim_id = self.after(delay, self._play)

    def _stop_anim(self):
        if self._anim_id:
            self.after_cancel(self._anim_id)
            self._anim_id = None
        self._frames = []

    def _go(self, delta):
        if not self._samples:
            return
        self._idx = (self._idx + delta) % len(self._samples)
        self._load()


# ── entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DyNavGUI().mainloop()
