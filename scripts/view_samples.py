#!/usr/bin/env python3
"""
view_samples.py — 생성된 샘플 빠른 확인 뷰어.
segment.gif + obs_0.png (카메라) + map.png (지도) 를 나란히 표시.
←/→ 또는 A/D 로 샘플 이동.

Usage:
    python scripts/view_samples.py                          # train 기본 경로
    python scripts/view_samples.py /path/to/dataset/train
"""
import json, sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk
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

if len(sys.argv) > 1:
    SAMPLES_ROOT = Path(sys.argv[1])
else:
    cfg_path = _REPO / "configs" / "paths.yaml"
    if cfg_path.exists():
        cfg = _load_yaml(cfg_path)
        SAMPLES_ROOT = Path(cfg.get("dataset_root", "/home/hmmdyn/data/frodobots/dataset")).expanduser() / "train"
    else:
        SAMPLES_ROOT = Path("/home/hmmdyn/data/frodobots/dataset/train")

DISPLAY  = 224
GIF_SIZE = 280


class Viewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sample Viewer")
        self.resizable(True, True)

        self._samples = sorted(SAMPLES_ROOT.iterdir()) if SAMPLES_ROOT.exists() else []
        self._idx     = 0
        self._frames  = []
        self._delays  = []
        self._fidx    = 0
        self._anim_id = None

        self._build_ui()
        self.bind("<Left>",  lambda _: self._go(-1))
        self.bind("<Right>", lambda _: self._go(+1))
        self.bind("a",       lambda _: self._go(-1))
        self.bind("d",       lambda _: self._go(+1))
        self._load()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(top, text="◀ 이전 (A/←)", command=lambda: self._go(-1)).pack(side=tk.LEFT)
        self._title = tk.StringVar()
        ttk.Label(top, textvariable=self._title,
                  font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=12)
        ttk.Button(top, text="다음 (D/→) ▶", command=lambda: self._go(+1)).pack(side=tk.LEFT)

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


if __name__ == "__main__":
    Viewer().mainloop()
