"""Zarr-backed reader for the FrodoBots-7k (LeRobot v1.6) dataset.

Verified layout (2026-06-10, /media/hmmdyn/MoAI Nav/frodobots_dataset):

  dataset_cache.zarr/            164,748,770 frames @ 10 fps, 187,156 episodes
    observation.filtered_position   (N, 2)  float64 — EKF position, local ENU
                                            metres (x=East, y=North), origin at
                                            episode start
    observation.filtered_heading    (N,)    float64 — EKF heading, radians CCW
                                            from East (validated against motion
                                            direction: median err 0.004 rad)
    observation.utm_position        (N, 2)  float64 — raw GPS in UTM (E, N)
    observation.utm_zone_number/letter      — letter stored as ASCII code
    observation.images.front.path  (N,)     — "videos/ride_<id>_<ts>_front_camera.mp4"
                                              (exactly one video per episode)
    observation.images.front.timestamp (N,) — seconds into that video; nominal
                                              step 0.1 s but dropouts up to ~4 s
                                              exist → split segments on gaps
  videos/                          224×128 @ ~20 fps, GOP=2 (fast exact seek)

Coordinate conventions used throughout the pipeline:

  body frame   : x = forward, y = left.
                 fwd = (cos h, sin h), left = (−sin h, cos h), h = filtered_heading.
  compass [deg]: (90 − degrees(h)) % 360   (0=N, CW) — what MapRenderer expects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# zone letters for which northing is measured from the equator (northern bands)
_NORTH_BANDS = set("NPQRSTUVWX")


@dataclass
class EpisodeData:
    """All per-frame arrays of one episode, plus derived geometry."""

    episode_idx: int
    start: int                  # global frame index of first frame
    pos: np.ndarray             # (n, 2) EKF ENU metres, origin = frame 0
    heading: np.ndarray         # (n,) radians CCW from East (unwrapped as stored)
    utm: np.ndarray             # (n, 2) raw GPS in UTM metres
    utm_zone: str               # e.g. "50R"
    video_path: str             # relative to dataset root, e.g. "videos/ride_…mp4"
    video_ts: np.ndarray        # (n,) seconds into video_path
    lat0: float                 # latitude of frame 0 (from UTM)
    lon0: float                 # longitude of frame 0 (from UTM)

    @property
    def n(self) -> int:
        return len(self.pos)

    @property
    def ride_id(self) -> str:
        """Ride identifier parsed from the video filename."""
        stem = Path(self.video_path).name
        # ride_<id>_<YYYYMMDDHHMMSS>_front_camera.mp4
        return stem.split("_front_camera")[0].split("_rear_camera")[0]

    def latlon(self, idx: int | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(lat, lon) of frame *idx* from the EKF position.

        Uses UTM zone of the episode + EKF displacement added to the UTM
        coordinate of frame 0.  More stable than the raw per-frame GPS.
        """
        import utm as _utm

        d = self.pos[idx] - self.pos[0]
        e = self.utm[0, 0] + d[..., 0]
        n = self.utm[0, 1] + d[..., 1]
        zone_num = int(self.utm_zone[:-1])
        northern = self.utm_zone[-1] in _NORTH_BANDS
        lat, lon = _utm.to_latlon(e, n, zone_num, northern=northern, strict=False)
        return lat, lon

    def compass_deg(self, idx: int) -> float:
        """Compass heading in degrees (0 = North, CW) for MapRenderer."""
        return (90.0 - math.degrees(self.heading[idx])) % 360.0

    def speed(self) -> np.ndarray:
        """(n,) speed in m/s from EKF position (first value duplicated)."""
        v = np.linalg.norm(np.diff(self.pos, axis=0), axis=1) * 10.0
        return np.concatenate([[v[0] if len(v) else 0.0], v])

    def to_body(self, idx: int, target_idx: int | np.ndarray) -> np.ndarray:
        """Displacement frame *idx* → *target_idx* in body frame (x fwd, y left)."""
        d = self.pos[target_idx] - self.pos[idx]            # (…, 2) ENU
        h = self.heading[idx]
        fwd = np.array([math.cos(h), math.sin(h)])
        left = np.array([-math.sin(h), math.cos(h)])
        return np.stack([d @ fwd, d @ left], axis=-1)


class Frodo7kReader:
    """Lazy zarr reader with cached episode boundaries."""

    def __init__(self, dataset_root: str | Path, bounds_cache: str | Path | None = None):
        import zarr

        self.root = Path(dataset_root)
        self.z = zarr.open(str(self.root / "dataset_cache.zarr"), mode="r")
        self._bounds = self._load_bounds(bounds_cache)

    # ------------------------------------------------------------------
    def _load_bounds(self, cache: str | Path | None) -> np.ndarray:
        if cache is not None:
            cache = Path(cache)
            if cache.exists():
                return np.load(cache)
        ep = self.z["episode_index"][:]
        n_ep = int(ep[-1]) + 1
        bounds = np.searchsorted(ep, np.arange(n_ep + 1))
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, bounds)
        return bounds

    @property
    def n_episodes(self) -> int:
        return len(self._bounds) - 1

    def episode_lengths(self) -> np.ndarray:
        return np.diff(self._bounds)

    # ------------------------------------------------------------------
    def episode(self, e: int) -> EpisodeData:
        s, t = int(self._bounds[e]), int(self._bounds[e + 1])
        z = self.z
        utm = z["observation.utm_position"][s:t]
        zone_num = int(z["observation.utm_zone_number"][s])
        zone_letter = chr(int(z["observation.utm_zone_letter"][s]))
        video_path = str(z["observation.images.front.path"][s])

        import utm as _utm

        lat0, lon0 = _utm.to_latlon(
            utm[0, 0], utm[0, 1], zone_num,
            northern=zone_letter in _NORTH_BANDS, strict=False,
        )
        return EpisodeData(
            episode_idx=e,
            start=s,
            pos=z["observation.filtered_position"][s:t],
            heading=z["observation.filtered_heading"][s:t],
            utm=utm,
            utm_zone=f"{zone_num}{zone_letter}",
            video_path=video_path,
            video_ts=z["observation.images.front.timestamp"][s:t],
            lat0=float(lat0),
            lon0=float(lon0),
        )

    def video_file(self, ep: EpisodeData) -> Path:
        return self.root / ep.video_path
