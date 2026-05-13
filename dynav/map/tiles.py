"""OSM tile fetching, caching, and stitching.

Coordinates use the standard Web Mercator / Slippy Map convention:
  tile_coord(lat, lon, zoom) → (tx, ty) where ty increases southward.
  latlon_to_global_pixel follows the same convention.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Optional

from PIL import Image

TILE_SIZE: int = 256


# ── Coordinate helpers ─────────────────────────────────────────────────────────

def tile_coord(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Return OSM tile (tx, ty) for the given lat/lon at *zoom*.

    Args:
        lat:  Latitude in degrees.
        lon:  Longitude in degrees.
        zoom: OSM zoom level (0–19).

    Returns:
        Integer tile coordinates (tx, ty).
    """
    n = 2 ** zoom
    tx = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    ty = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return tx, ty


def latlon_to_global_pixel(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Return sub-pixel global pixel coordinates at *zoom*.

    Global pixel (0, 0) is at the top-left corner of tile (0, 0).
    Pixel x increases eastward; pixel y increases southward.

    Args:
        lat:  Latitude in degrees.
        lon:  Longitude in degrees.
        zoom: OSM zoom level.

    Returns:
        (px, py) as floats.
    """
    n = 2 ** zoom
    px = (lon + 180.0) / 360.0 * n * TILE_SIZE
    lat_r = math.radians(lat)
    py = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n * TILE_SIZE
    return px, py


# ── TileCache ──────────────────────────────────────────────────────────────────

class TileCache:
    """Disk-backed OSM tile cache with HTTP fallback.

    Tiles are stored as ``<cache_dir>/<z>/<x>/<y>.png``.
    On a cache miss the tile is downloaded and saved for future use.
    On download failure a grey placeholder is returned so rendering can
    continue even when offline or the tile server is unavailable.

    Args:
        cache_dir:   Local directory for tile storage.
        tile_url:    URL template with ``{z}``, ``{x}``, ``{y}`` placeholders.
        user_agent:  HTTP User-Agent header (OSM policy requires identification).
        max_retries: Number of download attempts before giving up.
        retry_delay: Seconds to wait between retries.
    """

    def __init__(
        self,
        cache_dir: str,
        tile_url: str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        user_agent: str = "dynav-research/1.0",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._tile_url = tile_url
        self._user_agent = user_agent
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, z: int, x: int, y: int) -> Path:
        p = self._cache_dir / str(z) / str(x) / f"{y}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _grey_tile(self) -> Image.Image:
        return Image.new("RGB", (TILE_SIZE, TILE_SIZE), (200, 200, 200))

    def get(self, z: int, x: int, y: int) -> Image.Image:
        """Return the tile image for (z, x, y), downloading if necessary.

        Args:
            z: Zoom level.
            x: Tile column.
            y: Tile row.

        Returns:
            TILE_SIZE × TILE_SIZE PIL Image (RGB).
            Falls back to a grey placeholder on any failure.
        """
        path = self._cache_path(z, x, y)
        if path.exists():
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                path.unlink(missing_ok=True)

        url = self._tile_url.format(z=z, x=x, y=y)
        for attempt in range(self._max_retries):
            try:
                import urllib.request

                req = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                path.write_bytes(data)
                import io
                return Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay)

        return self._grey_tile()


# ── Canvas stitching ───────────────────────────────────────────────────────────

def stitch_tiles(
    cache: TileCache,
    zoom: int,
    center_lat: float,
    center_lon: float,
    canvas_px: int,
) -> tuple[Image.Image, tuple[float, float]]:
    """Stitch OSM tiles into a square canvas centred on (center_lat, center_lon).

    The canvas is large enough that after rotating by any heading angle and
    cropping to ``output_size``, no grey padding will be visible, provided
    ``canvas_px = ceil(output_size * sqrt(2)) + 2 * TILE_SIZE``.

    Args:
        cache:      TileCache instance.
        zoom:       OSM zoom level.
        center_lat: Centre latitude in degrees.
        center_lon: Centre longitude in degrees.
        canvas_px:  Desired canvas side length in pixels.

    Returns:
        canvas:   PIL Image (RGB) of size canvas_px × canvas_px.
        robot_px: Robot position in canvas coordinates (cx, cy).
    """
    # Global pixel position of the robot
    gx, gy = latlon_to_global_pixel(center_lat, center_lon, zoom)

    half = canvas_px / 2.0

    # Tile range needed to cover [gx-half, gx+half] × [gy-half, gy+half]
    tx_min = int((gx - half) / TILE_SIZE)
    ty_min = int((gy - half) / TILE_SIZE)
    tx_max = int(math.ceil((gx + half) / TILE_SIZE))
    ty_max = int(math.ceil((gy + half) / TILE_SIZE))

    # Pixel offset of top-left tile in global coords
    origin_px = tx_min * TILE_SIZE
    origin_py = ty_min * TILE_SIZE

    canvas_w = (tx_max - tx_min) * TILE_SIZE
    canvas_h = (ty_max - ty_min) * TILE_SIZE
    canvas = Image.new("RGB", (canvas_w, canvas_h), (200, 200, 200))

    for tx in range(tx_min, tx_max):
        for ty in range(ty_min, ty_max):
            tile = cache.get(zoom, tx, ty)
            paste_x = (tx - tx_min) * TILE_SIZE
            paste_y = (ty - ty_min) * TILE_SIZE
            canvas.paste(tile, (paste_x, paste_y))

    # Robot position within the full tile canvas
    robot_px = (gx - origin_px, gy - origin_py)

    # Crop to canvas_px × canvas_px centred on the robot
    cx, cy = robot_px
    left = int(cx - half)
    top  = int(cy - half)
    right  = left + canvas_px
    bottom = top  + canvas_px

    # Pad if the crop extends outside the tile canvas
    pad_left  = max(0, -left)
    pad_top   = max(0, -top)
    pad_right = max(0, right - canvas_w)
    pad_bot   = max(0, bottom - canvas_h)

    if any((pad_left, pad_top, pad_right, pad_bot)):
        padded = Image.new(
            "RGB",
            (canvas_w + pad_left + pad_right, canvas_h + pad_top + pad_bot),
            (200, 200, 200),
        )
        padded.paste(canvas, (pad_left, pad_top))
        canvas = padded
        left  += pad_left
        top   += pad_top
        right += pad_left
        bottom += pad_top
        cx += pad_left
        cy += pad_top

    canvas = canvas.crop((left, top, right, bottom))
    robot_px = (cx - left, cy - top)

    return canvas, robot_px
