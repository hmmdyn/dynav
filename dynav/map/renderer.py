"""MapRenderer — heading-up OSM map image generator.

Single entry point for all map rendering across the project:
  - Dataset building (rosbag, FrodoBots)
  - Inference ROS node

All visual parameters are identical across contexts; the only variable is
the source of ``route_latlons`` (OSRM /match for training, OSRM /route for
inference).

Modes
-----
``"rgb"``
    OSM tile RGB + coloured overlays.  Three-channel image compatible with
    ImageNet-pretrained EfficientNet-B0 weights.

``"hybrid"``
    Semantically separated channels:
      R — OSM tile converted to grayscale (road topology context).
      G — Future route as a Gaussian soft mask.
      B — Robot position (amp=1.0) and goal (amp=0.5) as Gaussians.
    Requires separate per-channel normalisation (see ``configs/map.yaml``).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from .tiles import TileCache, stitch_tiles, TILE_SIZE
from .routing import find_current_idx
from .overlay import (
    ROUTE_SIGMA_PX,
    draw_route_rgb,
    draw_robot_marker_rgb,
    draw_goal_marker_rgb,
    render_route_channel,
    render_position_goal_channel,
)


class MapRenderer:
    """Render heading-up OSM map images for dynav training and inference.

    Args:
        cache:          TileCache instance (shared across renderer calls).
        zoom:           OSM zoom level. 19 gives ~0.3 m/px near the equator.
        render_size:    Internal canvas size before heading-up rotation.
                        Must satisfy render_size > output_size * sqrt(2) to
                        avoid blank corners after rotation.
        output_size:    Final square image size (pixels) fed to MapEncoder.
        mode:           ``"rgb"`` or ``"hybrid"``.
        route_sigma_px: Gaussian σ for the route channel (hybrid mode only).
    """

    def __init__(
        self,
        cache: TileCache,
        zoom: int = 19,
        render_size: int = 512,
        output_size: int = 224,
        mode: str = "rgb",
        route_sigma_px: float = ROUTE_SIGMA_PX,
    ) -> None:
        if mode not in ("rgb", "hybrid"):
            raise ValueError(f"mode must be 'rgb' or 'hybrid', got {mode!r}")
        self._cache = cache
        self._zoom = zoom
        self._render_size = render_size
        self._output_size = output_size
        self._mode = mode
        self._route_sigma_px = route_sigma_px
        # Canvas size: padded so any heading rotation never exposes grey border
        self._canvas_px = math.ceil(render_size * math.sqrt(2)) + 2 * TILE_SIZE

    @classmethod
    def from_config(cls, cfg) -> "MapRenderer":
        """Construct from an OmegaConf ``cfg.map`` node.

        Args:
            cfg: OmegaConf DictConfig with a ``map`` sub-tree matching
                 ``configs/map.yaml``.

        Returns:
            Configured MapRenderer instance.
        """
        from .tiles import TileCache as _TC

        cache = _TC(
            cache_dir=cfg.map.tile_cache,
            tile_url=cfg.map.tile_url,
        )
        return cls(
            cache=cache,
            zoom=cfg.map.tile_zoom,
            render_size=cfg.map.render_size,
            output_size=cfg.map.output_size,
            mode=cfg.map.get("mode", "rgb"),
            route_sigma_px=cfg.map.get("route_sigma_px", ROUTE_SIGMA_PX),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        lat: float,
        lon: float,
        heading_deg: float,
        route_latlons: list[tuple[float, float]],
        goal_lat: float,
        goal_lon: float,
    ) -> Optional[Image.Image]:
        """Generate a heading-up map image centred on the robot.

        The robot's heading direction is rotated to point upward in the
        output image.  The route shown runs from the robot's current
        position to the goal; the past portion is omitted.

        Args:
            lat:            Robot latitude (degrees).
            lon:            Robot longitude (degrees).
            heading_deg:    Compass heading in degrees (0 = North, clockwise).
            route_latlons:  OSRM-computed route as [(lat, lon), ...].
                            Obtain via OSRMRouter.match() or OSRMRouter.route().
            goal_lat:       Goal latitude (degrees).
            goal_lon:       Goal longitude (degrees).

        Returns:
            PIL Image (RGB mode, output_size × output_size), or None if
            tile fetching failed or *route_latlons* is empty.
        """
        if not route_latlons:
            return None

        canvas, robot_px = stitch_tiles(
            self._cache, self._zoom, lat, lon, self._canvas_px
        )
        if canvas is None:
            return None

        # Closure: (lat, lon) → canvas pixel coordinates
        from .tiles import latlon_to_global_pixel

        origin_gx, origin_gy = latlon_to_global_pixel(lat, lon, self._zoom)
        cx_robot, cy_robot = robot_px
        # Global pixel of the canvas top-left corner
        canvas_origin_gx = origin_gx - cx_robot
        canvas_origin_gy = origin_gy - cy_robot

        def to_pixel(rlat: float, rlon: float) -> tuple[float, float]:
            gx, gy = latlon_to_global_pixel(rlat, rlon, self._zoom)
            return (gx - canvas_origin_gx, gy - canvas_origin_gy)

        goal_px = to_pixel(goal_lat, goal_lon)
        current_idx = find_current_idx(route_latlons, (lat, lon))

        if self._mode == "rgb":
            result = self._render_rgb(canvas, robot_px, goal_px, route_latlons, current_idx, to_pixel)
        else:
            result = self._render_hybrid(canvas, robot_px, goal_px, route_latlons, current_idx, to_pixel)

        if result is None:
            return None

        return self._rotate_and_crop(result, robot_px, heading_deg)

    # ------------------------------------------------------------------
    # Internal render modes
    # ------------------------------------------------------------------

    def _render_rgb(
        self,
        canvas: Image.Image,
        robot_px: tuple[float, float],
        goal_px: tuple[float, float],
        route_latlons: list[tuple[float, float]],
        current_idx: int,
        to_pixel,
    ) -> Image.Image:
        """Overlay coloured route, robot, and goal on the OSM tile canvas."""
        # Work on RGBA for alpha compositing
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        draw_route_rgb(draw, route_latlons, to_pixel, current_idx)
        draw_robot_marker_rgb(draw, *robot_px)
        draw_goal_marker_rgb(draw, *goal_px)

        base = canvas.convert("RGBA")
        composite = Image.alpha_composite(base, overlay)
        return composite.convert("RGB")

    def _render_hybrid(
        self,
        canvas: Image.Image,
        robot_px: tuple[float, float],
        goal_px: tuple[float, float],
        route_latlons: list[tuple[float, float]],
        current_idx: int,
        to_pixel,
    ) -> Image.Image:
        """Build a semantically separated 3-channel image.

        Ch R: OSM grayscale (luminance, uint8 [0,255])
        Ch G: Future route Gaussian mask (scaled to uint8 [0,255])
        Ch B: Robot (amp=1.0) + Goal (amp=0.5) Gaussians (scaled to uint8)
        """
        cpx = self._canvas_px

        # R channel — OSM grayscale
        r_ch = np.array(canvas.convert("L"), dtype=np.uint8)

        # G channel — route mask
        g_float = render_route_channel(
            cpx, route_latlons, to_pixel, current_idx, self._route_sigma_px
        )
        g_ch = (g_float * 255).astype(np.uint8)

        # B channel — position + goal
        b_float = render_position_goal_channel(cpx, robot_px, goal_px)
        b_ch = (b_float * 255).astype(np.uint8)

        rgb = np.stack([r_ch, g_ch, b_ch], axis=-1)
        return Image.fromarray(rgb, mode="RGB")

    # ------------------------------------------------------------------
    # Heading-up rotation + crop
    # ------------------------------------------------------------------

    def _rotate_and_crop(
        self,
        img: Image.Image,
        robot_px: tuple[float, float],
        heading_deg: float,
    ) -> Image.Image:
        """Rotate the canvas so the robot's heading points up, then crop.

        PIL.Image.rotate uses CCW-positive convention, so passing a CW
        compass heading rotates the canvas CCW by that amount, which brings
        the heading direction to the top.

        Args:
            img:        Canvas image (canvas_px × canvas_px).
            robot_px:   Rotation centre (robot position in canvas pixels).
            heading_deg: Compass heading (0 = North, clockwise).

        Returns:
            output_size × output_size PIL Image (RGB).
        """
        rotated = img.rotate(
            heading_deg,
            center=robot_px,
            expand=False,
            resample=Image.BICUBIC,
            fillcolor=(200, 200, 200),
        )

        # Crop output_size × output_size around the robot centre
        half = self._output_size // 2
        cx, cy = int(round(robot_px[0])), int(round(robot_px[1]))
        left   = cx - half
        top    = cy - half
        right  = left + self._output_size
        bottom = top  + self._output_size

        # Safety pad if crop extends outside canvas (shouldn't happen with
        # the canvas_px formula, but guards against edge cases)
        w, h = rotated.size
        if left < 0 or top < 0 or right > w or bottom > h:
            pad = self._output_size
            padded = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (200, 200, 200))
            padded.paste(rotated, (pad, pad))
            left += pad; top += pad; right += pad; bottom += pad
            rotated = padded

        cropped = rotated.crop((left, top, right, bottom))
        if cropped.size != (self._output_size, self._output_size):
            cropped = cropped.resize(
                (self._output_size, self._output_size), Image.LANCZOS
            )
        return cropped
