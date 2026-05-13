"""Map overlay drawing primitives for rgb and hybrid rendering modes.

All visual constants are defined here so that every pipeline (rosbag,
FrodoBots, inference ROS node) produces identical-looking map images.

rgb mode  — PIL ImageDraw, coloured overlays on OSM tile canvas.
hybrid mode — numpy arrays, one semantic channel per concern.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

# ── Visual constants (identical across all contexts) ───────────────────────────

# Future route (current position → goal)
ROUTE_FUTURE_COLOR: tuple[int, int, int, int] = (220, 50, 50, 200)
ROUTE_LINE_WIDTH: int = 3  # px

# Robot position marker — circle (heading-up makes arrow redundant)
ROBOT_COLOR: tuple[int, int, int, int] = (30, 100, 220, 230)
ROBOT_RADIUS: int = 6  # px

# Goal marker
GOAL_COLOR: tuple[int, int, int, int] = (50, 200, 80, 220)
GOAL_RADIUS: int = 5  # px

# hybrid mode Gaussian parameters
ROUTE_SIGMA_PX: float = 8.0
ROBOT_SIGMA_PX: float = 8.0
GOAL_SIGMA_PX: float = 5.0
GOAL_AMPLITUDE: float = 0.5  # goal peak relative to robot peak (1.0)


# ── rgb mode ───────────────────────────────────────────────────────────────────

def draw_route_rgb(
    draw: ImageDraw.ImageDraw,
    route_latlons: list[tuple[float, float]],
    to_pixel: Callable[[float, float], tuple[float, float]],
    current_idx: int,
) -> None:
    """Draw the future portion of the route on a PIL canvas.

    Only the segment from *current_idx* onward is rendered.
    Past portion is intentionally omitted — the robot marker conveys
    current position, and the model does not need trajectory history here.

    Args:
        draw:          PIL ImageDraw handle for the canvas.
        route_latlons: Full route [(lat, lon), ...].
        to_pixel:      Closure mapping (lat, lon) → (px, py) in canvas coords.
        current_idx:   Index of the route point nearest to the robot.
    """
    future = route_latlons[current_idx:]
    if len(future) < 2:
        return
    pts = [to_pixel(lat, lon) for lat, lon in future]
    draw.line(pts, fill=ROUTE_FUTURE_COLOR, width=ROUTE_LINE_WIDTH)


def draw_robot_marker_rgb(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
) -> None:
    """Draw a filled circle at the robot's position.

    Heading-up rendering makes direction arrows redundant, so a plain
    circle suffices.

    Args:
        draw: PIL ImageDraw handle.
        cx:   Circle centre x in canvas pixels.
        cy:   Circle centre y in canvas pixels.
    """
    r = ROBOT_RADIUS
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ROBOT_COLOR)


def draw_goal_marker_rgb(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
) -> None:
    """Draw a filled circle at the goal position.

    Args:
        draw: PIL ImageDraw handle.
        cx:   Circle centre x in canvas pixels.
        cy:   Circle centre y in canvas pixels.
    """
    r = GOAL_RADIUS
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=GOAL_COLOR)


# ── hybrid mode ────────────────────────────────────────────────────────────────

def _gaussian_blob(
    canvas_px: int,
    cx: float,
    cy: float,
    sigma: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Return a 2D Gaussian blob centred at (cx, cy).

    Args:
        canvas_px: Canvas side length (square).
        cx:        Centre x in pixels.
        cy:        Centre y in pixels.
        sigma:     Standard deviation in pixels.
        amplitude: Peak value (before clipping to [0, 1]).

    Returns:
        (canvas_px, canvas_px) float32 array, values in [0, amplitude].
    """
    ys, xs = np.mgrid[0:canvas_px, 0:canvas_px]
    blob = amplitude * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return blob.astype(np.float32)


def render_route_channel(
    canvas_px: int,
    route_latlons: list[tuple[float, float]],
    to_pixel: Callable[[float, float], tuple[float, float]],
    current_idx: int,
    sigma_px: float = ROUTE_SIGMA_PX,
) -> np.ndarray:
    """Render the future route as a Gaussian soft mask.

    Each route point from *current_idx* onward contributes a Gaussian blob.
    Overlapping blobs are summed and clipped to [0, 1].
    The Gaussian spread (σ ≈ 8 px at zoom 19) absorbs GPS position errors
    of ~3–5 m and makes the route clearly visible to the encoder.

    Args:
        canvas_px:     Canvas side length (must match the stitched canvas size).
        route_latlons: Full route [(lat, lon), ...].
        to_pixel:      Closure: (lat, lon) → (px, py) in canvas coordinates.
        current_idx:   Index of the route point nearest to the robot.
        sigma_px:      Gaussian σ in pixels.

    Returns:
        (canvas_px, canvas_px) float32 array in [0, 1].
    """
    channel = np.zeros((canvas_px, canvas_px), dtype=np.float32)
    future = route_latlons[current_idx:]
    for lat, lon in future:
        px, py = to_pixel(lat, lon)
        if 0 <= px < canvas_px and 0 <= py < canvas_px:
            channel += _gaussian_blob(canvas_px, px, py, sigma_px, amplitude=1.0)
    return np.clip(channel, 0.0, 1.0)


def render_position_goal_channel(
    canvas_px: int,
    robot_px: tuple[float, float],
    goal_px: tuple[float, float],
    robot_sigma_px: float = ROBOT_SIGMA_PX,
    goal_sigma_px: float = GOAL_SIGMA_PX,
    goal_amplitude: float = GOAL_AMPLITUDE,
) -> np.ndarray:
    """Render robot position and goal into a single channel.

    Robot and goal are distinguished by amplitude:
      - Robot:  Gaussian peak = 1.0
      - Goal:   Gaussian peak = GOAL_AMPLITUDE (0.5)

    Args:
        canvas_px:     Canvas side length in pixels.
        robot_px:      Robot position (x, y) in canvas pixels.
        goal_px:       Goal position (x, y) in canvas pixels.
        robot_sigma_px: Gaussian σ for the robot blob (pixels).
        goal_sigma_px:  Gaussian σ for the goal blob (pixels).
        goal_amplitude: Peak amplitude for the goal blob.

    Returns:
        (canvas_px, canvas_px) float32 array in [0, 1].
    """
    rx, ry = robot_px
    gx, gy = goal_px
    channel = _gaussian_blob(canvas_px, rx, ry, robot_sigma_px, amplitude=1.0)
    channel += _gaussian_blob(canvas_px, gx, gy, goal_sigma_px, amplitude=goal_amplitude)
    return np.clip(channel, 0.0, 1.0)
