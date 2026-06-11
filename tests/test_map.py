"""Unit tests for dynav/map/ package."""

from __future__ import annotations

import math
import io
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from dynav.map.tiles import (
    TILE_SIZE,
    tile_coord,
    latlon_to_global_pixel,
    TileCache,
    stitch_tiles,
)
from dynav.map.routing import (
    OSRMMatchError,
    OSRMRouter,
    _decode_polyline,
    _avg_deviation,
    is_route_valid,
    find_current_idx,
)
from dynav.map.segment import segment_gps_episode
from dynav.map.overlay import (
    ROBOT_RADIUS,
    GOAL_RADIUS,
    ROUTE_FUTURE_COLOR,
    render_route_channel,
    render_position_goal_channel,
)
from dynav.map.renderer import MapRenderer


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _grey_tile_png() -> bytes:
    img = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mock_cache(tmp_path) -> TileCache:
    """TileCache backed by tmp_path; no network calls."""
    cache = TileCache(str(tmp_path / "tiles"))
    # Pre-populate one tile so stitch_tiles doesn't hit the network
    z, x, y = 19, 0, 0
    path = cache._cache_path(z, x, y)
    path.write_bytes(_grey_tile_png())
    return cache


def _renderer(tmp_path, mode: str = "rgb") -> MapRenderer:
    cache = TileCache(str(tmp_path / "tiles"))
    # Patch TileCache.get to always return a grey tile
    cache.get = lambda z, x, y: Image.new("RGB", (TILE_SIZE, TILE_SIZE), (180, 180, 180))
    return MapRenderer(cache, zoom=19, render_size=256, output_size=64, mode=mode)


SAMPLE_ROUTE = [
    (37.288 + i * 0.0001, 126.976) for i in range(30)
]
SAMPLE_GPS = [
    (37.288 + i * 0.0001 + 0.00002, 126.976) for i in range(30)
]


# ── TileCache ──────────────────────────────────────────────────────────────────

class TestTileCache:
    def test_cache_hit_avoids_network(self, tmp_path):
        cache = TileCache(str(tmp_path / "tiles"))
        tile_path = cache._cache_path(19, 5, 3)
        tile_path.write_bytes(_grey_tile_png())

        with patch("urllib.request.urlopen") as mock_open:
            img = cache.get(19, 5, 3)
            mock_open.assert_not_called()
        assert img.size == (TILE_SIZE, TILE_SIZE)

    def test_network_called_on_miss(self, tmp_path):
        cache = TileCache(str(tmp_path / "tiles"))
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = _grey_tile_png()

        with patch("urllib.request.urlopen", return_value=mock_resp):
            img = cache.get(19, 5, 3)
        assert img.size == (TILE_SIZE, TILE_SIZE)

    def test_fallback_grey_on_error(self, tmp_path):
        cache = TileCache(str(tmp_path / "tiles"), max_retries=1, retry_delay=0)
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            img = cache.get(19, 99, 99)
        assert img.size == (TILE_SIZE, TILE_SIZE)
        px = img.getpixel((0, 0))
        assert px == (200, 200, 200)


# ── Tile coordinates ───────────────────────────────────────────────────────────

class TestTileCoord:
    def test_known_coord(self):
        # London (~51.5°N, 0°W), zoom 19
        tx, ty = tile_coord(51.5, 0.0, 19)
        assert isinstance(tx, int) and isinstance(ty, int)
        assert tx >= 0 and ty >= 0

    def test_global_pixel_increases_east(self):
        px1, _ = latlon_to_global_pixel(37.0, 126.0, 19)
        px2, _ = latlon_to_global_pixel(37.0, 127.0, 19)
        assert px2 > px1

    def test_global_pixel_increases_south(self):
        _, py1 = latlon_to_global_pixel(38.0, 126.0, 19)
        _, py2 = latlon_to_global_pixel(37.0, 126.0, 19)
        assert py2 > py1  # lower latitude → larger y


# ── OSRMRouter ─────────────────────────────────────────────────────────────────

def _osrm_match_response() -> bytes:
    import json
    # Minimal valid OSRM /match response with one matching
    encoded_poly = ""  # empty — decode_polyline returns []
    # Build a tiny 2-point polyline manually: (37.288, 126.976) (37.289, 126.976)
    # Encode as Google polyline (precision 5)
    def encode_val(v: int) -> str:
        v = ~(v << 1) if v < 0 else v << 1
        chunks = []
        while v >= 0x20:
            chunks.append(chr((0x20 | (v & 0x1F)) + 63))
            v >>= 5
        chunks.append(chr(v + 63))
        return "".join(chunks)

    def encode_polyline(pts):
        out = []
        prev_lat = prev_lon = 0
        for lat, lon in pts:
            dlat = round(lat * 1e5) - prev_lat
            dlon = round(lon * 1e5) - prev_lon
            prev_lat += dlat
            prev_lon += dlon
            out.append(encode_val(dlat) + encode_val(dlon))
        return "".join(out)

    poly = encode_polyline([(37.288, 126.976), (37.289, 126.976)])
    payload = {
        "code": "Ok",
        "matchings": [{"confidence": 0.9, "geometry": poly}],
    }
    return json.dumps(payload).encode()


def _osrm_route_response() -> bytes:
    import json

    def encode_val(v: int) -> str:
        v = ~(v << 1) if v < 0 else v << 1
        chunks = []
        while v >= 0x20:
            chunks.append(chr((0x20 | (v & 0x1F)) + 63))
            v >>= 5
        chunks.append(chr(v + 63))
        return "".join(chunks)

    def encode_polyline(pts):
        out = []
        prev_lat = prev_lon = 0
        for lat, lon in pts:
            dlat = round(lat * 1e5) - prev_lat
            dlon = round(lon * 1e5) - prev_lon
            prev_lat += dlat
            prev_lon += dlon
            out.append(encode_val(dlat) + encode_val(dlon))
        return "".join(out)

    poly = encode_polyline([(37.288, 126.976), (37.290, 126.977)])
    payload = {
        "code": "Ok",
        "routes": [{"geometry": poly}],
    }
    return json.dumps(payload).encode()


class TestOSRMRouter:
    def _mock_urlopen(self, data: bytes):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = data
        return patch("urllib.request.urlopen", return_value=mock_resp)

    def test_match_returns_route_and_deviation(self):
        router = OSRMRouter()
        with self._mock_urlopen(_osrm_match_response()):
            route, dev = router.match(SAMPLE_GPS[:5])
        assert len(route) >= 2
        assert isinstance(dev, float) and dev >= 0.0

    def test_match_raises_on_failure(self):
        import json
        bad = json.dumps({"code": "NoRoute"}).encode()
        router = OSRMRouter()
        with self._mock_urlopen(bad):
            with pytest.raises(OSRMMatchError):
                router.match(SAMPLE_GPS[:3])

    def test_route_returns_waypoints(self):
        router = OSRMRouter()
        with self._mock_urlopen(_osrm_route_response()):
            route = router.route((37.288, 126.976), (37.290, 126.977))
        assert len(route) >= 2

    def test_is_route_valid_pass(self):
        matched = [(37.288 + i * 0.0001, 126.976) for i in range(5)]
        gps     = [(37.288 + i * 0.0001 + 0.00002, 126.976) for i in range(5)]
        assert is_route_valid(matched, gps, threshold_m=10.0)

    def test_is_route_valid_fail(self):
        matched = [(37.288, 126.976), (37.289, 126.976)]
        gps     = [(37.288, 127.000), (37.289, 127.000)]  # far away
        assert not is_route_valid(matched, gps, threshold_m=10.0)

    def test_find_current_idx(self):
        """Returns the index of the *closest* route point (docstring contract)."""
        route = [(37.288 + i * 0.0001, 126.976) for i in range(10)]
        idx = find_current_idx(route, (37.2885, 126.976))
        assert idx == 5

    def test_find_current_idx_empty(self):
        assert find_current_idx([], (37.0, 126.0)) == 0


# ── Segment ────────────────────────────────────────────────────────────────────

class TestSegment:
    def _straight(self, n: int = 50, dt: float = 1.0):
        """Clean straight trajectory at walking speed (~1 m/step at 1 Hz)."""
        latlons = [(37.288 + i * 0.00001, 126.976) for i in range(n)]
        timestamps = [float(i) * dt for i in range(n)]
        return latlons, timestamps

    def test_clean_trajectory_one_segment(self):
        latlons, ts = self._straight(50)
        segs = segment_gps_episode(latlons, ts, min_length_m=5.0)
        assert len(segs) == 1
        assert len(segs[0]) == 50

    def test_splits_on_jump(self):
        latlons, ts = self._straight(20)
        # Insert a 100m jump at index 10
        latlons[10] = (37.288 + 0.001, 126.976)
        segs = segment_gps_episode(latlons, ts, jump_threshold_m=5.0, min_length_m=5.0)
        assert len(segs) >= 2

    def test_removes_long_stationary_block(self):
        latlons, ts = self._straight(60)
        # Replace indices 20–40 with stationary points (no movement)
        for i in range(20, 41):
            latlons[i] = latlons[20]
        segs = segment_gps_episode(
            latlons, ts,
            stationary_speed_mps=0.3,
            stationary_window_s=5.0,
            min_length_m=5.0,
        )
        # Should split into two segments (before and after stationary block)
        assert len(segs) >= 1
        # Neither segment should contain the stationary block interior
        for seg in segs:
            assert len(seg) < 40

    def test_discards_short_segment(self):
        latlons = [(37.288, 126.976), (37.2880001, 126.976)]  # ~1 cm
        ts = [0.0, 1.0]
        segs = segment_gps_episode(latlons, ts, min_length_m=10.0)
        assert segs == []

    def test_ends_on_loop(self):
        # Robot walks 30 steps north (~1.1 m/step, below the 5 m jump
        # threshold), leaves the 15 m loop radius, then returns to start
        n = 30
        latlons = [(37.288 + i * 0.00001, 126.976) for i in range(n)]
        # Add return leg
        latlons += [(37.288 + (n - i) * 0.00001, 126.976) for i in range(1, n + 1)]
        ts = [float(i) for i in range(len(latlons))]
        segs = segment_gps_episode(latlons, ts, loop_radius_m=15.0, min_length_m=5.0)
        # The loop back should trigger a segment break
        assert len(segs) >= 2

    def test_returns_empty_on_too_few_points(self):
        segs = segment_gps_episode([(37.0, 126.0)], [0.0])
        assert segs == []


# ── MapRenderer ────────────────────────────────────────────────────────────────

class TestMapRenderer:
    def test_output_size_rgb(self, tmp_path):
        r = _renderer(tmp_path, mode="rgb")
        img = r.render(37.2890, 126.9760, 0.0, SAMPLE_ROUTE, 37.2920, 126.9780)
        assert img is not None
        assert img.size == (64, 64)

    def test_output_size_hybrid(self, tmp_path):
        r = _renderer(tmp_path, mode="hybrid")
        img = r.render(37.2890, 126.9760, 0.0, SAMPLE_ROUTE, 37.2920, 126.9780)
        assert img is not None
        assert img.size == (64, 64)

    def test_returns_none_on_empty_route(self, tmp_path):
        r = _renderer(tmp_path, mode="rgb")
        img = r.render(37.289, 126.976, 0.0, [], 37.292, 126.978)
        assert img is None

    def test_rgb_mode_is_pil_image(self, tmp_path):
        r = _renderer(tmp_path, mode="rgb")
        img = r.render(37.2890, 126.9760, 45.0, SAMPLE_ROUTE, 37.2920, 126.9780)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_hybrid_mode_is_pil_image(self, tmp_path):
        r = _renderer(tmp_path, mode="hybrid")
        img = r.render(37.2890, 126.9760, 45.0, SAMPLE_ROUTE, 37.2920, 126.9780)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_hybrid_value_range(self, tmp_path):
        r = _renderer(tmp_path, mode="hybrid")
        img = r.render(37.2890, 126.9760, 0.0, SAMPLE_ROUTE, 37.2920, 126.9780)
        arr = np.array(img)
        assert arr.min() >= 0 and arr.max() <= 255

    def test_heading_north_same_as_zero(self, tmp_path):
        r = _renderer(tmp_path, mode="rgb")
        img0 = r.render(37.2890, 126.9760, 0.0, SAMPLE_ROUTE, 37.2920, 126.9780)
        img360 = r.render(37.2890, 126.9760, 360.0, SAMPLE_ROUTE, 37.2920, 126.9780)
        arr0 = np.array(img0).astype(float)
        arr360 = np.array(img360).astype(float)
        # 0° and 360° should produce nearly identical images
        assert np.mean(np.abs(arr0 - arr360)) < 5.0

    def test_different_headings_differ(self, tmp_path):
        r = _renderer(tmp_path, mode="rgb")
        img0   = r.render(37.2890, 126.9760,  0.0, SAMPLE_ROUTE, 37.2920, 126.9780)
        img90  = r.render(37.2890, 126.9760, 90.0, SAMPLE_ROUTE, 37.2920, 126.9780)
        arr0  = np.array(img0).astype(float)
        arr90 = np.array(img90).astype(float)
        assert np.mean(np.abs(arr0 - arr90)) > 1.0

    def test_invalid_mode_raises(self, tmp_path):
        cache = TileCache(str(tmp_path / "tiles"))
        with pytest.raises(ValueError):
            MapRenderer(cache, mode="invalid")


# ── Overlay channels ───────────────────────────────────────────────────────────

class TestOverlay:
    def test_route_channel_nonzero_for_future(self):
        cpx = 128
        route = [(0.0, float(i)) for i in range(10)]  # dummy latlon
        def to_pixel(lat, lon):
            return (lon * 10 + 10, 64.0)  # horizontal line
        ch = render_route_channel(cpx, route, to_pixel, current_idx=0)
        assert ch.max() > 0.0
        assert ch.dtype == np.float32
        assert ch.shape == (cpx, cpx)

    def test_route_channel_zero_before_current_idx(self):
        # With current_idx = all points, the future is empty → channel = 0
        cpx = 64
        route = [(0.0, float(i)) for i in range(5)]
        def to_pixel(lat, lon):
            return (lon * 5 + 10, 32.0)
        ch = render_route_channel(cpx, route, to_pixel, current_idx=len(route))
        assert ch.max() == 0.0

    def test_position_goal_channel_robot_peak(self):
        cpx = 64
        ch = render_position_goal_channel(cpx, (32.0, 32.0), (10.0, 10.0))
        assert ch.dtype == np.float32
        # Robot at centre should have the maximum value ≈ 1.0
        assert ch[32, 32] > 0.9

    def test_position_goal_channel_goal_lower_amplitude(self):
        cpx = 128
        ch = render_position_goal_channel(
            cpx, (64.0, 64.0), (20.0, 20.0),
            robot_sigma_px=4.0, goal_sigma_px=4.0
        )
        robot_val = ch[64, 64]
        goal_val  = ch[20, 20]
        # Goal peak should be about half the robot peak
        assert goal_val < robot_val

    def test_channel_values_in_range(self):
        cpx = 64
        ch = render_position_goal_channel(cpx, (32.0, 32.0), (10.0, 10.0))
        assert ch.min() >= 0.0 and ch.max() <= 1.0
