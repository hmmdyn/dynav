"""OSRM API wrapper for pedestrian route planning and GPS map-matching.

Two workflows:
  - Dataset building: match(gps_trajectory) → snapped route on OSM network.
  - Inference:        route(start, goal)     → planned pedestrian route.

Both return routes as [(lat, lon), ...] on the OSM road network, ensuring
that map images look the same at training and inference time.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from typing import Optional


# ── Exceptions ─────────────────────────────────────────────────────────────────

class OSRMMatchError(Exception):
    """Raised when OSRM /match fails or confidence is too low."""


class OSRMRouteError(Exception):
    """Raised when OSRM /route fails or returns no route."""


# ── Geodesy helper ─────────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2.0 * R * math.asin(math.sqrt(a))


def _decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """Decode a Google-style encoded polyline to [(lat, lon), ...].

    OSRM uses precision=5 (default) in its route/match geometry.
    """
    coords: list[tuple[float, float]] = []
    index = 0
    lat = 0
    lng = 0
    while index < len(encoded):
        for is_lng in (False, True):
            result = 0
            shift = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if is_lng:
                lng += delta
                coords.append((lat / 1e5, lng / 1e5))
            else:
                lat += delta
    return coords


# ── OSRMRouter ─────────────────────────────────────────────────────────────────

class OSRMRouter:
    """Thin wrapper around the OSRM HTTP API.

    Args:
        base_url:   OSRM server base URL (no trailing slash).
        profile:    Routing profile — ``"foot"`` for pedestrian.
        timeout_s:  HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "https://router.project-osrm.org",
        profile: str = "foot",
        timeout_s: float = 10.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._profile = profile
        self._timeout = timeout_s

    # ------------------------------------------------------------------

    def _get(self, url: str) -> dict:
        req = urllib.request.Request(
            url, headers={"User-Agent": "dynav-research/1.0"}
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())

    # ------------------------------------------------------------------

    def match(
        self,
        latlons: list[tuple[float, float]],
        timestamps: Optional[list[int]] = None,
        radius_m: float = 25.0,
    ) -> tuple[list[tuple[float, float]], float]:
        """Snap a GPS trajectory to the OSM road network.

        Calls OSRM ``/match/v1/{profile}`` with the full trajectory.
        Call once per episode/segment — not per sample.

        Args:
            latlons:    GPS trajectory as [(lat, lon), ...].
            timestamps: Unix epoch timestamps (seconds) for each point.
                        Providing them improves match quality.
            radius_m:   Per-point snapping search radius in metres.

        Returns:
            matched_route: Snapped route [(lat, lon), ...] on OSM network.
            avg_deviation_m: Mean distance from input GPS to matched route.

        Raises:
            OSRMMatchError: OSRM returned a non-Ok code or confidence < 0.1.
            ValueError: ``latlons`` has fewer than 2 points.
        """
        if len(latlons) < 2:
            raise ValueError("match() requires at least 2 GPS points.")

        coords = ";".join(f"{lon},{lat}" for lat, lon in latlons)
        radiuses = ";".join(str(int(radius_m)) for _ in latlons)

        params: dict[str, str] = {
            "overview": "full",
            "geometries": "polyline",
            "annotations": "false",
            "radiuses": radiuses,
        }
        if timestamps:
            params["timestamps"] = ";".join(str(t) for t in timestamps)

        qs = urllib.parse.urlencode(params)
        url = f"{self._base}/match/v1/{self._profile}/{coords}?{qs}"

        data = self._get(url)
        code = data.get("code", "")
        if code != "Ok":
            raise OSRMMatchError(f"OSRM /match returned code={code!r}")

        matchings = data.get("matchings", [])
        if not matchings:
            raise OSRMMatchError("OSRM /match returned no matchings.")

        # Concatenate all matching legs into one route
        route: list[tuple[float, float]] = []
        for matching in matchings:
            confidence = matching.get("confidence", 1.0)
            if confidence < 0.1:
                raise OSRMMatchError(f"OSRM match confidence too low: {confidence:.3f}")
            segment = _decode_polyline(matching["geometry"])
            if route and segment:
                route.extend(segment[1:])  # avoid duplicating junction point
            else:
                route.extend(segment)

        # Average deviation: mean haversine distance from each GPS point to
        # its nearest point on the matched route.
        avg_dev = _avg_deviation(latlons, route)
        return route, avg_dev

    # ------------------------------------------------------------------

    def route(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> list[tuple[float, float]]:
        """Compute a pedestrian route from *start* to *goal*.

        Calls OSRM ``/route/v1/{profile}``. Cache the result at mission
        start — do not call per frame.

        Args:
            start: (lat, lon) of the starting position.
            goal:  (lat, lon) of the destination.

        Returns:
            route: [(lat, lon), ...] on the OSM road network.

        Raises:
            OSRMRouteError: OSRM returned a non-Ok code or no routes.
        """
        s_lat, s_lon = start
        g_lat, g_lon = goal
        coords = f"{s_lon},{s_lat};{g_lon},{g_lat}"
        params = urllib.parse.urlencode(
            {"overview": "full", "geometries": "polyline"}
        )
        url = f"{self._base}/route/v1/{self._profile}/{coords}?{params}"

        data = self._get(url)
        code = data.get("code", "")
        if code != "Ok":
            raise OSRMRouteError(f"OSRM /route returned code={code!r}")

        routes = data.get("routes", [])
        if not routes:
            raise OSRMRouteError("OSRM /route returned no routes.")

        return _decode_polyline(routes[0]["geometry"])

    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg) -> "OSRMRouter":
        """Construct from an OmegaConf ``cfg.map`` node."""
        return cls(
            base_url=cfg.map.get("osrm_url", "https://router.project-osrm.org"),
            profile=cfg.map.get("osrm_profile", "foot"),
        )


# ── Utility functions ──────────────────────────────────────────────────────────

def _avg_deviation(
    gps_pts: list[tuple[float, float]],
    route: list[tuple[float, float]],
) -> float:
    """Mean haversine distance from each GPS point to its nearest route point.

    Args:
        gps_pts: Original GPS trajectory [(lat, lon), ...].
        route:   Matched/planned route [(lat, lon), ...].

    Returns:
        Average deviation in metres.
    """
    if not route:
        return float("inf")
    total = 0.0
    for lat, lon in gps_pts:
        min_d = min(
            _haversine_m(lat, lon, rlat, rlon) for rlat, rlon in route
        )
        total += min_d
    return total / len(gps_pts)


def is_route_valid(
    matched_route: list[tuple[float, float]],
    gps_trajectory: list[tuple[float, float]],
    threshold_m: float = 10.0,
) -> bool:
    """Return False if the mean GPS-to-route deviation exceeds *threshold_m*.

    A large deviation means the GPS trajectory follows a path not represented
    in OSM (e.g. an indoor corridor, unpaved campus path), making the matched
    route unreliable for training.

    Args:
        matched_route:   Route returned by :meth:`OSRMRouter.match`.
        gps_trajectory:  Raw GPS trajectory used for matching.
        threshold_m:     Maximum allowed average deviation in metres.

    Returns:
        True if the route is usable, False if it should be discarded.
    """
    if not matched_route:
        return False
    dev = _avg_deviation(gps_trajectory, matched_route)
    return dev <= threshold_m


def find_current_idx(
    route_latlons: list[tuple[float, float]],
    current_latlon: tuple[float, float],
) -> int:
    """Return the index of the route point closest to *current_latlon*.

    Used to split the route into "past" (unused) and "future" (rendered)
    portions at each frame.

    Args:
        route_latlons:  Full route as [(lat, lon), ...].
        current_latlon: Robot's current GPS position (lat, lon).

    Returns:
        Index into *route_latlons* (0 if the route is empty).
    """
    if not route_latlons:
        return 0
    c_lat, c_lon = current_latlon
    best_idx = 0
    best_d = float("inf")
    for i, (rlat, rlon) in enumerate(route_latlons):
        d = _haversine_m(c_lat, c_lon, rlat, rlon)
        if d < best_d:
            best_d = d
            best_idx = i
    return best_idx
