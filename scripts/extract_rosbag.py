"""Extract dynav training data from ROS2/ROS1 rosbag files.

Reads a rosbag, extracts synchronised camera/GPS/IMU messages, renders
heading-up OSM map images via dynav.map, and writes samples in the
DyNavDataset format:

    data/{split}/sample_XXXXXX/
        obs_0.png    # front camera, current frame
        obs_1.png    # front camera, 1 Δt ago
        obs_2.png    # front camera, 2 Δt ago
        obs_3.png    # rear camera, current frame
        map.png      # heading-up OSM map + OSRM route overlay, 224×224
        meta.json    # gt_waypoints, route_direction

Route generation:
    GPS trajectory → OSRM /match → snapped route on OSM network.
    Episodes where the mean GPS↔route deviation > 10 m are discarded
    (the path is not in OSM and cannot be matched reliably).

Usage::

    python scripts/extract_rosbag.py \\
        --bag path/to/recording.db3 \\
        --output data/ \\
        --split train \\
        [--goal-lat 37.56 --goal-lon 126.94] \\
        [--config configs/rosbag_topics.yaml] \\
        [--ros-version 2]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from dynav.map import MapRenderer, OSRMRouter, is_route_valid  # noqa: E402
from dynav.map.routing import OSRMMatchError  # noqa: E402
from dynav.utils.geometry import compute_route_direction  # noqa: E402

# ---------------------------------------------------------------------------
# Optional progress bar
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, total: int, desc: str = ""):
        return _tqdm(iterable, total=total, desc=desc)

except ImportError:
    class _FallbackProgress:
        def __init__(self, iterable, total: int, desc: str = ""):
            self._it = iterable
            self._total = total
            self._desc = desc
            self._n = 0

        def __iter__(self):
            for item in self._it:
                yield item
                self._n += 1
                if self._n % 50 == 0:
                    print(f"[{self._desc}] {self._n}/{self._total}")

        def update(self, n: int = 1):
            pass

        def close(self):
            pass

    def _progress(iterable, total: int, desc: str = ""):  # type: ignore[misc]
        return _FallbackProgress(iterable, total, desc)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NS_PER_S: int = 1_000_000_000
EARTH_RADIUS_M: float = 6_371_000.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class SyncedFrame:
    """All data needed to produce one training sample.

    Attributes:
        ts_ns:        Timestamp of the *current* front-camera frame (ns).
        front_img:    Current front-camera PIL image.
        front_past1:  Front camera 1 × past_dt ago.
        front_past2:  Front camera 2 × past_dt ago.
        rear_img:     Current rear-camera PIL image.
        lat:          GPS latitude at ts_ns.
        lon:          GPS longitude at ts_ns.
        heading_deg:  Compass heading, 0 = north, clockwise.
    """

    ts_ns: int
    front_img: Image.Image
    front_past1: Image.Image
    front_past2: Image.Image
    rear_img: Image.Image
    lat: float
    lon: float
    heading_deg: float


# ---------------------------------------------------------------------------
# Geodesy helpers
# ---------------------------------------------------------------------------
def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in metres."""
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return initial bearing (compass degrees) from point 1 to point 2."""
    d_lon = math.radians(lon2 - lon1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    x = math.sin(d_lon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _latlon_to_body(
    lat: float,
    lon: float,
    ref_lat: float,
    ref_lon: float,
    heading_deg: float,
) -> Tuple[float, float]:
    """Convert lat/lon to robot body frame (x=forward, y=left) in metres."""
    d_lat = math.radians(lat - ref_lat)
    d_lon = math.radians(lon - ref_lon)
    north_m = d_lat * EARTH_RADIUS_M
    east_m = d_lon * EARTH_RADIUS_M * math.cos(math.radians(ref_lat))
    heading_r = math.radians(heading_deg)
    x_fwd  = north_m * math.cos(heading_r) + east_m * math.sin(heading_r)
    y_left = -north_m * math.sin(heading_r) + east_m * math.cos(heading_r)
    return x_fwd, y_left


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------
def _decode_image_msg(msg) -> Image.Image:
    """Decode sensor_msgs/Image to PIL RGB."""
    encoding = msg.encoding.lower()
    h, w = int(msg.height), int(msg.width)
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if encoding in ("rgb8",):
        return Image.fromarray(data.reshape(h, w, 3), "RGB")
    elif encoding in ("bgr8",):
        arr = data.reshape(h, w, 3)
        return Image.fromarray(arr[:, :, ::-1].copy(), "RGB")
    elif encoding == "mono8":
        arr = data.reshape(h, w)
        return Image.fromarray(np.stack([arr] * 3, axis=-1), "RGB")
    elif encoding == "mono16":
        arr = (data.view(np.uint16).reshape(h, w) >> 8).astype(np.uint8)
        return Image.fromarray(np.stack([arr] * 3, axis=-1), "RGB")
    else:
        arr = data.reshape(h, w, 3)
        return Image.fromarray(arr[:, :, ::-1].copy(), "RGB")


def _decode_compressed_msg(msg) -> Image.Image:
    """Decode sensor_msgs/CompressedImage to PIL RGB."""
    import cv2
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode returned None — corrupt compressed image?")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "RGB")


def _is_compressed_topic(topic: str) -> bool:
    return topic.endswith("/compressed")


def _decode_any_image(msg, topic: str) -> Image.Image:
    if _is_compressed_topic(topic):
        return _decode_compressed_msg(msg)
    return _decode_image_msg(msg)


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------
class BagExtractor:
    """Extracts synchronised training samples from a ROS rosbag.

    Args:
        bag_path:    Path to the .db3 (ROS2) or .bag (ROS1) file.
        cfg:         OmegaConf DictConfig from ``rosbag_topics.yaml`` merged
                     with ``map.yaml``.
        goal_lat:    Optional goal latitude; defaults to last GPS point.
        goal_lon:    Optional goal longitude; defaults to last GPS point.
        ros_version: 1 or 2; auto-detected from extension if None.
    """

    def __init__(
        self,
        bag_path: Path,
        cfg,
        goal_lat: Optional[float] = None,
        goal_lon: Optional[float] = None,
        ros_version: Optional[int] = None,
    ) -> None:
        self.bag_path = bag_path
        self.cfg = cfg
        self.goal_lat = goal_lat
        self.goal_lon = goal_lon
        self._ros_version = ros_version if ros_version is not None else self._detect_ros_version(bag_path)

        self._renderer = MapRenderer.from_config(cfg)
        self._router   = OSRMRouter.from_config(cfg)

    # ------------------------------------------------------------------
    # Static / class helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_ros_version(path: Path) -> int:
        suffix = path.suffix.lower()
        if suffix == ".bag":
            return 1
        return 2

    # ------------------------------------------------------------------
    # Message reading
    # ------------------------------------------------------------------

    def _read_messages(
        self, bag_path: Path, ros_version: int
    ) -> Dict[str, List[Tuple[int, object]]]:
        """Read all messages from bag, grouped by topic."""
        cfg = self.cfg

        dual_topic: Optional[str] = cfg.topics.get("dual_camera")
        if dual_topic:
            camera_topics: set = {dual_topic}
        else:
            camera_topics = {cfg.topics.front_camera, cfg.topics.rear_camera}

        topics_of_interest = camera_topics | {cfg.topics.gps, cfg.topics.imu}

        odom_topic: Optional[str] = cfg.topics.get("odom")
        if odom_topic:
            topics_of_interest.add(odom_topic)

        gps_heading_topic: Optional[str] = cfg.topics.get("gps_heading")
        if gps_heading_topic:
            topics_of_interest.add(gps_heading_topic)

        messages: Dict[str, List[Tuple[int, object]]] = {t: [] for t in topics_of_interest}

        if ros_version == 2:
            from rosbags.rosbag2 import Reader
            from rosbags.typesys import Stores, get_typestore

            typestore = get_typestore(Stores.ROS2_HUMBLE)
            with Reader(str(bag_path)) as reader:
                connections = [c for c in reader.connections if c.topic in topics_of_interest]
                total = sum(c.msgcount for c in connections)
                pbar = _progress(reader.messages(connections=connections), total=total, desc="reading bag")
                for connection, timestamp, rawdata in pbar:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    messages[connection.topic].append((int(timestamp), msg))
        else:
            from rosbags.rosbag1 import Reader as Reader1
            from rosbags.typesys import Stores, get_typestore

            typestore = get_typestore(Stores.ROS1_NOETIC)
            with Reader1(str(bag_path)) as reader:
                connections = [c for c in reader.connections.values() if c.topic in topics_of_interest]
                total = sum(c.msgcount for c in connections)
                pbar = _progress(reader.messages(connections=connections), total=total, desc="reading bag")
                for connection, timestamp, rawdata in pbar:
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                    messages[connection.topic].append((int(timestamp), msg))

        for topic in messages:
            messages[topic].sort(key=lambda x: x[0])

        return messages

    # ------------------------------------------------------------------
    # GPS trajectory
    # ------------------------------------------------------------------

    def _build_gps_trajectory(
        self, gps_msgs: List[Tuple[int, object]]
    ) -> List[Tuple[int, float, float]]:
        """Build (ts_ns, lat, lon) list, filtering invalid fixes."""
        trajectory: List[Tuple[int, float, float]] = []
        for ts_ns, msg in gps_msgs:
            if hasattr(msg, "status") and hasattr(msg.status, "status"):
                if msg.status.status < 0:
                    continue
            lat, lon = float(msg.latitude), float(msg.longitude)
            if math.isnan(lat) or math.isnan(lon):
                continue
            trajectory.append((ts_ns, lat, lon))
        return trajectory

    # ------------------------------------------------------------------
    # Heading
    # ------------------------------------------------------------------

    @staticmethod
    def _split_dual_fisheye(img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        w, h = img.size
        mid = w // 2
        return img.crop((0, 0, mid, h)), img.crop((mid, 0, w, h))

    def _get_heading(
        self,
        ts_ns: int,
        imu_msgs: List[Tuple[int, object]],
        gps_traj: List[Tuple[int, float, float]],
        source: str,
        odom_msgs: Optional[List[Tuple[int, object]]] = None,
        gps_heading_msgs: Optional[List[Tuple[int, object]]] = None,
    ) -> Optional[float]:
        """Return compass heading (degrees, 0=North, CW) at *ts_ns*."""
        if source == "gps_heading":
            if not gps_heading_msgs:
                return None
            tol_ns = int(self.cfg.sync.imu_tol_s * NS_PER_S)
            msg = self._find_nearest(gps_heading_msgs, ts_ns, tol_ns)
            if msg is None:
                return None
            q = msg.quaternion
            yaw_rad = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            return (90.0 - math.degrees(yaw_rad)) % 360.0

        if source == "imu":
            tol_ns = int(self.cfg.sync.imu_tol_s * NS_PER_S)
            msg = self._find_nearest(imu_msgs, ts_ns, tol_ns)
            if msg is None:
                return None
            q = msg.orientation
            yaw_rad = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            return (90.0 - math.degrees(yaw_rad)) % 360.0

        elif source == "gps":
            idx = self._find_nearest_index(gps_traj, ts_ns)
            if idx is None or idx + 1 >= len(gps_traj):
                return None
            _, lat0, lon0 = gps_traj[idx]
            _, lat1, lon1 = gps_traj[idx + 1]
            if _haversine_m(lat0, lon0, lat1, lon1) < 0.1:
                return None
            return _bearing_deg(lat0, lon0, lat1, lon1)

        elif source == "odom":
            if not odom_msgs:
                return None
            tol_ns = int(self.cfg.sync.imu_tol_s * NS_PER_S)
            msg = self._find_nearest(odom_msgs, ts_ns, tol_ns)
            if msg is None:
                return None
            q = msg.pose.pose.orientation
            yaw_rad = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            return (90.0 - math.degrees(yaw_rad)) % 360.0

        return None

    # ------------------------------------------------------------------
    # Time-nearest helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_nearest(
        msgs: List[Tuple[int, object]],
        ts_ns: int,
        tol_ns: int,
    ) -> Optional[object]:
        if not msgs:
            return None
        lo, hi = 0, len(msgs) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if msgs[mid][0] < ts_ns:
                lo = mid + 1
            else:
                hi = mid
        best_idx = lo
        best_diff = abs(msgs[lo][0] - ts_ns)
        if lo > 0:
            d = abs(msgs[lo - 1][0] - ts_ns)
            if d < best_diff:
                best_idx = lo - 1
                best_diff = d
        if best_diff > tol_ns:
            return None
        return msgs[best_idx][1]

    @staticmethod
    def _find_nearest_index(
        traj: List[Tuple[int, float, float]],
        ts_ns: int,
    ) -> Optional[int]:
        if not traj:
            return None
        lo, hi = 0, len(traj) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if traj[mid][0] < ts_ns:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0 and abs(traj[lo - 1][0] - ts_ns) < abs(traj[lo][0] - ts_ns):
            return lo - 1
        return lo

    # ------------------------------------------------------------------
    # Frame synchronisation
    # ------------------------------------------------------------------

    def _build_synced_frames(
        self,
        messages: Dict[str, List[Tuple[int, object]]],
        cfg,
    ) -> List[SyncedFrame]:
        """Synchronise front/rear camera, GPS, and IMU into SyncedFrames."""
        gps_topic = cfg.topics.gps
        imu_topic = cfg.topics.imu
        odom_topic: Optional[str] = cfg.topics.get("odom")
        gps_heading_topic: Optional[str] = cfg.topics.get("gps_heading")
        dual_topic: Optional[str] = cfg.topics.get("dual_camera")

        if dual_topic:
            dual_msgs = messages[dual_topic]
            front_msgs = dual_msgs
            rear_msgs = dual_msgs
            front_topic = dual_topic
            rear_topic = dual_topic
            _dual_mode = True
        else:
            front_topic = cfg.topics.front_camera
            rear_topic = cfg.topics.rear_camera
            front_msgs = messages[front_topic]
            rear_msgs = messages[rear_topic]
            _dual_mode = False

        gps_msgs = messages[gps_topic]
        imu_msgs = messages[imu_topic]
        odom_msgs = messages.get(odom_topic, []) if odom_topic else []
        gps_heading_msgs = messages.get(gps_heading_topic, []) if gps_heading_topic else []

        gps_traj = self._build_gps_trajectory(gps_msgs)

        interval_ns = int(NS_PER_S / cfg.sync.camera_hz)
        past_dt_ns = int(cfg.sync.past_dt_s * NS_PER_S)
        gps_tol_ns = int(cfg.sync.gps_tol_s * NS_PER_S)
        imu_tol_ns = int(cfg.sync.imu_tol_s * NS_PER_S)
        min_speed = cfg.sync.min_speed_mps

        frames: List[SyncedFrame] = []
        last_emitted_ts: Optional[int] = None

        for i, (ts_ns, _msg) in enumerate(front_msgs):
            if last_emitted_ts is not None and (ts_ns - last_emitted_ts) < interval_ns:
                continue

            gps_msg = self._find_nearest(gps_msgs, ts_ns, gps_tol_ns)
            if gps_msg is None:
                continue
            lat = float(gps_msg.latitude)
            lon = float(gps_msg.longitude)
            if math.isnan(lat) or math.isnan(lon):
                continue

            speed: Optional[float] = None
            if odom_msgs:
                odom_msg = self._find_nearest(odom_msgs, ts_ns, imu_tol_ns * 5)
                if odom_msg is not None:
                    speed = abs(float(odom_msg.twist.twist.linear.x))
            if speed is None:
                traj_idx = self._find_nearest_index(gps_traj, ts_ns)
                if traj_idx is not None and traj_idx > 0:
                    prev_ts, prev_lat, prev_lon = gps_traj[traj_idx - 1]
                    cur_ts, cur_lat, cur_lon = gps_traj[traj_idx]
                    dt_s = (cur_ts - prev_ts) / NS_PER_S
                    if dt_s > 0:
                        speed = _haversine_m(prev_lat, prev_lon, cur_lat, cur_lon) / dt_s
            if speed is not None and speed < min_speed:
                continue

            heading_deg = self._get_heading(
                ts_ns, imu_msgs, gps_traj, cfg.sync.heading_source,
                odom_msgs, gps_heading_msgs=gps_heading_msgs,
            )
            if heading_deg is None:
                continue

            rear_msg = self._find_nearest(rear_msgs, ts_ns, tol_ns=imu_tol_ns * 5)
            if rear_msg is None:
                continue

            front_past1_msg = self._find_nearest(front_msgs, ts_ns - past_dt_ns, tol_ns=past_dt_ns // 2)
            front_past2_msg = self._find_nearest(front_msgs, ts_ns - 2 * past_dt_ns, tol_ns=past_dt_ns // 2)
            if front_past1_msg is None or front_past2_msg is None:
                continue

            try:
                if _dual_mode:
                    dual_current = _decode_any_image(_msg, front_topic)
                    front_img, rear_img = self._split_dual_fisheye(dual_current)
                    dual_past1 = _decode_any_image(front_past1_msg, front_topic)
                    front_past1, _ = self._split_dual_fisheye(dual_past1)
                    dual_past2 = _decode_any_image(front_past2_msg, front_topic)
                    front_past2, _ = self._split_dual_fisheye(dual_past2)
                else:
                    front_img = _decode_any_image(_msg, front_topic)
                    rear_img = _decode_any_image(rear_msg, rear_topic)
                    front_past1 = _decode_any_image(front_past1_msg, front_topic)
                    front_past2 = _decode_any_image(front_past2_msg, front_topic)
            except Exception as exc:
                print(f"[warn] image decode failed at ts={ts_ns}: {exc}")
                continue

            frames.append(SyncedFrame(
                ts_ns=ts_ns,
                front_img=front_img,
                front_past1=front_past1,
                front_past2=front_past2,
                rear_img=rear_img,
                lat=lat,
                lon=lon,
                heading_deg=heading_deg,
            ))
            last_emitted_ts = ts_ns

        return frames

    # ------------------------------------------------------------------
    # GT waypoints
    # ------------------------------------------------------------------

    def _gt_waypoints(
        self,
        gps_traj: List[Tuple[int, float, float]],
        matched_route: List[Tuple[float, float]],
        current_ts_ns: int,
        current_lat: float,
        current_lon: float,
        heading_deg: float,
        cfg,
    ) -> Tuple[List[List[float]], float]:
        """Compute ground-truth waypoints and route direction for one frame.

        Args:
            gps_traj:      Full GPS trajectory (ts_ns, lat, lon).
            matched_route: OSRM-matched route [(lat, lon), ...].
            current_ts_ns: Timestamp of the current frame.
            current_lat:   GPS latitude at the current frame.
            current_lon:   GPS longitude at the current frame.
            heading_deg:   Robot compass heading (degrees, 0=North CW).
            cfg:           OmegaConf config.

        Returns:
            (waypoints, route_direction_rad) where:
            - waypoints: List of [x_norm, y_norm] pairs, length H.
            - route_direction_rad: bearing to lookahead point in body frame
              radians (positive = left of forward).
        """
        horizon: int = cfg.data.horizon
        wp_spacing_m: float = cfg.data.wp_spacing_m
        max_dist_m: float = cfg.data.max_wp_dist_m

        start_idx = self._find_nearest_index(gps_traj, current_ts_ns)
        if start_idx is None:
            return [[0.0, 0.0]] * horizon, 0.0

        waypoints: List[List[float]] = []
        target_dist_m = wp_spacing_m
        search_idx = start_idx
        accumulated_m = 0.0
        prev_lat, prev_lon = current_lat, current_lon

        while len(waypoints) < horizon and search_idx + 1 < len(gps_traj):
            search_idx += 1
            _, next_lat, next_lon = gps_traj[search_idx]
            step_m = _haversine_m(prev_lat, prev_lon, next_lat, next_lon)
            accumulated_m += step_m
            prev_lat, prev_lon = next_lat, next_lon

            while accumulated_m >= target_dist_m and len(waypoints) < horizon:
                overshoot = accumulated_m - target_dist_m
                frac = 1.0 - overshoot / step_m if step_m > 0 else 1.0
                wp_lat = gps_traj[search_idx - 1][1] + frac * (next_lat - gps_traj[search_idx - 1][1])
                wp_lon = gps_traj[search_idx - 1][2] + frac * (next_lon - gps_traj[search_idx - 1][2])
                x_m, y_m = _latlon_to_body(wp_lat, wp_lon, current_lat, current_lon, heading_deg)
                x_norm = max(-1.0, min(1.0, x_m / max_dist_m))
                y_norm = max(-1.0, min(1.0, y_m / max_dist_m))
                waypoints.append([x_norm, y_norm])
                target_dist_m += wp_spacing_m

        while len(waypoints) < horizon:
            waypoints.append(waypoints[-1] if waypoints else [0.0, 0.0])

        # route_direction: radians in robot body frame (positive = left)
        # Using compute_route_direction() with OSRM matched route for consistency.
        enu_heading = math.radians(90.0 - heading_deg)  # compass CW → ENU CCW
        route_direction_rad = compute_route_direction(
            np.array([current_lat, current_lon]),
            np.array(matched_route),
            enu_heading,
            lookahead_distance=10.0,
        )

        return waypoints, float(route_direction_rad)

    # ------------------------------------------------------------------
    # Main extraction entry point
    # ------------------------------------------------------------------

    def extract(
        self,
        output_dir: Path,
        split: str = "train",
        start_idx: int = 0,
    ) -> int:
        """Run the full extraction pipeline and write samples to disk.

        Args:
            output_dir: Root directory; samples go to ``output_dir/{split}/``.
            split:      Dataset split name (e.g. ``"train"``, ``"val"``).
            start_idx:  Starting sample index for directory naming.

        Returns:
            Number of samples successfully written.
        """
        cfg = self.cfg
        obs_size: int = cfg.data.obs_size

        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"[extract] reading bag: {self.bag_path}")
        messages = self._read_messages(self.bag_path, self._ros_version)

        gps_msgs = messages[cfg.topics.gps]
        gps_traj = self._build_gps_trajectory(gps_msgs)
        if not gps_traj:
            print("[error] no valid GPS fixes in bag — aborting")
            return 0

        goal_lat = self.goal_lat if self.goal_lat is not None else gps_traj[-1][1]
        goal_lon = self.goal_lon if self.goal_lon is not None else gps_traj[-1][2]
        print(f"[extract] goal = ({goal_lat:.6f}, {goal_lon:.6f})")

        # ── OSRM map matching (episode level, 1 API call) ──────────────────────
        episode_gps = [(lat, lon) for _, lat, lon in gps_traj]
        episode_ts  = [ts_ns // NS_PER_S for ts_ns, _, _ in gps_traj]
        radius_m = float(cfg.map.get("osrm_match_radius_rosbag", 25.0))
        threshold_m = float(cfg.map.get("osrm_valid_threshold_m", 10.0))

        print(f"[extract] OSRM map matching ({len(episode_gps)} GPS points, radius={radius_m}m) …")
        try:
            matched_route, avg_dev = self._router.match(episode_gps, episode_ts, radius_m=radius_m)
        except OSRMMatchError as exc:
            print(f"[error] OSRM match failed: {exc} — aborting")
            return 0

        print(f"[extract] matched route: {len(matched_route)} points, avg deviation={avg_dev:.1f}m")
        if not is_route_valid(matched_route, episode_gps, threshold_m=threshold_m):
            print(f"[error] avg deviation {avg_dev:.1f}m > {threshold_m}m threshold — "
                  "episode path not in OSM, aborting")
            return 0

        # ── Sync frames ────────────────────────────────────────────────────────
        print("[extract] synchronising frames …")
        frames = self._build_synced_frames(messages, cfg)
        print(f"[extract] {len(frames)} synced frames")

        written = 0
        pbar = _progress(enumerate(frames), total=len(frames), desc="writing samples")

        for i, frame in pbar:
            sample_idx = start_idx + i
            sample_dir = split_dir / f"sample_{sample_idx:06d}"
            sample_dir.mkdir(exist_ok=True)

            def _save_obs(img: Image.Image, name: str) -> None:
                img.resize((obs_size, obs_size), Image.LANCZOS).save(sample_dir / name)

            _save_obs(frame.front_img,   "obs_0.png")
            _save_obs(frame.front_past1, "obs_1.png")
            _save_obs(frame.front_past2, "obs_2.png")
            _save_obs(frame.rear_img,    "obs_3.png")

            # Map image — uses OSRM matched route
            map_img = self._renderer.render(
                lat=frame.lat,
                lon=frame.lon,
                heading_deg=frame.heading_deg,
                route_latlons=matched_route,
                goal_lat=goal_lat,
                goal_lon=goal_lon,
            )
            if map_img is None:
                print(f"[warn] tile render failed for sample {sample_idx}, skipping")
                sample_dir.rmdir()
                continue
            map_img.save(sample_dir / "map.png")

            # GT waypoints + route direction (radians, body frame)
            waypoints, route_dir_rad = self._gt_waypoints(
                gps_traj=gps_traj,
                matched_route=matched_route,
                current_ts_ns=frame.ts_ns,
                current_lat=frame.lat,
                current_lon=frame.lon,
                heading_deg=frame.heading_deg,
                cfg=cfg,
            )

            meta = {
                "ts_ns": frame.ts_ns,
                "lat": frame.lat,
                "lon": frame.lon,
                "heading_deg": frame.heading_deg,
                "goal_lat": goal_lat,
                "goal_lon": goal_lon,
                "gt_waypoints": waypoints,
                "route_direction": route_dir_rad,   # radians, body frame
            }
            with open(sample_dir / "meta.json", "w") as fh:
                json.dump(meta, fh, indent=2)

            written += 1

        print(f"[extract] wrote {written} samples to {split_dir}")
        return written


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def _load_config(config_path: Path, map_config_path: Optional[Path] = None):
    """Load rosbag_topics.yaml merged with map.yaml via OmegaConf."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    if map_config_path is None:
        map_config_path = config_path.parent / "map.yaml"
    if map_config_path.exists():
        map_cfg = OmegaConf.load(map_config_path)
        cfg = OmegaConf.merge({"map": map_cfg}, cfg)
    return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract dynav training samples from a ROS2/ROS1 rosbag."
    )
    parser.add_argument("--bag", required=True, type=Path, metavar="PATH",
                        help="Path to the .db3 or .bag file (or directory for split bags).")
    parser.add_argument("--output", required=True, type=Path, metavar="DIR",
                        help="Root output directory; samples → OUTPUT/SPLIT/sample_XXXXXX/.")
    parser.add_argument("--split", default="train", metavar="NAME",
                        help="Dataset split name (default: train).")
    parser.add_argument("--goal-lat", type=float, default=None, metavar="LAT",
                        help="Goal latitude (default: last GPS point).")
    parser.add_argument("--goal-lon", type=float, default=None, metavar="LON",
                        help="Goal longitude (default: last GPS point).")
    parser.add_argument("--config", type=Path,
                        default=_REPO / "configs" / "rosbag_topics.yaml", metavar="YAML",
                        help="Path to rosbag_topics.yaml (default: configs/rosbag_topics.yaml).")
    parser.add_argument("--map-config", type=Path, default=None, metavar="YAML",
                        help="Path to map.yaml (default: same dir as --config).")
    parser.add_argument("--ros-version", type=int, choices=[1, 2], default=None, metavar="N",
                        help="ROS version: 1 or 2 (auto-detected if omitted).")
    parser.add_argument("--start-idx", type=int, default=0, metavar="N",
                        help="Starting sample index for appending to existing dataset.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    bag_path = args.bag.resolve()
    if not bag_path.exists():
        print(f"[error] bag not found: {bag_path}", file=sys.stderr)
        sys.exit(1)

    cfg = _load_config(args.config.resolve(), args.map_config)

    extractor = BagExtractor(
        bag_path=bag_path,
        cfg=cfg,
        goal_lat=args.goal_lat,
        goal_lon=args.goal_lon,
        ros_version=args.ros_version,
    )
    n_written = extractor.extract(
        output_dir=args.output.resolve(),
        split=args.split,
        start_idx=args.start_idx,
    )
    sys.exit(0 if n_written > 0 else 1)


if __name__ == "__main__":
    main()
