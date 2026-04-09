"""Extract dynav training data from ROS2/ROS1 rosbag files.

Reads a rosbag, extracts synchronized camera/GPS/IMU messages, renders
heading-up OSM map images via the osmnav submodule, and writes samples in
the dynavDataset format:

    data/{split}/sample_XXXXXX/
        obs_0.png    # front camera, current frame
        obs_1.png    # front camera, 1 Δt ago
        obs_2.png    # front camera, 2 Δt ago
        obs_3.png    # rear camera, current frame
        map.png      # heading-up OSM map + route overlay, 224×224
        meta.json    # gt_waypoints, route_direction

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Repository paths — must be set before project imports
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "osmnav/src/osmnav"))
sys.path.insert(0, str(_REPO / "osmnav/src/nomad_map_context"))

from nomad_map_context.tile_cache import TileCache  # noqa: E402
from nomad_map_context.tile_renderer import TileRenderer  # noqa: E402
from nomad_map_context.route_overlay import (  # noqa: E402
    draw_goal_marker,
    draw_position_marker,
    draw_route,
)
from nomad_map_context.image_processor import (  # noqa: E402
    crop_ego_view,
    rotate_north_up,
)

# ---------------------------------------------------------------------------
# Optional progress bar
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, total: int, desc: str = ""):
        return _tqdm(iterable, total=total, desc=desc)

except ImportError:  # pragma: no cover
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
EARTH_RADIUS_M: float = 6_371_000.0  # mean Earth radius


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
    """Return the great-circle distance in metres between two lat/lon points.

    Args:
        lat1: Latitude of point 1 in degrees.
        lon1: Longitude of point 1 in degrees.
        lat2: Latitude of point 2 in degrees.
        lon2: Longitude of point 2 in degrees.

    Returns:
        Distance in metres.
    """
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
    """Compute the initial bearing (compass degrees) from point 1 to point 2.

    Args:
        lat1: Origin latitude in degrees.
        lon1: Origin longitude in degrees.
        lat2: Destination latitude in degrees.
        lon2: Destination longitude in degrees.

    Returns:
        Bearing in degrees, 0 = north, clockwise.
    """
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
    """Convert a lat/lon to robot body-frame (x=forward, y=left) in metres.

    Args:
        lat:         Target latitude.
        lon:         Target longitude.
        ref_lat:     Robot reference latitude.
        ref_lon:     Robot reference longitude.
        heading_deg: Robot compass heading in degrees (0=north, CW).

    Returns:
        (x_m, y_m) in robot body frame; x is forward, y is left.
    """
    # ENU displacement (east, north)
    d_lat = math.radians(lat - ref_lat)
    d_lon = math.radians(lon - ref_lon)
    north_m = d_lat * EARTH_RADIUS_M
    east_m = d_lon * EARTH_RADIUS_M * math.cos(math.radians(ref_lat))

    # Rotate from ENU into body frame.  heading_deg is CW from north.
    heading_r = math.radians(heading_deg)
    x_fwd = north_m * math.cos(heading_r) + east_m * math.sin(heading_r)
    y_left = -north_m * math.sin(heading_r) + east_m * math.cos(heading_r)
    return x_fwd, y_left


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------
def _decode_image_msg(msg) -> Image.Image:
    """Decode a sensor_msgs/Image message to a PIL RGB image.

    Args:
        msg: Deserialised ROS sensor_msgs/Image.

    Returns:
        PIL Image in RGB mode.
    """
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
        # Fallback: interpret as bgr8
        arr = data.reshape(h, w, 3)
        return Image.fromarray(arr[:, :, ::-1].copy(), "RGB")


def _decode_compressed_msg(msg) -> Image.Image:
    """Decode a sensor_msgs/CompressedImage message to a PIL RGB image.

    Args:
        msg: Deserialised ROS sensor_msgs/CompressedImage.

    Returns:
        PIL Image in RGB mode.
    """
    import cv2  # imported here to keep cv2 optional for raw-image-only bags

    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode returned None — corrupt compressed image?")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "RGB")


def _is_compressed_topic(topic: str) -> bool:
    """Return True if the topic name suggests a compressed image stream.

    Args:
        topic: ROS topic string.

    Returns:
        True when the topic ends with ``/compressed``.
    """
    return topic.endswith("/compressed")


def _decode_any_image(msg, topic: str) -> Image.Image:
    """Dispatch to the correct decoder based on the topic name.

    Args:
        msg:   Deserialised ROS image message (raw or compressed).
        topic: ROS topic string used to determine encoding.

    Returns:
        PIL Image in RGB mode.
    """
    if _is_compressed_topic(topic):
        return _decode_compressed_msg(msg)
    return _decode_image_msg(msg)


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------
class BagExtractor:
    """Extracts synchronised training samples from a ROS rosbag.

    Args:
        bag_path: Path to the .db3 (ROS2) or .bag (ROS1) file.
        cfg:      OmegaConf DictConfig loaded from ``rosbag_topics.yaml``.
        goal_lat: Optional goal latitude; defaults to last GPS point.
        goal_lon: Optional goal longitude; defaults to last GPS point.
        ros_version: 1 or 2; if None, auto-detected from file extension.
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

        tile_cache = TileCache(cache_dir=cfg.map.tile_cache)
        self._renderer = TileRenderer(
            cache=tile_cache,
            zoom=cfg.map.tile_zoom,
            output_size=cfg.map.render_size,
        )

    # ------------------------------------------------------------------
    # Static / class helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_ros_version(path: Path) -> int:
        """Detect whether *path* is a ROS1 or ROS2 bag by file extension.

        Args:
            path: Path to the bag file or directory.

        Returns:
            1 for ROS1 (.bag), 2 for ROS2 (.db3, .mcap, or directory).
        """
        suffix = path.suffix.lower()
        if suffix == ".bag":
            return 1
        # .db3, .mcap, or a directory (ROS2 split-bag)
        return 2

    # ------------------------------------------------------------------
    # Message reading
    # ------------------------------------------------------------------

    def _read_messages(
        self, bag_path: Path, ros_version: int
    ) -> Dict[str, List[Tuple[int, object]]]:
        """Read all messages from *bag_path* and group them by topic.

        Args:
            bag_path:    Path to the bag file / directory.
            ros_version: 1 or 2.

        Returns:
            Mapping ``{topic: [(ts_ns, msg), ...]}``, sorted by timestamp.
        """
        cfg = self.cfg

        # Determine camera topics.  When dual_camera is set, the dual
        # compressed stream is used and split into front/rear at sync time.
        dual_topic: Optional[str] = cfg.topics.get("dual_camera")
        if dual_topic:
            camera_topics: set = {dual_topic}
        else:
            camera_topics = {cfg.topics.front_camera, cfg.topics.rear_camera}

        topics_of_interest = camera_topics | {cfg.topics.gps, cfg.topics.imu}

        # Optional topics
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
                pbar = _progress(
                    reader.messages(connections=connections),
                    total=total,
                    desc="reading bag",
                )
                for connection, timestamp, rawdata in pbar:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    messages[connection.topic].append((int(timestamp), msg))
        else:
            from rosbags.rosbag1 import Reader as Reader1
            from rosbags.typesys import Stores, get_typestore

            typestore = get_typestore(Stores.ROS1_NOETIC)
            with Reader1(str(bag_path)) as reader:
                connections = [
                    c for c in reader.connections.values()
                    if c.topic in topics_of_interest
                ]
                total = sum(c.msgcount for c in connections)
                pbar = _progress(
                    reader.messages(connections=connections),
                    total=total,
                    desc="reading bag",
                )
                for connection, timestamp, rawdata in pbar:
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                    messages[connection.topic].append((int(timestamp), msg))

        # Sort each topic by timestamp
        for topic in messages:
            messages[topic].sort(key=lambda x: x[0])

        return messages

    # ------------------------------------------------------------------
    # GPS trajectory
    # ------------------------------------------------------------------

    def _build_gps_trajectory(
        self, gps_msgs: List[Tuple[int, object]]
    ) -> List[Tuple[int, float, float]]:
        """Build a list of (ts_ns, lat, lon) from NavSatFix messages.

        Args:
            gps_msgs: List of (timestamp_ns, msg) pairs from the GPS topic.

        Returns:
            List of (ts_ns, lat, lon) sorted by timestamp, with NaN-filtered
            and STATUS_NO_FIX entries removed.
        """
        trajectory: List[Tuple[int, float, float]] = []
        for ts_ns, msg in gps_msgs:
            # Filter invalid fixes (status < 0 means NO_FIX in sensor_msgs/NavSatStatus)
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
        """Split a dual-fisheye image into front (left) and rear (right) halves.

        The Insta360 X2/X3 dual stream places the front lens on the left half
        and the back lens on the right half of the frame.

        Args:
            img: Full dual-fisheye PIL Image (width = 2 × height, approximately).

        Returns:
            (front_img, rear_img) as separate PIL Images of equal size.
        """
        w, h = img.size
        mid = w // 2
        front = img.crop((0, 0, mid, h))
        rear = img.crop((mid, 0, w, h))
        return front, rear

    def _get_heading(
        self,
        ts_ns: int,
        imu_msgs: List[Tuple[int, object]],
        gps_traj: List[Tuple[int, float, float]],
        source: str,
        odom_msgs: Optional[List[Tuple[int, object]]] = None,
        gps_heading_msgs: Optional[List[Tuple[int, object]]] = None,
    ) -> Optional[float]:
        """Compute compass heading (degrees, CW from north) at *ts_ns*.

        Args:
            ts_ns:             Query timestamp in nanoseconds.
            imu_msgs:          Sorted list of (ts_ns, IMU msg) pairs.
            gps_traj:          Sorted GPS trajectory list.
            source:            One of ``"gps_heading"``, ``"imu"``, ``"gps"``,
                               or ``"odom"``.
            odom_msgs:         Optional odometry message list (source="odom").
            gps_heading_msgs:  Optional QuaternionStamped list (source="gps_heading").

        Returns:
            Heading in degrees, or None if the required data is unavailable.
        """
        if source == "gps_heading":
            if not gps_heading_msgs:
                return None
            tol_ns = int(self.cfg.sync.imu_tol_s * NS_PER_S)
            msg = self._find_nearest(gps_heading_msgs, ts_ns, tol_ns)
            if msg is None:
                return None
            # geometry_msgs/QuaternionStamped: msg.quaternion.{x,y,z,w}
            q = msg.quaternion
            # ENU quaternion — convert yaw (CCW from East) to compass (CW from North)
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
            # ENU quaternion: yaw is CCW from East
            yaw_rad = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            # Convert ENU yaw to compass bearing (CW from north)
            return (90.0 - math.degrees(yaw_rad)) % 360.0

        elif source == "gps":
            # Estimate heading from two consecutive GPS points
            idx = self._find_nearest_index(gps_traj, ts_ns)
            if idx is None or idx + 1 >= len(gps_traj):
                return None
            _, lat0, lon0 = gps_traj[idx]
            _, lat1, lon1 = gps_traj[idx + 1]
            dist = _haversine_m(lat0, lon0, lat1, lon1)
            if dist < 0.1:  # too short to estimate bearing reliably
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
        """Return the message closest in time to *ts_ns* within *tol_ns*.

        Args:
            msgs:   Sorted list of (timestamp_ns, msg).
            ts_ns:  Query timestamp in nanoseconds.
            tol_ns: Maximum allowed time difference in nanoseconds.

        Returns:
            The nearest message, or None if no message is within tolerance.
        """
        if not msgs:
            return None
        # Binary search for insertion point
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
        """Return the index in *traj* of the entry closest to *ts_ns*.

        Args:
            traj:  Sorted list of (ts_ns, lat, lon).
            ts_ns: Query timestamp in nanoseconds.

        Returns:
            Index, or None if *traj* is empty.
        """
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
        """Synchronise front/rear camera, GPS, and IMU into SyncedFrames.

        Downsamples the front-camera stream to *camera_hz*, then for each
        retained frame finds matching GPS/IMU data and the two past frames.
        Frames where the robot is stationary or GPS/IMU sync fails are
        discarded.

        Args:
            messages: Topic-keyed message lists from :meth:`_read_messages`.
            cfg:      OmegaConf config.

        Returns:
            List of synchronised frames, ready for map rendering.
        """
        gps_topic = cfg.topics.gps
        imu_topic = cfg.topics.imu
        odom_topic: Optional[str] = cfg.topics.get("odom")
        gps_heading_topic: Optional[str] = cfg.topics.get("gps_heading")
        dual_topic: Optional[str] = cfg.topics.get("dual_camera")

        # Camera message lists — dual mode splits the image at decode time
        if dual_topic:
            dual_msgs = messages[dual_topic]
            front_msgs = dual_msgs   # iterated as primary stream
            rear_msgs = dual_msgs    # same list; front/rear decoded together below
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
            # Downsample
            if last_emitted_ts is not None and (ts_ns - last_emitted_ts) < interval_ns:
                continue

            # GPS sync
            gps_msg = self._find_nearest(gps_msgs, ts_ns, gps_tol_ns)
            if gps_msg is None:
                continue
            lat = float(gps_msg.latitude)
            lon = float(gps_msg.longitude)
            if math.isnan(lat) or math.isnan(lon):
                continue

            # Speed check (use GPS-derived speed if available, else estimate)
            gps_speed: Optional[float] = None
            if hasattr(gps_msg, "speed") and not math.isnan(float(gps_msg.speed)):
                gps_speed = float(gps_msg.speed)
            else:
                # Estimate from GPS trajectory
                traj_idx = self._find_nearest_index(gps_traj, ts_ns)
                if traj_idx is not None and traj_idx > 0:
                    prev_ts, prev_lat, prev_lon = gps_traj[traj_idx - 1]
                    cur_ts, cur_lat, cur_lon = gps_traj[traj_idx]
                    dt_s = (cur_ts - prev_ts) / NS_PER_S
                    if dt_s > 0:
                        gps_speed = _haversine_m(prev_lat, prev_lon, cur_lat, cur_lon) / dt_s
            if gps_speed is not None and gps_speed < min_speed:
                continue

            # Heading
            heading_deg = self._get_heading(
                ts_ns, imu_msgs, gps_traj, cfg.sync.heading_source, odom_msgs,
                gps_heading_msgs=gps_heading_msgs,
            )
            if heading_deg is None:
                continue

            # Rear camera (or dual — same list)
            rear_msg = self._find_nearest(rear_msgs, ts_ns, tol_ns=imu_tol_ns * 5)
            if rear_msg is None:
                continue

            # Past front frames
            front_past1_msg = self._find_nearest(
                front_msgs, ts_ns - past_dt_ns, tol_ns=past_dt_ns // 2
            )
            front_past2_msg = self._find_nearest(
                front_msgs, ts_ns - 2 * past_dt_ns, tol_ns=past_dt_ns // 2
            )
            if front_past1_msg is None or front_past2_msg is None:
                continue

            # Decode images
            try:
                if _dual_mode:
                    # Decode dual image once, split into front and rear halves
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

            frames.append(
                SyncedFrame(
                    ts_ns=ts_ns,
                    front_img=front_img,
                    front_past1=front_past1,
                    front_past2=front_past2,
                    rear_img=rear_img,
                    lat=lat,
                    lon=lon,
                    heading_deg=heading_deg,
                )
            )
            last_emitted_ts = ts_ns

        return frames

    # ------------------------------------------------------------------
    # GT waypoints
    # ------------------------------------------------------------------

    def _gt_waypoints(
        self,
        gps_traj: List[Tuple[int, float, float]],
        current_ts_ns: int,
        current_lat: float,
        current_lon: float,
        heading_deg: float,
        cfg,
    ) -> Tuple[List[List[float]], float]:
        """Compute ground-truth waypoints and route direction for one frame.

        Walks forward along *gps_traj* from the current position, sampling
        points at increments of ``cfg.data.wp_spacing_m``.  Each waypoint is
        expressed in robot body frame (x=forward, y=left) and normalised to
        [-1, 1] by ``cfg.data.max_wp_dist_m``.

        Args:
            gps_traj:      Full GPS trajectory as (ts_ns, lat, lon) list.
            current_ts_ns: Timestamp of the current frame.
            current_lat:   GPS latitude at the current frame.
            current_lon:   GPS longitude at the current frame.
            heading_deg:   Robot heading at the current frame (compass deg).
            cfg:           OmegaConf config.

        Returns:
            (waypoints, route_direction_deg) where:
            - waypoints is a list of [x_norm, y_norm] pairs, length H.
            - route_direction_deg is the bearing toward the first waypoint.
        """
        horizon: int = cfg.data.horizon
        wp_spacing_m: float = cfg.data.wp_spacing_m
        max_dist_m: float = cfg.data.max_wp_dist_m

        # Find the index in the trajectory closest to the current timestamp
        start_idx = self._find_nearest_index(gps_traj, current_ts_ns)
        if start_idx is None:
            # Fallback: return zero waypoints
            return [[0.0, 0.0]] * horizon, heading_deg

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
                # Linearly interpolate to the exact target distance
                overshoot = accumulated_m - target_dist_m
                frac = 1.0 - overshoot / step_m if step_m > 0 else 1.0
                wp_lat = gps_traj[search_idx - 1][1] + frac * (next_lat - gps_traj[search_idx - 1][1])
                wp_lon = gps_traj[search_idx - 1][2] + frac * (next_lon - gps_traj[search_idx - 1][2])

                x_m, y_m = _latlon_to_body(wp_lat, wp_lon, current_lat, current_lon, heading_deg)
                # Normalise to [-1, 1]
                x_norm = max(-1.0, min(1.0, x_m / max_dist_m))
                y_norm = max(-1.0, min(1.0, y_m / max_dist_m))
                waypoints.append([x_norm, y_norm])
                target_dist_m += wp_spacing_m

        # Pad with the last waypoint if not enough trajectory remains
        while len(waypoints) < horizon:
            waypoints.append(waypoints[-1] if waypoints else [0.0, 0.0])

        # Route direction: bearing from current position to first waypoint
        if len(gps_traj) > start_idx + 1:
            _, fw_lat, fw_lon = gps_traj[start_idx + 1]
            route_dir = _bearing_deg(current_lat, current_lon, fw_lat, fw_lon)
        else:
            route_dir = heading_deg

        return waypoints, route_dir

    # ------------------------------------------------------------------
    # Map rendering
    # ------------------------------------------------------------------

    def _render_map(
        self,
        lat: float,
        lon: float,
        heading_deg: float,
        gps_traj: List[Tuple[int, float, float]],
        goal_lat: float,
        goal_lon: float,
        output_size: int,
    ) -> Optional[Image.Image]:
        """Render a heading-up OSM map image for one frame.

        Fetches tiles centred on (lat, lon), draws the GPS route, robot
        position, and goal marker, then rotates so the robot's heading faces
        up and crops to *output_size* × *output_size*.

        Args:
            lat:         Robot latitude.
            lon:         Robot longitude.
            heading_deg: Robot compass heading.
            gps_traj:    Full GPS trajectory (used as route overlay).
            goal_lat:    Goal latitude.
            goal_lon:    Goal longitude.
            output_size: Final image edge length in pixels.

        Returns:
            PIL Image (RGB, output_size × output_size), or None on failure.
        """
        result = self._renderer.render(lat, lon)
        if result is None:
            return None
        tile_img, geo_tf = result

        # Draw route (full trajectory)
        route_latlons = [(lat_, lon_) for _, lat_, lon_ in gps_traj]
        draw_route(tile_img, route_latlons, geo_tf)

        # Robot position marker
        rob_px, rob_py = TileRenderer.latlon_to_pixel(lat, lon, geo_tf)
        draw_position_marker(tile_img, rob_px, rob_py, heading_deg)

        # Goal marker
        goal_px, goal_py = TileRenderer.latlon_to_pixel(goal_lat, goal_lon, geo_tf)
        draw_goal_marker(tile_img, goal_px, goal_py)

        # Rotate so heading faces up
        rotated, new_pos = rotate_north_up(tile_img, heading_deg, (rob_px, rob_py))

        # Crop egocentric view
        cropped, _ = crop_ego_view(rotated, new_pos, output_size=output_size)

        return cropped.resize((output_size, output_size), Image.LANCZOS)

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

        Reads the bag, synchronises frames, renders map images, computes GT
        waypoints, and writes each sample as a directory containing PNGs and
        a JSON metadata file.

        Args:
            output_dir: Root directory under which ``{split}/`` will be created.
            split:      Dataset split name (e.g. ``"train"``, ``"val"``).
            start_idx:  Starting index for sample directory names (useful when
                        appending to an existing dataset).

        Returns:
            Number of samples successfully written.
        """
        cfg = self.cfg
        obs_size: int = cfg.data.obs_size
        map_output_size: int = cfg.map.output_size

        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"[extract] reading bag: {self.bag_path}")
        messages = self._read_messages(self.bag_path, self._ros_version)

        front_topic = cfg.topics.front_camera
        gps_topic = cfg.topics.gps
        gps_msgs = messages[gps_topic]

        gps_traj = self._build_gps_trajectory(gps_msgs)
        if not gps_traj:
            print("[error] no valid GPS fixes in bag — aborting")
            return 0

        # Resolve goal coordinates
        goal_lat = self.goal_lat if self.goal_lat is not None else gps_traj[-1][1]
        goal_lon = self.goal_lon if self.goal_lon is not None else gps_traj[-1][2]
        print(f"[extract] goal = ({goal_lat:.6f}, {goal_lon:.6f})")

        print("[extract] synchronising frames …")
        frames = self._build_synced_frames(messages, cfg)
        print(f"[extract] {len(frames)} synced frames")

        written = 0
        pbar = _progress(enumerate(frames), total=len(frames), desc="writing samples")

        for i, frame in pbar:
            sample_idx = start_idx + i
            sample_dir = split_dir / f"sample_{sample_idx:06d}"
            sample_dir.mkdir(exist_ok=True)

            # Observation images (resize to obs_size)
            def _save_obs(img: Image.Image, name: str) -> None:
                img_resized = img.resize((obs_size, obs_size), Image.LANCZOS)
                img_resized.save(sample_dir / name)

            _save_obs(frame.front_img, "obs_0.png")
            _save_obs(frame.front_past1, "obs_1.png")
            _save_obs(frame.front_past2, "obs_2.png")
            _save_obs(frame.rear_img, "obs_3.png")

            # Map image
            map_img = self._render_map(
                lat=frame.lat,
                lon=frame.lon,
                heading_deg=frame.heading_deg,
                gps_traj=gps_traj,
                goal_lat=goal_lat,
                goal_lon=goal_lon,
                output_size=map_output_size,
            )
            if map_img is None:
                print(f"[warn] tile render failed for sample {sample_idx}, skipping")
                sample_dir.rmdir()
                continue

            map_img = map_img.resize((map_output_size, map_output_size), Image.LANCZOS)
            map_img.save(sample_dir / "map.png")

            # GT waypoints
            waypoints, route_dir = self._gt_waypoints(
                gps_traj=gps_traj,
                current_ts_ns=frame.ts_ns,
                current_lat=frame.lat,
                current_lon=frame.lon,
                heading_deg=frame.heading_deg,
                cfg=cfg,
            )

            # meta.json
            meta = {
                "ts_ns": frame.ts_ns,
                "lat": frame.lat,
                "lon": frame.lon,
                "heading_deg": frame.heading_deg,
                "goal_lat": goal_lat,
                "goal_lon": goal_lon,
                "gt_waypoints": waypoints,   # List[List[float]], shape (H, 2)
                "route_direction": route_dir,
            }
            with open(sample_dir / "meta.json", "w") as fh:
                json.dump(meta, fh, indent=2)

            written += 1

        print(f"[extract] wrote {written} samples to {split_dir}")
        return written


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def _load_config(config_path: Path):
    """Load rosbag_topics.yaml via OmegaConf.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        OmegaConf DictConfig.
    """
    from omegaconf import OmegaConf

    return OmegaConf.load(config_path)


# ---------------------------------------------------------------------------
# Default config (written if the file does not exist)
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_YAML = """\
topics:
  front_camera: "/fisheye/front/image"
  rear_camera:  "/fisheye/back/image"
  gps:          "/j100_0519/sensors/gps_0/fix"
  gps_heading:  "/j100_0519/sensors/gps_0/heading"
  imu:          "/imu/data"
  # dual_camera: "/fisheye/dual/image/compressed"  # alternative to front/rear

sync:
  camera_hz: 10.0
  past_dt_s: 0.5
  gps_tol_s: 0.5
  imu_tol_s: 0.1
  min_speed_mps: 0.3
  heading_source: "gps_heading"   # "gps_heading", "imu", "gps", or "odom"

map:
  tile_cache: "/tmp/nomad_tile_cache"
  tile_zoom: 17
  render_size: 512
  output_size: 224

data:
  horizon: 5
  max_wp_dist_m: 2.5
  wp_spacing_m: 0.5
  obs_size: 224
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Extract dynav training samples from a ROS2/ROS1 rosbag."
    )
    parser.add_argument(
        "--bag",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the .db3 or .bag rosbag file (or directory for split bags).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="DIR",
        help="Root output directory; samples written to OUTPUT/SPLIT/sample_XXXXXX/.",
    )
    parser.add_argument(
        "--split",
        default="train",
        metavar="NAME",
        help="Dataset split name (default: train).",
    )
    parser.add_argument(
        "--goal-lat",
        type=float,
        default=None,
        metavar="LAT",
        help="Goal latitude (default: last GPS point in bag).",
    )
    parser.add_argument(
        "--goal-lon",
        type=float,
        default=None,
        metavar="LON",
        help="Goal longitude (default: last GPS point in bag).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO / "configs" / "rosbag_topics.yaml",
        metavar="YAML",
        help="Path to rosbag_topics.yaml (default: configs/rosbag_topics.yaml).",
    )
    parser.add_argument(
        "--ros-version",
        type=int,
        choices=[1, 2],
        default=None,
        metavar="N",
        help="ROS version: 1 or 2 (auto-detected from extension if omitted).",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        metavar="N",
        help="Starting sample index, for appending to an existing dataset (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the extract_rosbag CLI."""
    args = _parse_args()

    bag_path: Path = args.bag.resolve()
    if not bag_path.exists():
        print(f"[error] bag not found: {bag_path}", file=sys.stderr)
        sys.exit(1)

    config_path: Path = args.config.resolve()
    if not config_path.exists():
        print(f"[info] config not found; writing default to {config_path}")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(_DEFAULT_CONFIG_YAML)

    cfg = _load_config(config_path)

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
