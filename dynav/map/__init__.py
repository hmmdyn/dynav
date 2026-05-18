"""dynav.map — unified OSM map rendering for training and inference."""

from .tiles import TileCache
from .routing import OSRMRouter, OSRMMatchError, OSRMRouteError, is_route_valid, find_current_idx
from .segment import segment_gps_episode
from .renderer import MapRenderer
from .osm_snap import (
    fetch_ped_network,
    snap_trajectory,
    snap_trajectory_graph,
    build_osm_graph,
    dijkstra_path,
    bearing_deg,
    haversine_m,
)

__all__ = [
    # Core rendering
    "TileCache",
    "MapRenderer",
    # OSRM routing (ROS inference + rosbag training)
    "OSRMRouter",
    "OSRMMatchError",
    "OSRMRouteError",
    "is_route_valid",
    "find_current_idx",
    # OSM pedestrian snap (FrodoBots dataset building, offline)
    "fetch_ped_network",
    "snap_trajectory",
    "snap_trajectory_graph",
    "build_osm_graph",
    "dijkstra_path",
    "bearing_deg",
    "haversine_m",
    # GPS segmentation
    "segment_gps_episode",
]
