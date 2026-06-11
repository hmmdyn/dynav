"""FrodoBots-7k (LeRobot v1.6) dataset pipeline.

Replaces the legacy CSV/m3u8 pipeline (extract_frodobots_segments.py →
extract_frodobots_frames.py → build_frodobots_dataset.py) with a two-stage,
fully automated build that consumes the zarr cache directly:

  Stage "index": episode QA → segment refinement → OSM route + route QA
                 → per-sample candidate enumeration with maneuver/environment
                 labels (no image I/O).
  Stage "build": stratified candidate selection → video frame decode +
                 map rendering → samples + QA report.

Ground-truth geometry comes from the dataset's EKF outputs
(``observation.filtered_position`` — local ENU metres,
``observation.filtered_heading`` — radians CCW from East), never from raw
GPS interpolation.
"""

from .reader import Frodo7kReader, EpisodeData
from .qa import refine_segments, segment_qa, SegmentQA
from .classify import classify_candidates, maneuver_label, scene_label
from .sampling import candidate_indices, select_balanced, ride_split

__all__ = [
    "Frodo7kReader", "EpisodeData",
    "refine_segments", "segment_qa", "SegmentQA",
    "classify_candidates", "maneuver_label", "scene_label",
    "candidate_indices", "select_balanced", "ride_split",
]
