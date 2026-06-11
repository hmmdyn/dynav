"""Build the dynav training dataset from FrodoBots-7k (LeRobot v1.6).

Replaces the legacy 3-step pipeline (extract_frodobots_segments /
extract_frodobots_frames / build_frodobots_dataset).  Stages are resumable
and every accept/reject decision is recorded with its reason.

  index   episode QA → segment refinement → OSM route + route QA
          → labelled sample candidates.  Writes index/ep_XXXXXX.json.
  select  ride-level split + stratified quota fill.
          Writes selected_{train,val}.json + composition section of report.
  build   video decode + map render → train/val sample dirs
          (obs_0..3.png, map.png, meta.json) + per-sample QA.
  report  stats.json + report.md + contact sheets (+ optional GIFs).

Usage::

    python scripts/build_frodobots7k_dataset.py --stage index --limit 300
    python scripts/build_frodobots7k_dataset.py --stage select
    python scripts/build_frodobots7k_dataset.py --stage build
    python scripts/build_frodobots7k_dataset.py --stage report
    python scripts/build_frodobots7k_dataset.py --stage all --limit 300

GT geometry comes from EKF outputs (filtered_position / filtered_heading) —
no GPS interpolation, no bearing-window heading.  See dynav/frodo7k/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from dynav.frodo7k.reader import Frodo7kReader, EpisodeData
from dynav.frodo7k.qa import QAConfig, refine_segments, segment_qa, route_qa
from dynav.frodo7k.classify import classify_candidates, scene_label
from dynav.frodo7k.sampling import candidate_indices, ride_split, select_balanced
from dynav.map import MapRenderer
from dynav.map.tiles import TileCache
from dynav.map.osm_snap import (
    _DRIVE_PED_TAGS,
    fetch_network_bbox,
    fetch_scene_bbox,
    route_by_graph,
    snap_trajectory,
)

R_EARTH = 6_371_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def qa_config(cfg: dict) -> QAConfig:
    return QAConfig(**cfg["qa"])


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _local_metric(lat0: float, lon0: float):
    """(lat, lon) ↔ local (x=E, y=N) metres anchored at (lat0, lon0)."""
    mlat = R_EARTH * math.pi / 180.0
    mlon = mlat * math.cos(math.radians(lat0))

    def to_xy(lats, lons):
        return np.stack([(np.asarray(lons) - lon0) * mlon,
                         (np.asarray(lats) - lat0) * mlat], axis=-1)

    return to_xy


def _xy_to_latlon(xy: np.ndarray, lat0: float, lon0: float) -> tuple[float, float]:
    """Inverse of _local_metric's to_xy for a single (x=E, y=N) point."""
    mlat = R_EARTH * math.pi / 180.0
    mlon = mlat * math.cos(math.radians(lat0))
    return (lat0 + float(xy[1]) / mlat, lon0 + float(xy[0]) / mlon)


def _project_to_polyline(route_xy: np.ndarray, p: np.ndarray
                         ) -> tuple[int, np.ndarray, float, float]:
    """Project point *p* onto a polyline (segment-wise, not vertex-wise).

    Graph routes have sparse vertices (long straight OSM edges), so the
    nearest-*vertex* distance can be tens of metres even when the robot sits
    exactly on the line — all robot↔route geometry must use this instead.

    Returns:
        (segment_index, projected_point, arc_position_m, distance_m)
    """
    seg = np.diff(route_xy, axis=0)                       # (M-1, 2)
    seg_len2 = np.maximum(np.sum(seg ** 2, axis=1), 1e-12)
    t = np.clip(np.sum((p - route_xy[:-1]) * seg, axis=1) / seg_len2, 0.0, 1.0)
    proj = route_xy[:-1] + seg * t[:, None]               # (M-1, 2)
    d = np.linalg.norm(proj - p, axis=1)
    j = int(np.argmin(d))
    seg_len = np.sqrt(seg_len2)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    return j, proj[j], float(cum[j] + t[j] * seg_len[j]), float(d[j])


def _route_direction(route_xy: np.ndarray, pos_xy: np.ndarray,
                     heading: float, lookahead_m: float) -> float:
    """Body-frame angle (rad) to the route point *lookahead_m* ahead of the
    robot's projection onto the route polyline."""
    seg_len = np.linalg.norm(np.diff(route_xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    _, _, arc_s, _ = _project_to_polyline(route_xy, pos_xy)
    target_s = min(arc_s + lookahead_m, cum[-1])
    i1 = min(int(np.searchsorted(cum, target_s)), len(route_xy) - 1)
    d = route_xy[i1] - pos_xy
    fwd = np.array([math.cos(heading), math.sin(heading)])
    left = np.array([-math.sin(heading), math.cos(heading)])
    return float(math.atan2(d @ left, d @ fwd))


# ─────────────────────────────────────────────────────────────────────────────
# Stage: index
# ─────────────────────────────────────────────────────────────────────────────

def stage_index(cfg: dict, episodes: list[int] | None, limit: int | None) -> None:
    paths = cfg["paths"]
    out_dir = Path(paths["output_root"]) / "index"
    out_dir.mkdir(parents=True, exist_ok=True)
    osm_cache = Path(paths["osm_cache"])

    reader = Frodo7kReader(paths["frodo7k_root"],
                           bounds_cache=Path(paths["output_root"]) / "ep_bounds.npy")
    qcfg = qa_config(cfg)
    rcfg = cfg["route"]
    ccfg = cfg["candidates"]
    scfg = cfg["sample"]
    obs_lookback = scfg["obs_stride"] * (scfg["n_obs"] - 1)
    horizon = scfg["wp_offsets"][-1]

    if episodes is None:
        lens = reader.episode_lengths()
        episodes = [int(e) for e in np.where(lens >= qcfg.min_frames)[0]]
    if limit:
        episodes = episodes[:limit]

    print(f"[index] {len(episodes)} episodes → {out_dir}")
    t0, n_done, n_seg_ok, n_cand = time.time(), 0, 0, 0

    for e in episodes:
        out_path = out_dir / f"ep_{e:06d}.json"
        if out_path.exists():
            n_done += 1
            continue
        try:
            rec = _index_episode(reader, e, qcfg, rcfg, ccfg, scfg,
                                 obs_lookback, horizon, osm_cache)
        except Exception as ex:  # record failures, never crash the run
            rec = {"episode": e, "error": f"{type(ex).__name__}: {ex}", "segments": []}
        out_path.write_text(json.dumps(rec))
        n_done += 1
        for s in rec["segments"]:
            if s["qa"]["accepted"] and s.get("route_qa", {}).get("accepted"):
                n_seg_ok += 1
                n_cand += len(s.get("candidates", []))
        if n_done % 50 == 0:
            el = time.time() - t0
            print(f"  {n_done}/{len(episodes)}  ok_segs={n_seg_ok} cands={n_cand} "
                  f"({el:.0f}s, {el / max(n_done, 1):.2f}s/ep)", flush=True)

    print(f"[index] done: {n_done} episodes, {n_seg_ok} accepted segments, "
          f"{n_cand} candidates ({(time.time() - t0) / 60:.1f} min)")


def _index_episode(reader: Frodo7kReader, e: int, qcfg: QAConfig,
                   rcfg: dict, ccfg: dict, scfg: dict,
                   obs_lookback: int, horizon: int, osm_cache: Path) -> dict:
    ep = reader.episode(e)
    rec = {"episode": e, "ride_id": ep.ride_id, "video": ep.video_path,
           "utm_zone": ep.utm_zone, "n_frames": ep.n, "segments": []}

    for s, t in refine_segments(ep, qcfg):
        seg_rec: dict = {"qa": None}
        qa = segment_qa(ep, s, t, qcfg)
        seg_rec["qa"] = qa.to_dict()
        if not qa.accepted:
            rec["segments"].append(seg_rec)
            continue

        # trajectory lat/lon from EKF (subsampled for routing)
        sub = slice(s, t, rcfg["snap_subsample"])
        lats, lons = ep.latlon(np.arange(*sub.indices(ep.n)))
        lats, lons = np.atleast_1d(lats), np.atleast_1d(lons)

        edges, all_pts, way_count = fetch_network_bbox(
            float(lats.min()), float(lons.min()),
            float(lats.max()), float(lons.max()),
            tags=_DRIVE_PED_TAGS, cache_dir=osm_cache,
        )
        if len(edges) < 5:
            seg_rec["route_qa"] = {"accepted": False,
                                   "reject_reasons": ["no_osm_network"]}
            rec["segments"].append(seg_rec)
            continue

        snapped, mean_dist = snap_trajectory(list(lats), list(lons), edges, all_pts)

        # Route by graph wayfinding (same generator style as inference
        # /route): one Dijkstra path to the segment goal, via-anchors only
        # where the demo provably leaves the shortest path.  Per-point
        # projection zigzag (the legacy "weird map" failure mode) cannot
        # occur by construction.
        rbg = route_by_graph(
            list(lats), list(lons), edges, all_pts,
            max_dev_m=rcfg["max_route_dev_m"],
            max_anchors=rcfg["max_anchors"],
            max_edge_angle_deg=rcfg["max_edge_angle_deg"],
        )
        if rbg is None:
            seg_rec["route_qa"] = {"accepted": False,
                                   "reject_reasons": ["route_demo_divergent"]}
            rec["segments"].append(seg_rec)
            continue
        route, n_anchors, dev_p90 = rbg

        to_xy = _local_metric(ep.lat0, ep.lon0)
        route_xy = to_xy([p[0] for p in route], [p[1] for p in route])
        traj_xy = to_xy(lats, lons)
        rqa = route_qa(
            route_xy, traj_xy, mean_dist,
            max_snap_mean_m=rcfg["max_snap_mean_m"],
            len_ratio_range=tuple(rcfg["len_ratio_range"]),
            max_gap_m=rcfg["max_gap_m"],
            min_monotonic_frac=rcfg["min_monotonic_frac"],
            max_dir_agree_med_deg=rcfg["max_dir_agree_med_deg"],
            n_anchors=n_anchors, dev_p90_m=dev_p90,
        )
        seg_rec["route_qa"] = rqa.to_dict()
        if not rqa.accepted:
            rec["segments"].append(seg_rec)
            continue

        seg_rec["route_latlons"] = [[round(a, 7), round(b, 7)] for a, b in route]

        idxs = candidate_indices(
            ep.heading, s, t, obs_lookback, horizon,
            base_stride=ccfg["base_stride"], turn_stride=ccfg["turn_stride"],
            turn_rate_deg_s=ccfg["turn_rate_deg_s"],
        )
        # bbox area for way density
        dlat = (lats.max() - lats.min()) * R_EARTH * math.pi / 180.0
        dlon = ((lons.max() - lons.min()) * R_EARTH * math.pi / 180.0
                * math.cos(math.radians(ep.lat0)))
        area_km2 = max(dlat + 300.0, 300.0) * max(dlon + 300.0, 300.0) / 1e6

        # scene classification (city / park / straight_road / other)
        try:
            green_polys, building_count = fetch_scene_bbox(
                float(lats.min()), float(lons.min()),
                float(lats.max()), float(lons.max()),
                cache_dir=osm_cache,
            )
        except Exception:
            green_polys, building_count = [], 0
        scene = scene_label(
            lats, lons, green_polys, building_count, area_km2,
            way_density=way_count / area_km2,
            arc_m=qa.arc_m, net_disp_m=qa.net_disp_m,
            turn_total_deg=qa.turn_total_deg,
            park_inside_frac=ccfg["park_inside_frac"],
            city_building_per_km2=ccfg["city_building_per_km2"],
            urban_way_density=ccfg["urban_way_density"],
            straight_min_ratio=ccfg["straight_min_ratio"],
            straight_max_turn_deg_per_m=ccfg["straight_max_turn_deg_per_m"],
        )
        seg_rec["scene"] = scene
        seg_rec["building_count"] = building_count

        seg_rec["candidates"] = classify_candidates(
            ep, s, idxs, horizon, edges, way_count, area_km2, scene=scene,
            intersection_radius_m=ccfg["intersection_radius_m"],
            lookahead_m=ccfg["intersection_lookahead_m"],
            urban_way_density=ccfg["urban_way_density"],
            sparse_way_density=ccfg["sparse_way_density"],
        )
        rec["segments"].append(seg_rec)
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Stage: select
# ─────────────────────────────────────────────────────────────────────────────

def stage_select(cfg: dict) -> None:
    out_root = Path(cfg["paths"]["output_root"])
    index_dir = out_root / "index"
    sel_cfg = cfg["select"]

    pools = {"train": [], "val": []}
    seg_stats: Counter = Counter()
    reject_stats: Counter = Counter()

    for p in sorted(index_dir.glob("ep_*.json")):
        rec = json.loads(p.read_text())
        if rec.get("error"):
            reject_stats["episode_error"] += 1
            continue
        split = ride_split(rec["ride_id"], sel_cfg["val_fraction"])
        for si, seg in enumerate(rec["segments"]):
            if not seg["qa"]["accepted"]:
                for r in seg["qa"]["reject_reasons"]:
                    reject_stats[f"qa:{r}"] += 1
                continue
            rqa = seg.get("route_qa", {})
            if not rqa.get("accepted"):
                for r in rqa.get("reject_reasons", ["missing"]):
                    reject_stats[f"route:{r}"] += 1
                continue
            seg_stats["accepted_segments"] += 1
            key = f"{rec['episode']}:{si}"
            for c in seg.get("candidates", []):
                c2 = dict(c)
                c2.update(episode=rec["episode"], seg_i=si, segment_key=key,
                          ride_id=rec["ride_id"], split=split)
                pools[split].append(c2)

    print(f"[select] candidates: train={len(pools['train'])} val={len(pools['val'])}")
    print(f"[select] segment rejects: {dict(reject_stats)}")

    selected = {}
    for split, target in [("train", sel_cfg["target_train"]),
                          ("val", sel_cfg["target_val"])]:
        selected[split] = select_balanced(
            pools[split], target,
            maneuver_targets=sel_cfg["maneuver_targets"],
            scene_targets=sel_cfg.get("scene_targets"),
            max_per_segment=sel_cfg["max_per_segment"],
            seed=sel_cfg["seed"],
        )
        (out_root / f"selected_{split}.json").write_text(json.dumps(selected[split]))
        comp_m = Counter(c["maneuver"] for c in selected[split])
        comp_s = Counter(c.get("scene", "other") for c in selected[split])
        n_int = sum(c["near_intersection"] for c in selected[split])
        print(f"[select] {split}: {len(selected[split])} samples")
        print(f"         maneuver: {dict(comp_m)}")
        print(f"         scene: {dict(comp_s)}  near_intersection: {n_int}")

    stats = {
        "candidates": {k: len(v) for k, v in pools.items()},
        "selected": {k: len(v) for k, v in selected.items()},
        "rejects": dict(reject_stats),
        "composition": {
            split: {
                "maneuver": dict(Counter(c["maneuver"] for c in sel)),
                "scene": dict(Counter(c.get("scene", "other") for c in sel)),
                "env_density": dict(Counter(c["env_density"] for c in sel)),
                "near_intersection_frac":
                    round(sum(c["near_intersection"] for c in sel) / max(len(sel), 1), 3),
                "difficulty_mean":
                    round(float(np.mean([c["difficulty"] for c in sel])), 3) if sel else 0,
            } for split, sel in selected.items()
        },
    }
    (out_root / "selection_stats.json").write_text(json.dumps(stats, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Stage: build
# ─────────────────────────────────────────────────────────────────────────────

class _VideoDecoder:
    """Decode frames nearest to requested video timestamps (one container)."""

    def __init__(self, path: Path):
        import av

        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.tb = float(self.stream.time_base)

    def get_frames(self, ts_list: list[float], tol: float = 0.02) -> dict[float, "np.ndarray"]:
        """Return {ts: HxWx3 RGB} for each requested ts (sorted internally)."""
        out: dict[float, np.ndarray] = {}
        pending = sorted(set(ts_list))
        if not pending:
            return out
        self.container.seek(int(pending[0] / self.tb), stream=self.stream)
        pi = 0
        last_decode_ts = -1e9
        for frame in self.container.decode(self.stream):
            ts = float(frame.pts * self.tb)
            while pi < len(pending) and ts >= pending[pi] - tol:
                if abs(ts - pending[pi]) <= max(tol, 0.11):
                    out[pending[pi]] = frame.to_ndarray(format="rgb24")
                pi += 1
            if pi >= len(pending):
                break
            if pending[pi] - ts > 5.0 and ts - last_decode_ts > 0:
                self.container.seek(int(pending[pi] / self.tb), stream=self.stream)
            last_decode_ts = ts
        return out

    def close(self):
        self.container.close()


def _resize_obs(img: np.ndarray, size: int, mode: str) -> np.ndarray:
    import cv2

    if mode == "letterbox":
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        y0, x0 = (size - nh) // 2, (size - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)


def stage_build(cfg: dict) -> None:
    import cv2
    from PIL import Image

    paths = cfg["paths"]
    out_root = Path(paths["output_root"])
    index_dir = out_root / "index"
    scfg = cfg["sample"]
    mcfg = cfg["map"]

    tile_cache = TileCache(cache_dir=paths["tile_cache"], tile_url=mcfg["tile_url"])
    renderer = MapRenderer(
        cache=tile_cache, zoom=mcfg["tile_zoom"],
        render_size=mcfg["render_size"], output_size=mcfg["output_size"],
        mode=mcfg["mode"], crop_ratio=mcfg["crop_ratio"],
    )
    reader = Frodo7kReader(paths["frodo7k_root"],
                           bounds_cache=out_root / "ep_bounds.npy")

    obs_stride = scfg["obs_stride"]
    n_obs = scfg["n_obs"]
    wp_offsets = scfg["wp_offsets"]
    norm_m = scfg["waypoint_norm_m"]
    size = scfg["image_size"]

    build_qa: Counter = Counter()
    t0 = time.time()

    for split in ["train", "val"]:
        sel_path = out_root / f"selected_{split}.json"
        if not sel_path.exists():
            print(f"[build] {sel_path} missing — run --stage select first")
            return
        selected = json.loads(sel_path.read_text())
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        by_ep: dict[int, list[dict]] = {}
        for c in selected:
            by_ep.setdefault(c["episode"], []).append(c)

        print(f"[build] {split}: {len(selected)} samples in {len(by_ep)} episodes")
        n_done = 0

        for e, cands in sorted(by_ep.items()):
            idx_rec = json.loads((index_dir / f"ep_{e:06d}.json").read_text())
            ep = reader.episode(e)
            video = reader.video_file(ep)
            if not video.exists():
                build_qa["missing_video"] += len(cands)
                continue

            # skip episodes whose samples all exist (resume)
            todo = [c for c in cands
                    if not (split_dir / _sample_name(ep, c) / "meta.json").exists()]
            if not todo:
                n_done += len(cands)
                continue

            ts_needed = []
            for c in todo:
                for k in range(n_obs):
                    ts_needed.append(float(ep.video_ts[c["idx"] - k * obs_stride]))
            dec = _VideoDecoder(video)
            frames = dec.get_frames(ts_needed)
            dec.close()

            for c in todo:
                ok, reason = _write_sample(
                    ep, c, idx_rec, frames, renderer, split_dir,
                    obs_stride, n_obs, wp_offsets, norm_m, size,
                    scfg["obs_resize"], scfg["route_lookahead_m"],
                    scfg.get("route_wp_max_deg", 75.0),
                )
                build_qa["ok" if ok else f"drop:{reason}"] += 1
                n_done += 1

            if n_done and n_done % 500 < len(todo):
                el = time.time() - t0
                print(f"  {split} {n_done}/{len(selected)} "
                      f"({el / 60:.1f} min, {n_done / max(el, 1):.1f}/s)", flush=True)

    (out_root / "build_stats.json").write_text(json.dumps(dict(build_qa), indent=2))
    print(f"[build] done ({(time.time() - t0) / 60:.1f} min): {dict(build_qa)}")


def _sample_name(ep: EpisodeData, c: dict) -> str:
    return f"sample_{ep.ride_id}_ep{c['episode']:06d}_{c['idx']:06d}"


def _write_sample(ep, c, idx_rec, frames, renderer, split_dir,
                  obs_stride, n_obs, wp_offsets, norm_m, size,
                  obs_resize, route_lookahead_m,
                  route_wp_max_deg: float = 75.0) -> tuple[bool, str]:
    import cv2

    idx = c["idx"]
    seg = idx_rec["segments"][c["seg_i"]]
    route = seg.get("route_latlons")
    if not route or len(route) < 2:
        return False, "no_route"

    # observations
    obs_imgs = []
    for k in range(n_obs):
        ts = float(ep.video_ts[idx - k * obs_stride])
        img = frames.get(ts)
        if img is None:
            return False, "missing_frame"
        obs_imgs.append(_resize_obs(img, size, obs_resize))

    # GT waypoints — EKF position/heading, body frame, metres
    seg_end = int(seg["qa"]["end"])
    wp_idx = np.minimum(idx + np.array(wp_offsets, dtype=np.int64), seg_end - 1)
    wp_m = ep.to_body(idx, wp_idx)                       # (H, 2)
    clipped = bool(np.any(np.abs(wp_m) > norm_m))
    wp_norm = np.clip(wp_m / norm_m, -1.0, 1.0)

    # waypoint sanity: significant backward motion is a geometry error
    if wp_m[-1, 0] < -1.0:
        return False, "wp_backward"

    # route direction + map
    lat, lon = ep.latlon(idx)
    to_xy = _local_metric(ep.lat0, ep.lon0)
    # to_xy is anchored at frame 0's lat/lon → shift into the EKF pos frame
    route_xy = to_xy([p[0] for p in route], [p[1] for p in route]) + ep.pos[0]
    rd = _route_direction(route_xy, ep.pos[idx], float(ep.heading[idx]),
                          route_lookahead_m)

    # map↔GT consistency: if the rendered route and the demonstrated motion
    # disagree (robot walked a path missing from OSM), the sample teaches the
    # model to contradict its own map input — drop it.
    wp_dir = math.atan2(wp_m[-1, 1], wp_m[-1, 0])
    rw_angle = abs((wp_dir - rd + math.pi) % (2 * math.pi) - math.pi)
    if math.degrees(rw_angle) > route_wp_max_deg:
        return False, "route_wp_conflict"

    # render the future part of the route only, starting at the robot's
    # *polyline* projection (segment-wise — vertex distance is meaningless on
    # sparse graph routes; slicing here also avoids the legacy
    # global-nearest-point bug on self-overlapping routes)
    j0, proj_pt, _, lateral = _project_to_polyline(route_xy, ep.pos[idx])
    if lateral > 15.0:
        return False, "robot_far_from_route"
    proj_ll = _xy_to_latlon(proj_pt - ep.pos[0], ep.lat0, ep.lon0)
    route_future = [proj_ll] + [tuple(p) for p in route[j0 + 1:]]
    if len(route_future) < 2:
        route_future = [tuple(p) for p in route[-2:]]
    goal_lat, goal_lon = route[-1]

    map_pil = renderer.render(
        lat=float(lat), lon=float(lon), heading_deg=ep.compass_deg(idx),
        route_latlons=[tuple(p) for p in route_future],
        goal_lat=float(goal_lat), goal_lon=float(goal_lon),
    )
    if map_pil is None:
        return False, "map_render_failed"

    sdir = split_dir / _sample_name(ep, c)
    sdir.mkdir(parents=True, exist_ok=True)
    for oi, img in enumerate(obs_imgs):
        cv2.imwrite(str(sdir / f"obs_{oi}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    map_pil.save(sdir / "map.png")

    meta = {
        "source": "frodobots7k",
        "ride_id": ep.ride_id,
        "episode": c["episode"],
        "seg_i": c["seg_i"],
        "frame_idx": idx,
        "lat": float(lat), "lon": float(lon),
        "heading_rad_enu": float(ep.heading[idx]),
        "heading_deg_compass": ep.compass_deg(idx),
        "goal_lat": float(goal_lat), "goal_lon": float(goal_lon),
        "gt_waypoints": [[round(float(x), 5) for x in w] for w in wp_norm],
        "gt_waypoints_m": [[round(float(x), 4) for x in w] for w in wp_m],
        "waypoint_norm_m": norm_m,
        "wp_clipped": clipped,
        "route_direction": round(rd, 5),
        "route_wp_angle_deg": round(math.degrees(rw_angle), 1),
        "route_lateral_m": round(lateral, 2),
        "map_mode": "rgb",
        "labels": {k: c.get(k) for k in
                   ["maneuver", "dh_deg", "near_intersection",
                    "env_density", "scene", "difficulty", "speed_ms"]},
        "qa": {"segment": seg["qa"], "route": seg["route_qa"]},
    }
    (sdir / "meta.json").write_text(json.dumps(meta, indent=2))
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Stage: report
# ─────────────────────────────────────────────────────────────────────────────

def stage_report(cfg: dict) -> None:
    from PIL import Image, ImageDraw

    out_root = Path(cfg["paths"]["output_root"])
    rep_dir = out_root / "report"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    lines = ["# FrodoBots-7k dataset build report", ""]
    for name in ["selection_stats.json", "build_stats.json"]:
        p = out_root / name
        if p.exists():
            lines += [f"## {name}", "```json", p.read_text(), "```", ""]

    for split in ["train", "val"]:
        split_dir = out_root / split
        if not split_dir.exists():
            continue
        samples = sorted(d for d in split_dir.iterdir() if (d / "meta.json").exists())
        lines += [f"## {split}: {len(samples)} samples", ""]
        if not samples:
            continue

        k = min(cfg["report"]["contact_sheet_samples"], len(samples))
        pick = [samples[i] for i in sorted(rng.choice(len(samples), k, replace=False))]
        cols = 8
        cell = 224
        rows = (k + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell * 2, rows * cell), (30, 30, 30))
        for i, sdir in enumerate(pick):
            meta = json.loads((sdir / "meta.json").read_text())
            obs = Image.open(sdir / "obs_0.png")
            mp = Image.open(sdir / "map.png").convert("RGB")
            # overlay GT waypoints on the map (body frame → map px; heading-up:
            # +x fwd = up, +y left = left; crop_ratio scales m→px)
            zoom_mpp = (156543.03392 * math.cos(math.radians(meta["lat"]))
                        / (2 ** cfg["map"]["tile_zoom"]))
            px_per_m = 1.0 / zoom_mpp / cfg["map"]["crop_ratio"]
            dr = ImageDraw.Draw(mp)
            cx = cy = cell // 2
            for wx, wy in meta["gt_waypoints_m"]:
                u = cx - wy * px_per_m
                v = cy - wx * px_per_m
                dr.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(255, 200, 0))
            dr.text((4, 4), meta["labels"]["maneuver"], fill=(255, 255, 0))
            r, ccol = divmod(i, cols)
            sheet.paste(obs.resize((cell, cell)), (ccol * cell * 2, r * cell))
            sheet.paste(mp, (ccol * cell * 2 + cell, r * cell))
        sheet_path = rep_dir / f"contact_sheet_{split}.png"
        sheet.save(sheet_path)
        lines += [f"![{split}](report/contact_sheet_{split}.png)", ""]

        # label composition from meta files (ground truth of what was built)
        comp_m, comp_e, comp_s = Counter(), Counter(), Counter()
        n_clip = n_int = 0
        laterals = []
        for sdir in samples:
            meta = json.loads((sdir / "meta.json").read_text())
            comp_m[meta["labels"]["maneuver"]] += 1
            comp_e[meta["labels"]["env_density"]] += 1
            comp_s[meta["labels"].get("scene", "other")] += 1
            n_clip += meta["wp_clipped"]
            n_int += meta["labels"]["near_intersection"]
            if "route_lateral_m" in meta:
                laterals.append(meta["route_lateral_m"])
        lat_stats = ""
        if laterals:
            lat_stats = (f"med {np.median(laterals):.1f} m / "
                         f"p90 {np.percentile(laterals, 90):.1f} m / "
                         f"max {max(laterals):.1f} m")
        lines += [
            f"- maneuver: {dict(comp_m)}",
            f"- scene: {dict(comp_s)}",
            f"- env_density: {dict(comp_e)}",
            f"- near_intersection: {n_int} ({n_int / len(samples):.1%})",
            f"- wp_clipped: {n_clip} ({n_clip / len(samples):.1%})",
            f"- route_lateral_m (robot↔route offset): {lat_stats}", "",
        ]

    (out_root / "report.md").write_text("\n".join(lines))
    print(f"[report] → {out_root / 'report.md'}")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=_REPO / "configs" / "frodo7k.yaml")
    ap.add_argument("--stage", choices=["index", "select", "build", "report", "all"],
                    required=True)
    ap.add_argument("--episodes", type=str, default=None,
                    help="episode range 'start:end' for index stage")
    ap.add_argument("--limit", type=int, default=None,
                    help="max episodes for index stage")
    args = ap.parse_args()

    cfg = load_config(args.config)
    episodes = None
    if args.episodes:
        a, b = args.episodes.split(":")
        episodes = list(range(int(a), int(b)))

    stages = ([args.stage] if args.stage != "all"
              else ["index", "select", "build", "report"])
    for st in stages:
        if st == "index":
            stage_index(cfg, episodes, args.limit)
        elif st == "select":
            stage_select(cfg)
        elif st == "build":
            stage_build(cfg)
        elif st == "report":
            stage_report(cfg)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
