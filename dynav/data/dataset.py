"""Dataset class for Map Navigation Model."""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


# ── Production dataset ─────────────────────────────────────────────────────────

class DyNavDataset(Dataset):
    """Reads real navigation data from a structured directory on disk.

    Expected layout::

        data_dir/
          {split}/
            sample_000000/
              obs_0.png      # front camera, current frame
              obs_1.png      # front camera, past frame 1
              obs_2.png      # front camera, past frame 2
              obs_3.png      # front camera, past frame 3
              map.png        # OSM top-down map with route overlay
              meta.json      # {"gt_waypoints": [[dx,dy]×H], "route_direction": float}
            sample_000001/
              ...

    Args:
        data_dir: Root data directory (contains ``train/``, ``val/``, etc.).
        split: Dataset split name (``"train"``, ``"val"``, ``"test"``).
        transform: Transform applied to each observation image (PIL → Tensor).
            Defaults to ``get_train_transforms`` for train split and
            ``get_eval_transforms`` otherwise.
        map_transform: Transform applied to the map image. Defaults to
            ``get_eval_transforms`` (no augmentation — map is deterministic).
        image_size: Resize target passed to default transforms when
            ``transform`` is None.
        default_waypoint_norm_m: Fallback normalization radius (meters) for
            samples whose meta.json predates the ``waypoint_norm_m`` field
            (pre-frodo7k datasets used 2.5 m). Used to report ADE/FDE in meters.
    """

    _OBS_FILES = ["obs_0.png", "obs_1.png", "obs_2.png", "obs_3.png"]
    _MAP_FILE  = "map.png"
    _META_FILE = "meta.json"

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transform=None,
        map_transform=None,
        image_size: int = 224,
        default_waypoint_norm_m: float = 2.5,
    ) -> None:
        self.default_waypoint_norm_m = default_waypoint_norm_m
        from dynav.data.transforms import (
            get_eval_transforms,
            get_map_train_transforms,
            get_obs_train_transforms,
        )

        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Dataset split not found: {split_dir}\n"
                "Expected: data_dir/{split}/sample_XXXXXX/"
                "{obs_0..3.png, map.png, meta.json}"
            )

        self.samples = sorted(p for p in split_dir.iterdir() if p.is_dir())
        if not self.samples:
            raise ValueError(f"No sample directories found in {split_dir}")

        self.transform = (
            transform
            or (get_obs_train_transforms(image_size) if split == "train"
                else get_eval_transforms(image_size))
        )
        self.map_transform = (
            map_transform
            or (get_map_train_transforms(image_size) if split == "train"
                else get_eval_transforms(image_size))
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load and return one sample.

        Returns:
            Dict with keys:
                - ``observations``: (N_obs, 3, H, W) float tensor.
                - ``map_image``:    (3, H, W) float tensor.
                - ``gt_waypoints``: (H, 2) float tensor, normalized to [-1, 1].
                - ``route_direction``: scalar tensor (radians, body frame).
                - ``waypoint_norm_m``: scalar tensor — normalization radius in
                  meters (for de-normalizing metrics to ADE/FDE in m).
                - ``maneuver``: str label for stratified metrics
                  (``"unknown"`` if the dataset has no labels).
        """
        from PIL import Image  # lazy import — not needed for DummyDyNavDataset

        sample_dir = self.samples[idx]

        obs_tensors = []
        for fname in self._OBS_FILES:
            img = Image.open(sample_dir / fname).convert("RGB")
            obs_tensors.append(self.transform(img))              # (3, H, W)
        observations = torch.stack(obs_tensors, dim=0)           # (N_obs, 3, H, W)

        map_pil  = Image.open(sample_dir / self._MAP_FILE).convert("RGB")
        map_image = self.map_transform(map_pil)                  # (3, H, W)

        with open(sample_dir / self._META_FILE) as f:
            meta = json.load(f)

        gt_waypoints    = torch.tensor(meta["gt_waypoints"], dtype=torch.float32)
        route_direction = torch.tensor(meta["route_direction"], dtype=torch.float32)

        norm_m   = meta.get("waypoint_norm_m", self.default_waypoint_norm_m)
        maneuver = meta.get("labels", {}).get("maneuver", "unknown")

        return {
            "observations":    observations,     # (N_obs, 3, H, W)
            "map_image":       map_image,         # (3, H, W)
            "gt_waypoints":    gt_waypoints,      # (H, 2)
            "route_direction": route_direction,   # scalar
            "waypoint_norm_m": torch.tensor(norm_m, dtype=torch.float32),  # scalar
            "maneuver":        maneuver,          # str (collates to list[str])
        }

