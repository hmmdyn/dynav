"""Dataset classes for Map Navigation Model.

DyNavDataset: production dataset that reads from disk.
DummyDyNavDataset: in-memory random dataset for overfitting tests and
    pipeline debugging — no data files required.
"""

import json
import math
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
              obs_3.png      # rear camera, current frame
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
    ) -> None:
        from dynav.data.transforms import get_eval_transforms, get_train_transforms

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
            or (get_train_transforms(image_size) if split == "train"
                else get_eval_transforms(image_size))
        )
        self.map_transform = map_transform or get_eval_transforms(image_size)

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

        return {
            "observations":    observations,     # (N_obs, 3, H, W)
            "map_image":       map_image,         # (3, H, W)
            "gt_waypoints":    gt_waypoints,      # (H, 2)
            "route_direction": route_direction,   # scalar
        }


# ── Dummy dataset ──────────────────────────────────────────────────────────────

class DummyDyNavDataset(Dataset):
    """In-memory dummy dataset for overfitting tests and pipeline debugging.

    All tensors are pre-generated at construction time with a fixed seed, so
    the same data is returned across epochs (necessary for overfitting).

    Args:
        size: Number of samples in the dataset.
        n_obs: Number of observation frames per sample (N_obs). Should match
            ``obs_context_length + 2`` from config.
        image_size: Spatial resolution of observation and map images (H = W).
        horizon: Number of waypoints to predict (H).
        fixed_targets: If True, all samples share identical gt_waypoints
            (a straight-ahead trajectory) and route_direction=0. This makes
            overfitting trivially fast — use for sanity checks. If False,
            each sample gets independent random targets.
        seed: Random seed for reproducible data generation.

    Example:
        >>> ds = DummyDyNavDataset(size=10, fixed_targets=True)
        >>> sample = ds[0]
        >>> sample["observations"].shape   # (4, 3, 224, 224)
        >>> sample["gt_waypoints"].shape   # (5, 2)
    """

    def __init__(
        self,
        size: int = 100,
        n_obs: int = 4,
        image_size: int = 224,
        horizon: int = 5,
        fixed_targets: bool = False,
        seed: int = 42,
    ) -> None:
        self.size    = size
        self.n_obs   = n_obs
        self.horizon = horizon

        gen = torch.Generator().manual_seed(seed)

        # Pre-generate all inputs (fixed across epochs for reproducible overfitting)
        self._observations = torch.randn(
            size, n_obs, 3, image_size, image_size, generator=gen
        )                                                        # (N, N_obs, 3, H, W)
        self._map_images = torch.randn(
            size, 3, image_size, image_size, generator=gen
        )                                                        # (N, 3, H, W)

        if fixed_targets:
            # Straight-ahead trajectory: x increases uniformly, y = 0.
            # Route direction = 0 (forward = East in body frame).
            wps = torch.zeros(horizon, 2)
            wps[:, 0] = torch.linspace(0.1, 0.5, horizon)       # x: forward progress
            self._gt_waypoints    = wps.unsqueeze(0).repeat(size, 1, 1)  # (N, H, 2)
            self._route_direction = torch.zeros(size)            # (N,)
        else:
            self._gt_waypoints = torch.empty(
                size, horizon, 2, generator=gen
            ).uniform_(-0.8, 0.8)                                # (N, H, 2)
            self._route_direction = torch.empty(
                size, generator=gen
            ).uniform_(0, 2 * math.pi)                          # (N,)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return one pre-generated sample.

        Returns:
            Dict with keys ``observations``, ``map_image``, ``gt_waypoints``,
            ``route_direction``. Shapes match DyNavDataset.__getitem__.
        """
        return {
            "observations":    self._observations[idx],     # (N_obs, 3, H, W)
            "map_image":       self._map_images[idx],       # (3, H, W)
            "gt_waypoints":    self._gt_waypoints[idx],     # (H, 2)
            "route_direction": self._route_direction[idx],  # scalar
        }
