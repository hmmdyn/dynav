"""Image transforms for Map Navigation Model.

Observation and map images require different augmentation strategies:

- Observation images (egocentric camera): ColorJitter models real-world
  lighting variation; mild RandomAffine models robot vibration/pitch.
  **Augmentation parameters are sampled ONCE per sample and applied
  identically to all obs frames** (ConsistentObsTransform). Per-frame
  independent sampling corrupts the inter-frame motion signal: at 0.5 s
  stride the real ego-motion shift is the same order as the augmentation
  jitter, so a model trained with independent jitter learns to ignore
  motion cues — exactly the speed information the time-parameterized
  waypoint GT requires.
- Map images (OSM tile API): Color is deterministic across all sessions
  (Carto voyager_nolabels always renders identical colors), so ColorJitter
  is not applied. Only geometric augmentation is used to simulate real-world
  sensor noise:
    - Rotation  ±MAP_AUG_DEGREES   → IMU/compass heading error
    - Translate ±MAP_AUG_TRANSLATE → GPS position error
  At zoom=19, 1 px ≈ 0.5 m on the 224px output; translate=0.03 ≈ ±7 px ≈ ±3 m.

Public API
----------
get_obs_train_transforms  : sequence-consistent augmented pipeline (train).
get_map_train_transforms  : geometry-only augmentation for map images (train).
get_eval_transforms       : no-augmentation pipeline for both modalities (val/inference).
"""

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF

# ImageNet statistics (used by pretrained EfficientNet-B0)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Map augmentation parameters — calibrated to real-world sensor noise
_MAP_AUG_DEGREES   = 10          # heading error: ±10°
_MAP_AUG_TRANSLATE = (0.03, 0.03)  # GPS error: ±3% ≈ ±7 px ≈ ±3 m at zoom=19


class ConsistentObsTransform:
    """Sequence-consistent augmentation for an obs frame stack.

    Samples ColorJitter and affine parameters once per call and applies them
    identically to every frame, preserving relative inter-frame motion (the
    model's only source of ego-speed). Camera vibration/lighting changes are
    shared across a 1.5 s window in reality, so shared parameters are also
    the physically correct noise model.

    Call with a list of PIL Images → returns a (N, 3, H, W) float tensor.
    """

    consistent_sequence = True   # dataset dispatches on this attribute

    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size
        self.jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05,
        )
        self.degrees = (-5.0, 5.0)
        self.translate = (0.04, 0.04)
        self.scale = (0.95, 1.05)

    def __call__(self, imgs: list) -> torch.Tensor:
        s = self.image_size
        # Sample shared parameters once
        fn_idx, b, c, sat, h = transforms.ColorJitter.get_params(
            self.jitter.brightness, self.jitter.contrast,
            self.jitter.saturation, self.jitter.hue,
        )
        angle, translations, scale, shear = transforms.RandomAffine.get_params(
            list(self.degrees), self.translate, self.scale, None, [s, s],
        )

        out = []
        for img in imgs:
            img = TF.resize(img, [s, s])
            for fn_id in fn_idx:
                if fn_id == 0 and b is not None:
                    img = TF.adjust_brightness(img, b)
                elif fn_id == 1 and c is not None:
                    img = TF.adjust_contrast(img, c)
                elif fn_id == 2 and sat is not None:
                    img = TF.adjust_saturation(img, sat)
                elif fn_id == 3 and h is not None:
                    img = TF.adjust_hue(img, h)
            img = TF.affine(img, angle=angle, translate=list(translations),
                            scale=scale, shear=list(shear))
            img = TF.normalize(TF.to_tensor(img), _IMAGENET_MEAN, _IMAGENET_STD)
            out.append(img)
        return torch.stack(out)   # (N, 3, H, W)


def get_obs_train_transforms(image_size: int = 224) -> ConsistentObsTransform:
    """Sequence-consistent augmented transform for obs frames (training only).

    Applies color jitter (lighting variation) and mild affine augmentation
    (robot vibration / pitch), with parameters shared across the frame stack
    — see ConsistentObsTransform for why independence is harmful.

    Args:
        image_size: Target square resolution (H = W).

    Returns:
        ``ConsistentObsTransform`` converting a list of PIL Images to a
        normalized float tensor of shape (N, 3, image_size, image_size).
    """
    return ConsistentObsTransform(image_size)


def get_map_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Geometry-only augmented transform for OSM map images (training only).

    Color augmentation is intentionally excluded: the map is rendered by a
    deterministic tile API (Carto voyager_nolabels) and always produces the
    same colors. Augmenting colors would corrupt the semantic meaning of the
    route overlay (red line) and road colors.

    Only geometric augmentation is applied to simulate sensor noise:
        - Rotation  ±10°       : IMU/compass heading error
        - Translate ±3%        : GPS position error (~±3 m at zoom=19)

    Pipeline:
        Resize → RandomAffine → ToTensor → Normalize

    Args:
        image_size: Target square resolution (H = W).

    Returns:
        ``transforms.Compose`` converting a PIL Image to a normalized
        float tensor of shape (3, image_size, image_size).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomAffine(
            degrees=_MAP_AUG_DEGREES,
            translate=_MAP_AUG_TRANSLATE,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """No-augmentation transform for validation and inference (both modalities).

    Args:
        image_size: Target square resolution (H = W).

    Returns:
        ``transforms.Compose`` converting a PIL Image to a normalized
        float tensor of shape (3, image_size, image_size).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
