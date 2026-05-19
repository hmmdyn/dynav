"""Image transforms for Map Navigation Model.

Observation and map images require different augmentation strategies:

- Observation images (egocentric camera): ColorJitter models real-world
  lighting variation; mild RandomAffine models robot vibration/pitch.
- Map images (OSM tile API): Color is deterministic across all sessions
  (Carto voyager_nolabels always renders identical colors), so ColorJitter
  is not applied. Only geometric augmentation is used to simulate real-world
  sensor noise:
    - Rotation  ±MAP_AUG_DEGREES   → IMU/compass heading error
    - Translate ±MAP_AUG_TRANSLATE → GPS position error
  At zoom=19, 1 px ≈ 0.5 m on the 224px output; translate=0.03 ≈ ±7 px ≈ ±3 m.

Public API
----------
get_obs_train_transforms  : augmented pipeline for observation images (train).
get_map_train_transforms  : geometry-only augmentation for map images (train).
get_eval_transforms       : no-augmentation pipeline for both modalities (val/inference).
"""

from torchvision import transforms

# ImageNet statistics (used by pretrained EfficientNet-B0)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Map augmentation parameters — calibrated to real-world sensor noise
_MAP_AUG_DEGREES   = 10          # heading error: ±10°
_MAP_AUG_TRANSLATE = (0.03, 0.03)  # GPS error: ±3% ≈ ±7 px ≈ ±3 m at zoom=19


def get_obs_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Augmented transform for egocentric observation images (training only).

    Applies color jitter (lighting variation) and mild affine augmentation
    (robot vibration / pitch). Color augmentation is appropriate here because
    the physical camera is subject to real-world illumination changes.

    Pipeline:
        Resize → ColorJitter → RandomAffine → ToTensor → Normalize

    Args:
        image_size: Target square resolution (H = W).

    Returns:
        ``transforms.Compose`` converting a PIL Image to a normalized
        float tensor of shape (3, image_size, image_size).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.04, 0.04),
            scale=(0.95, 1.05),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


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
