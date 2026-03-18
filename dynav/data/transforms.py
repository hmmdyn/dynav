"""Image transforms for Map Navigation Model.

get_train_transforms: augmented pipeline for training.
get_eval_transforms:  minimal pipeline for validation/inference.

Both pipelines normalize with ImageNet mean/std, compatible with the
pretrained EfficientNet-B0 backbone.
"""

from torchvision import transforms

# ImageNet statistics (used by pretrained EfficientNet-B0)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Build the training augmentation pipeline.

    Augmentations are deliberately mild — the robot's egocentric view and
    the map image carry precise spatial information that strong augmentation
    would corrupt.

    Pipeline:
        Resize → ColorJitter → RandomAffine → ToTensor → Normalize

    Args:
        image_size: Target square resolution (H = W).

    Returns:
        ``transforms.Compose`` that converts a PIL Image to a normalized
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


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Build the evaluation / inference transform pipeline.

    No augmentation — only resize and normalize.

    Args:
        image_size: Target square resolution (H = W).

    Returns:
        ``transforms.Compose`` that converts a PIL Image to a normalized
        float tensor of shape (3, image_size, image_size).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
