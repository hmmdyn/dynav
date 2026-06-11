"""Tests for sequence-consistent observation augmentation."""

import torch
from PIL import Image

from dynav.data.transforms import (
    ConsistentObsTransform,
    get_eval_transforms,
    get_obs_train_transforms,
)


def _frames(n: int = 4, size: int = 64) -> list[Image.Image]:
    """n identical mid-gray frames."""
    return [Image.new("RGB", (size, size), (128, 100, 90)) for _ in range(n)]


class TestConsistentObsTransform:
    def test_output_shape(self) -> None:
        tf = get_obs_train_transforms(image_size=224)
        out = tf(_frames())
        assert out.shape == (4, 3, 224, 224)

    def test_marker_attribute(self) -> None:
        assert getattr(get_obs_train_transforms(), "consistent_sequence", False)
        assert not getattr(get_eval_transforms(), "consistent_sequence", False)

    def test_identical_frames_get_identical_outputs(self) -> None:
        """Same input frames must receive the SAME sampled augmentation —
        per-frame independent jitter would make these differ."""
        tf = ConsistentObsTransform(image_size=64)
        out = tf(_frames())
        for i in range(1, 4):
            assert torch.equal(out[0], out[i])

    def test_stochastic_across_calls(self) -> None:
        """Different calls should (almost surely) sample different params."""
        tf = ConsistentObsTransform(image_size=64)
        torch.manual_seed(0)
        a = tf(_frames())
        b = tf(_frames())
        assert not torch.equal(a, b)
