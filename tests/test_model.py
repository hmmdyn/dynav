"""Tests for dynavModel end-to-end assembly.

Run with: pytest tests/test_model.py -v
"""

import math

import pytest
import torch
from omegaconf import OmegaConf

from dynav.models.map_nav_model import dynavModel

# ── Config helpers ─────────────────────────────────────────────────────────────

def _load_cfg(decoder_type: str = "cross_attention") -> object:
    """Load default config and override decoder type."""
    cfg = OmegaConf.load("configs/default.yaml")
    cfg.decoder.type = decoder_type
    return cfg


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cross_model() -> dynavModel:
    return dynavModel.from_config(_load_cfg("cross_attention"))


@pytest.fixture(scope="module")
def self_model() -> dynavModel:
    return dynavModel.from_config(_load_cfg("self_attention"))


def _dummy_inputs(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (observations, map_image) dummy tensors."""
    obs = torch.randn(batch, 4, 3, 224, 224)   # (B, N_obs, C, H, W)
    mp  = torch.randn(batch, 3, 224, 224)       # (B, C, H, W)
    return obs, mp


# ── Forward pass shape tests ───────────────────────────────────────────────────

class TestForwardShape:
    """Verify output shapes for both decoder types."""

    @pytest.mark.parametrize("model_fixture", ["cross_model", "self_model"])
    def test_waypoints_shape(self, request: pytest.FixtureRequest, model_fixture: str) -> None:
        """Waypoints output must be (B, H, 2) = (2, 5, 2)."""
        model: dynavModel = request.getfixturevalue(model_fixture)
        obs, mp = _dummy_inputs()
        with torch.no_grad():
            out = model(obs, mp)
        assert out["waypoints"].shape == (2, 5, 2), (
            f"Expected (2, 5, 2), got {tuple(out['waypoints'].shape)}"
        )

    def test_cross_attention_weights_shape(self, cross_model: dynavModel) -> None:
        """Cross-attention weights must be a list of 4 tensors, each (B, N_o, N_m)."""
        obs, mp = _dummy_inputs()
        with torch.no_grad():
            out = cross_model(obs, mp, return_attention=True)
        attn = out["attention_weights"]
        assert attn is not None
        assert len(attn) == 4
        for i, w in enumerate(attn):
            assert w.shape == (2, 4, 49), (
                f"Layer {i}: expected (2, 4, 49), got {tuple(w.shape)}"
            )

    def test_attention_none_by_default(self, cross_model: dynavModel) -> None:
        """Without return_attention, attention_weights must be None."""
        obs, mp = _dummy_inputs()
        with torch.no_grad():
            out = cross_model(obs, mp)
        assert out["attention_weights"] is None

    def test_self_attention_weights_always_none(self, self_model: dynavModel) -> None:
        """SelfAttentionDecoder always returns None for attention_weights."""
        obs, mp = _dummy_inputs()
        with torch.no_grad():
            out = self_model(obs, mp, return_attention=True)
        assert out["attention_weights"] is None


# ── Parameter count tests ──────────────────────────────────────────────────────

class TestParameterCount:
    """Verify model size is within expected range."""

    @pytest.mark.parametrize("model_fixture", ["cross_model", "self_model"])
    def test_total_params_in_range(
        self, request: pytest.FixtureRequest, model_fixture: str
    ) -> None:
        """Total trainable parameters must be between 10M and 20M."""
        model: dynavModel = request.getfixturevalue(model_fixture)
        counts = model.count_parameters()
        total = counts["total"]
        assert 10_000_000 <= total <= 20_000_000, (
            f"Total params {total:,} outside expected range [10M, 20M]"
        )

    def test_components_sum_to_total(self, cross_model: dynavModel) -> None:
        """Component parameter counts must sum to total."""
        counts = cross_model.count_parameters()
        component_sum = (
            counts["visual_encoder"]
            + counts["map_encoder"]
            + counts["decoder"]
            + counts["waypoint_head"]
        )
        assert component_sum == counts["total"]

    def test_encoders_dominate(self, cross_model: dynavModel) -> None:
        """EfficientNet backbones should account for the majority of parameters."""
        counts = cross_model.count_parameters()
        encoder_params = counts["visual_encoder"] + counts["map_encoder"]
        assert encoder_params > counts["total"] * 0.5


# ── Freeze / unfreeze tests ────────────────────────────────────────────────────

class TestFreezeUnfreeze:
    """Verify freeze/unfreeze affects encoder backbone parameters."""

    def test_freeze_reduces_trainable_params(self, cross_model: dynavModel) -> None:
        """After freeze, total trainable params must decrease."""
        counts_before = cross_model.count_parameters()["total"]
        cross_model.freeze_encoders()
        counts_after = cross_model.count_parameters()["total"]
        assert counts_after < counts_before
        cross_model.unfreeze_encoders()  # restore

    def test_unfreeze_restores_trainable_params(self, cross_model: dynavModel) -> None:
        """After unfreeze, total trainable params must match original count."""
        counts_full = cross_model.count_parameters()["total"]
        cross_model.freeze_encoders()
        cross_model.unfreeze_encoders()
        counts_restored = cross_model.count_parameters()["total"]
        assert counts_full == counts_restored


# ── Backward pass test ─────────────────────────────────────────────────────────

class TestBackwardPass:
    """Verify gradient flow through the full model."""

    @pytest.mark.parametrize("model_fixture", ["cross_model", "self_model"])
    def test_backward_succeeds(
        self, request: pytest.FixtureRequest, model_fixture: str
    ) -> None:
        """loss.backward() must succeed and produce gradients for all parameters."""
        model: dynavModel = request.getfixturevalue(model_fixture)
        model.zero_grad()

        obs, mp = _dummy_inputs()
        out = model(obs, mp)
        loss = out["waypoints"].sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_invalid_decoder_type_raises(self) -> None:
        """Unsupported decoder type must raise ValueError."""
        cfg = _load_cfg("cross_attention")
        cfg.decoder.type = "diffusion"
        with pytest.raises(ValueError, match="Unknown decoder type"):
            dynavModel(cfg)
