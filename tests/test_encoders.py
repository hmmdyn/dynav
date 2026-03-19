"""Tests for VisualEncoder and MapEncoder.

Run with: pytest tests/test_encoders.py -v
"""

import pytest
import torch
import torch.nn as nn

from dynav.models.encoders import MapEncoder, VisualEncoder


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def visual_encoder() -> VisualEncoder:
    """VisualEncoder with pretrained weights, token_dim=256, n_obs=4."""
    return VisualEncoder(token_dim=256, n_obs=4, pretrained=True)


@pytest.fixture(scope="module")
def map_encoder() -> MapEncoder:
    """MapEncoder with pretrained weights, token_dim=256."""
    return MapEncoder(token_dim=256, pretrained=True)


# ── VisualEncoder tests ────────────────────────────────────────────────────────

class TestVisualEncoder:
    """Tests for VisualEncoder."""

    def test_output_shape(self, visual_encoder: VisualEncoder) -> None:
        """Output shape must be (B, N_obs, token_dim) = (2, 4, 256)."""
        x = torch.randn(2, 4, 3, 224, 224)  # (B, N_obs, C, H, W)
        with torch.no_grad():
            tokens = visual_encoder(x)       # (B, N_obs, token_dim)
        assert tokens.shape == (2, 4, 256), (
            f"Expected (2, 4, 256), got {tuple(tokens.shape)}"
        )

    def test_pretrained_weights_loaded(self, visual_encoder: VisualEncoder) -> None:
        """Pretrained backbone must have non-zero weights (not default init)."""
        # EfficientNet-B0 first conv weight — pretrained values are non-trivial
        first_conv_weight = next(visual_encoder.features.parameters())
        # A randomly initialized weight would have std ≈ 0.02–0.05;
        # pretrained weights have structured non-zero values.
        assert first_conv_weight.abs().max().item() > 0.0

    def test_freeze_sets_requires_grad_false(self, visual_encoder: VisualEncoder) -> None:
        """After freeze_backbone(), backbone params must have requires_grad=False."""
        visual_encoder.freeze_backbone()
        for param in visual_encoder.features.parameters():
            assert not param.requires_grad, "features param should be frozen"
        for param in visual_encoder.avgpool.parameters():
            assert not param.requires_grad, "avgpool param should be frozen"

    def test_unfreeze_sets_requires_grad_true(self, visual_encoder: VisualEncoder) -> None:
        """After unfreeze_backbone(), backbone params must have requires_grad=True."""
        visual_encoder.unfreeze_backbone()
        for param in visual_encoder.features.parameters():
            assert param.requires_grad, "features param should be unfrozen"
        for param in visual_encoder.avgpool.parameters():
            assert param.requires_grad, "avgpool param should be unfrozen"

    def test_proj_and_pos_enc_always_trainable(self, visual_encoder: VisualEncoder) -> None:
        """Projection head and temporal pos enc must remain trainable regardless of freeze."""
        visual_encoder.freeze_backbone()
        assert visual_encoder.proj.weight.requires_grad
        assert visual_encoder.temporal_pos_enc.requires_grad
        visual_encoder.unfreeze_backbone()  # restore state for other tests


# ── MapEncoder tests ───────────────────────────────────────────────────────────

class TestMapEncoder:
    """Tests for MapEncoder."""

    def test_output_shape(self, map_encoder: MapEncoder) -> None:
        """Output shape must be (B, 49, token_dim) = (2, 49, 256)."""
        x = torch.randn(2, 3, 224, 224)  # (B, C, H, W)
        with torch.no_grad():
            tokens = map_encoder(x)       # (B, 49, token_dim)
        assert tokens.shape == (2, 49, 256), (
            f"Expected (2, 49, 256), got {tuple(tokens.shape)}"
        )

    def test_pretrained_weights_loaded(self, map_encoder: MapEncoder) -> None:
        """Pretrained backbone must have non-zero weights (not default init)."""
        first_conv_weight = next(map_encoder.features.parameters())
        assert first_conv_weight.abs().max().item() > 0.0

    def test_freeze_sets_requires_grad_false(self, map_encoder: MapEncoder) -> None:
        """After freeze_backbone(), backbone params must have requires_grad=False."""
        map_encoder.freeze_backbone()
        for param in map_encoder.features.parameters():
            assert not param.requires_grad, "features param should be frozen"

    def test_unfreeze_sets_requires_grad_true(self, map_encoder: MapEncoder) -> None:
        """After unfreeze_backbone(), backbone params must have requires_grad=True."""
        map_encoder.unfreeze_backbone()
        for param in map_encoder.features.parameters():
            assert param.requires_grad, "features param should be unfrozen"

    def test_proj_and_pos_enc_always_trainable(self, map_encoder: MapEncoder) -> None:
        """Projection head and 2D pos enc must remain trainable regardless of freeze."""
        map_encoder.freeze_backbone()
        assert map_encoder.proj.weight.requires_grad
        assert map_encoder.pos_enc_2d.requires_grad
        map_encoder.unfreeze_backbone()  # restore state for other tests


# ── MapEncoder positional encoding type tests ──────────────────────────────────

class TestMapEncoderPosEncType:
    """Verify learnable and sinusoidal pos_enc_type options."""

    def test_learnable_output_shape(self) -> None:
        """Learnable pos enc: output must be (B, 49, 256)."""
        enc = MapEncoder(token_dim=256, pretrained=False, pos_enc_type="learnable")
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            tokens = enc(x)
        assert tokens.shape == (2, 49, 256), (
            f"Expected (2, 49, 256), got {tuple(tokens.shape)}"
        )

    def test_sinusoidal_output_shape(self) -> None:
        """Sinusoidal pos enc: output must be (B, 49, 256)."""
        enc = MapEncoder(token_dim=256, pretrained=False, pos_enc_type="sinusoidal")
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            tokens = enc(x)
        assert tokens.shape == (2, 49, 256), (
            f"Expected (2, 49, 256), got {tuple(tokens.shape)}"
        )

    def test_learnable_pos_enc_is_parameter(self) -> None:
        """Learnable pos enc must be an nn.Parameter (requires_grad=True)."""
        enc = MapEncoder(token_dim=256, pretrained=False, pos_enc_type="learnable")
        assert isinstance(enc.pos_enc_2d, nn.Parameter)
        assert enc.pos_enc_2d.requires_grad

    def test_sinusoidal_pos_enc_not_parameter(self) -> None:
        """Sinusoidal pos enc must be a buffer — not a learnable parameter."""
        enc = MapEncoder(token_dim=256, pretrained=False, pos_enc_type="sinusoidal")
        # Must not appear in named_parameters()
        param_names = {name for name, _ in enc.named_parameters()}
        assert "pos_enc_2d" not in param_names, (
            "pos_enc_2d should be a buffer, not a learnable parameter"
        )
        # Must be accessible as an attribute (registered buffer)
        assert hasattr(enc, "pos_enc_2d")
        assert not enc.pos_enc_2d.requires_grad

    def test_invalid_pos_enc_type_raises(self) -> None:
        """Unknown pos_enc_type must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown pos_enc_type"):
            MapEncoder(token_dim=256, pretrained=False, pos_enc_type="rotary")


# ── Independence test ──────────────────────────────────────────────────────────

class TestEncoderIndependence:
    """Verify that VisualEncoder and MapEncoder have entirely separate weights."""

    def test_backbones_are_independent_instances(
        self,
        visual_encoder: VisualEncoder,
        map_encoder: MapEncoder,
    ) -> None:
        """Backbone parameter tensors must be distinct objects (separate weights)."""
        vis_params = set(id(p) for p in visual_encoder.features.parameters())
        map_params = set(id(p) for p in map_encoder.features.parameters())
        overlap = vis_params & map_params
        assert len(overlap) == 0, (
            f"{len(overlap)} parameter(s) are shared between visual and map encoders"
        )

    def test_gradient_update_does_not_affect_other_encoder(
        self,
        visual_encoder: VisualEncoder,
        map_encoder: MapEncoder,
    ) -> None:
        """A gradient step on VisualEncoder must not change MapEncoder weights."""
        # Snapshot a map encoder weight before any update
        map_param = next(map_encoder.features.parameters())
        map_weight_before = map_param.data.clone()

        # Forward + backward on visual encoder only
        x = torch.randn(1, 4, 3, 224, 224)
        tokens = visual_encoder(x)
        loss = tokens.sum()
        loss.backward()

        # Map encoder weight must be unchanged
        assert torch.equal(map_param.data, map_weight_before), (
            "MapEncoder weights changed after backward on VisualEncoder"
        )

        # Clean up gradients
        visual_encoder.zero_grad()
