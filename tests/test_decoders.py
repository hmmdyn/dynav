"""Tests for CrossAttentionDecoder, SelfAttentionDecoder, and WaypointHead.

Run with: pytest tests/test_decoders.py -v
"""

from typing import Optional

import pytest
import torch

from dynav.models.decoders import (
    CrossAttentionDecoder,
    SelfAttentionDecoder,
    WaypointHead,
)

# ── Common dimensions (match configs/default.yaml) ────────────────────────────
TOKEN_DIM = 256
N_OBS = 4
N_MAP = 9
N_LAYERS = 4
N_HEADS = 4
D_FF = 512
HORIZON = 5
BATCH = 2


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cross_decoder() -> CrossAttentionDecoder:
    return CrossAttentionDecoder(
        token_dim=TOKEN_DIM, n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF
    )


@pytest.fixture(scope="module")
def self_decoder() -> SelfAttentionDecoder:
    return SelfAttentionDecoder(
        token_dim=TOKEN_DIM, n_obs=N_OBS, n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF
    )


@pytest.fixture(scope="module")
def waypoint_head() -> WaypointHead:
    return WaypointHead(
        token_dim=TOKEN_DIM, hidden_dim=128, prediction_horizon=HORIZON
    )


@pytest.fixture
def obs_tokens() -> torch.Tensor:
    return torch.randn(BATCH, N_OBS, TOKEN_DIM)  # (B, N_o, d)


@pytest.fixture
def map_tokens() -> torch.Tensor:
    return torch.randn(BATCH, N_MAP, TOKEN_DIM)  # (B, N_m, d)


# ── CrossAttentionDecoder tests ────────────────────────────────────────────────

class TestCrossAttentionDecoder:
    """Tests for CrossAttentionDecoder."""

    def test_context_output_shape(
        self,
        cross_decoder: CrossAttentionDecoder,
        obs_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
    ) -> None:
        """Context vector must be (B, d) = (2, 256)."""
        with torch.no_grad():
            context, _ = cross_decoder(obs_tokens, map_tokens)
        assert context.shape == (BATCH, TOKEN_DIM), (
            f"Expected ({BATCH}, {TOKEN_DIM}), got {tuple(context.shape)}"
        )

    def test_attention_weights_not_returned_by_default(
        self,
        cross_decoder: CrossAttentionDecoder,
        obs_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
    ) -> None:
        """Without return_attention, attn_weights must be None."""
        with torch.no_grad():
            _, attn = cross_decoder(obs_tokens, map_tokens, return_attention=False)
        assert attn is None

    def test_attention_weights_shape(
        self,
        cross_decoder: CrossAttentionDecoder,
        obs_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
    ) -> None:
        """Each layer's attention weights must be (B, N_o, N_m)."""
        with torch.no_grad():
            _, attn_list = cross_decoder(obs_tokens, map_tokens, return_attention=True)

        assert attn_list is not None
        assert len(attn_list) == N_LAYERS, (
            f"Expected {N_LAYERS} attention tensors, got {len(attn_list)}"
        )
        for i, attn in enumerate(attn_list):
            assert attn.shape == (BATCH, N_OBS, N_MAP), (
                f"Layer {i}: expected ({BATCH}, {N_OBS}, {N_MAP}), got {tuple(attn.shape)}"
            )

    def test_gradient_flow(
        self,
        cross_decoder: CrossAttentionDecoder,
        obs_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
    ) -> None:
        """All parameters must receive gradients after backward."""
        obs = obs_tokens.detach().requires_grad_(False)
        mp  = map_tokens.detach().requires_grad_(False)

        context, _ = cross_decoder(obs, mp)
        loss = context.sum()
        loss.backward()

        for name, param in cross_decoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ── SelfAttentionDecoder tests ─────────────────────────────────────────────────

class TestSelfAttentionDecoder:
    """Tests for SelfAttentionDecoder (drop-in replacement verification)."""

    def test_context_output_shape(
        self,
        self_decoder: SelfAttentionDecoder,
        obs_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
    ) -> None:
        """Context vector must be (B, d) = (2, 256) — same as CrossAttentionDecoder."""
        with torch.no_grad():
            context, _ = self_decoder(obs_tokens, map_tokens)
        assert context.shape == (BATCH, TOKEN_DIM), (
            f"Expected ({BATCH}, {TOKEN_DIM}), got {tuple(context.shape)}"
        )

    def test_attention_always_none(
        self,
        self_decoder: SelfAttentionDecoder,
        obs_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
    ) -> None:
        """SelfAttentionDecoder always returns None for attention (interface compat)."""
        with torch.no_grad():
            _, attn_true  = self_decoder(obs_tokens, map_tokens, return_attention=True)
            _, attn_false = self_decoder(obs_tokens, map_tokens, return_attention=False)
        assert attn_true  is None
        assert attn_false is None

    def test_drop_in_replacement_same_signature(
        self,
        cross_decoder: CrossAttentionDecoder,
        self_decoder: SelfAttentionDecoder,
        obs_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
    ) -> None:
        """Both decoders must accept the same args and return the same output shape."""
        with torch.no_grad():
            ctx_cross, _ = cross_decoder(obs_tokens, map_tokens)
            ctx_self,  _ = self_decoder(obs_tokens, map_tokens)
        assert ctx_cross.shape == ctx_self.shape

    def test_gradient_flow(
        self,
        self_decoder: SelfAttentionDecoder,
        obs_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
    ) -> None:
        """All parameters must receive gradients after backward."""
        obs = obs_tokens.detach().requires_grad_(False)
        mp  = map_tokens.detach().requires_grad_(False)

        context, _ = self_decoder(obs, mp)
        loss = context.sum()
        loss.backward()

        for name, param in self_decoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ── WaypointHead tests ─────────────────────────────────────────────────────────

class TestWaypointHead:
    """Tests for WaypointHead."""

    def test_waypoints_output_shape(
        self, waypoint_head: WaypointHead
    ) -> None:
        """Output must be (B, H, 2) = (2, 5, 2)."""
        context = torch.randn(BATCH, TOKEN_DIM)
        with torch.no_grad():
            waypoints = waypoint_head(context)
        assert waypoints.shape == (BATCH, HORIZON, 2), (
            f"Expected ({BATCH}, {HORIZON}, 2), got {tuple(waypoints.shape)}"
        )

    def test_waypoints_bounded(
        self, waypoint_head: WaypointHead
    ) -> None:
        """All waypoint values must lie in [-1, 1] (tanh output)."""
        context = torch.randn(BATCH, TOKEN_DIM) * 10  # large inputs stress-test tanh
        with torch.no_grad():
            waypoints = waypoint_head(context)
        assert waypoints.min().item() >= -1.0 - 1e-6
        assert waypoints.max().item() <=  1.0 + 1e-6

    def test_gradient_flow(
        self, waypoint_head: WaypointHead
    ) -> None:
        """All parameters must receive gradients after backward."""
        context = torch.randn(BATCH, TOKEN_DIM)
        waypoints = waypoint_head(context)
        loss = waypoints.sum()
        loss.backward()

        for name, param in waypoint_head.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_inherits_base_action_head(
        self, waypoint_head: WaypointHead
    ) -> None:
        """WaypointHead must be an instance of BaseActionHead."""
        from dynav.models.decoders import BaseActionHead
        assert isinstance(waypoint_head, BaseActionHead)
