"""Tests for navigation loss functions and NavigationLoss class.

Run with: pytest tests/test_losses.py -v
"""

import math

import pytest
import torch
from omegaconf import OmegaConf

from dynav.losses.navigation_losses import (
    NavigationLoss,
    compute_direction_loss,
    compute_progress_loss,
    compute_smooth_loss,
    compute_waypoint_loss,
)

# ── Shared dims ────────────────────────────────────────────────────────────────
B, H = 4, 5


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.load("configs/default.yaml")


@pytest.fixture(scope="module")
def nav_loss(cfg) -> NavigationLoss:
    return NavigationLoss(cfg)


# ── compute_waypoint_loss ──────────────────────────────────────────────────────

class TestWaypointLoss:
    def test_output_is_scalar(self) -> None:
        pred = torch.randn(B, H, 2)
        gt   = torch.randn(B, H, 2)
        loss = compute_waypoint_loss(pred, gt)
        assert loss.shape == ()

    def test_zero_when_pred_equals_gt(self) -> None:
        gt = torch.randn(B, H, 2)
        loss = compute_waypoint_loss(gt, gt)
        assert loss.item() < 1e-6, f"Expected ~0 loss, got {loss.item()}"

    def test_positive_for_different_inputs(self) -> None:
        pred = torch.zeros(B, H, 2)
        gt   = torch.ones(B, H, 2)
        loss = compute_waypoint_loss(pred, gt)
        assert loss.item() > 0

    def test_value_range_reasonable(self) -> None:
        # With normalized waypoints in [-1, 1], L1 max per element is 2
        pred = torch.ones(B, H, 2)
        gt   = -torch.ones(B, H, 2)
        loss = compute_waypoint_loss(pred, gt)
        assert 0 < loss.item() <= 2.0 + 1e-6


# ── compute_direction_loss ─────────────────────────────────────────────────────

class TestDirectionLoss:
    def test_output_is_scalar(self) -> None:
        pred = torch.randn(B, H, 2)
        rdir = torch.zeros(B)
        loss = compute_direction_loss(pred, rdir)
        assert loss.shape == ()

    def test_range_zero_to_two(self) -> None:
        """1 - cos is always in [0, 2]."""
        for _ in range(20):
            pred = torch.randn(B, H, 2)
            rdir = torch.rand(B) * 2 * math.pi
            loss = compute_direction_loss(pred, rdir)
            assert -1e-6 <= loss.item() <= 2.0 + 1e-6

    def test_low_when_aligned(self) -> None:
        """Loss should be near 0 when trajectory direction matches route."""
        angle = math.pi / 4
        # Waypoints that all point in direction `angle`
        step = torch.tensor([[math.cos(angle), math.sin(angle)]])
        pred = step.expand(B, H, 2)  # (B, H, 2) all steps in same direction
        rdir = torch.full((B,), angle)
        loss = compute_direction_loss(pred, rdir)
        assert loss.item() < 0.05, f"Expected low loss, got {loss.item()}"

    def test_high_when_opposed(self) -> None:
        """Loss should be near 2 when trajectory is directly opposed to route."""
        angle = math.pi / 6
        step  = torch.tensor([[math.cos(angle), math.sin(angle)]])
        pred  = step.expand(B, H, 2)
        # Route direction is opposite
        rdir  = torch.full((B,), angle + math.pi)
        loss  = compute_direction_loss(pred, rdir)
        assert loss.item() > 1.5, f"Expected high loss, got {loss.item()}"


# ── compute_progress_loss ──────────────────────────────────────────────────────

class TestProgressLoss:
    def test_output_is_scalar(self) -> None:
        pred = torch.randn(B, H, 2)
        rdir = torch.zeros(B)
        loss = compute_progress_loss(pred, rdir)
        assert loss.shape == ()

    def test_negative_when_progress_made(self) -> None:
        """When all waypoints point along the route, progress loss is negative."""
        angle = 0.0
        step  = torch.tensor([[1.0, 0.0]])          # forward = East = route
        pred  = step.expand(B, H, 2)
        rdir  = torch.zeros(B)
        loss  = compute_progress_loss(pred, rdir)
        assert loss.item() < 0, f"Expected negative loss, got {loss.item()}"

    def test_positive_when_regressing(self) -> None:
        """When waypoints oppose the route, progress loss is positive."""
        step  = torch.tensor([[-1.0, 0.0]])         # backward
        pred  = step.expand(B, H, 2)
        rdir  = torch.zeros(B)                      # route is forward (East)
        loss  = compute_progress_loss(pred, rdir)
        assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"


# ── compute_smooth_loss ────────────────────────────────────────────────────────

class TestSmoothLoss:
    def test_output_is_scalar(self) -> None:
        pred = torch.randn(B, H, 2)
        loss = compute_smooth_loss(pred)
        assert loss.shape == ()

    def test_zero_for_constant_trajectory(self) -> None:
        """All-same waypoints have zero smoothness loss."""
        pred = torch.ones(B, H, 2)
        loss = compute_smooth_loss(pred)
        assert loss.item() < 1e-6, f"Expected ~0 smooth loss, got {loss.item()}"

    def test_lower_for_smooth_than_jittery(self) -> None:
        """A smooth trajectory must produce lower loss than a jittery one."""
        smooth = torch.zeros(B, H, 2)
        # Alternating ±1 along x — maximally jittery
        jittery = torch.zeros(B, H, 2)
        jittery[:, ::2, 0]  =  1.0
        jittery[:, 1::2, 0] = -1.0
        loss_smooth  = compute_smooth_loss(smooth)
        loss_jittery = compute_smooth_loss(jittery)
        assert loss_smooth.item() < loss_jittery.item()

    def test_nonnegative(self) -> None:
        for _ in range(10):
            pred = torch.randn(B, H, 2)
            assert compute_smooth_loss(pred).item() >= 0


# ── NavigationLoss ─────────────────────────────────────────────────────────────

class TestNavigationLoss:
    """Tests for the composite NavigationLoss class."""

    def _inputs(self):
        pred = torch.randn(B, H, 2).tanh()
        gt   = torch.randn(B, H, 2).tanh()
        rdir = torch.rand(B) * 2 * math.pi
        return pred, gt, rdir

    def test_returns_scalar_and_dict(self, nav_loss: NavigationLoss) -> None:
        pred, gt, rdir = self._inputs()
        total, loss_dict = nav_loss(pred, gt, rdir)
        assert total.shape == ()
        for key in ("loss/total", "loss/waypoint", "loss/direction",
                    "loss/progress", "loss/smooth"):
            assert key in loss_dict, f"Missing key '{key}' in loss_dict"

    def test_total_equals_weighted_sum(self, nav_loss: NavigationLoss) -> None:
        """total must equal L_wp + λ_dir*L_dir + λ_prog*L_prog + λ_sm*L_sm."""
        pred, gt, rdir = self._inputs()
        total, d = nav_loss(pred, gt, rdir)
        expected = (
            d["loss/waypoint"]
            + nav_loss.lambda_direction * d["loss/direction"]
            + nav_loss.lambda_progress  * d["loss/progress"]
            + nav_loss.lambda_smooth    * d["loss/smooth"]
        )
        assert abs(total.item() - expected.item()) < 1e-5, (
            f"total={total.item():.6f} != expected={expected.item():.6f}"
        )

    def test_lambdas_loaded_from_config(self, cfg, nav_loss: NavigationLoss) -> None:
        """Lambda values must match those in configs/default.yaml."""
        assert nav_loss.lambda_direction == cfg.loss.lambda_direction
        assert nav_loss.lambda_progress  == cfg.loss.lambda_progress
        assert nav_loss.lambda_smooth    == cfg.loss.lambda_smooth

    def test_gradient_flows_through_total(self, nav_loss: NavigationLoss) -> None:
        """backward() on total loss must succeed (end-to-end differentiability)."""
        # Use a leaf tensor so .grad is populated after backward
        pred = torch.randn(B, H, 2, requires_grad=True)
        gt   = torch.randn(B, H, 2)
        rdir = torch.rand(B) * 2 * math.pi
        total, _ = nav_loss(pred, gt, rdir)
        total.backward()
        assert pred.grad is not None


# ── Enable/disable flags ────────────────────────────────────────────────────────

class TestNavigationLossEnableFlags:
    """Verify that enable_* flags correctly zero out individual loss terms."""

    def _make_loss(self, **overrides) -> NavigationLoss:
        cfg = OmegaConf.load("configs/default.yaml")
        for key, val in overrides.items():
            OmegaConf.update(cfg, f"loss.{key}", val)
        return NavigationLoss(cfg)

    def _inputs(self):
        pred = torch.randn(B, H, 2).tanh()
        gt   = torch.randn(B, H, 2).tanh()
        rdir = torch.rand(B) * 2 * math.pi
        return pred, gt, rdir

    def test_disable_direction_zeros_loss(self) -> None:
        """enable_direction=false → loss/direction is 0.0."""
        loss_fn = self._make_loss(enable_direction=False)
        pred, gt, rdir = self._inputs()
        _, d = loss_fn(pred, gt, rdir)
        assert d["loss/direction"].item() == 0.0

    def test_disable_progress_zeros_loss(self) -> None:
        """enable_progress=false → loss/progress is 0.0."""
        loss_fn = self._make_loss(enable_progress=False)
        pred, gt, rdir = self._inputs()
        _, d = loss_fn(pred, gt, rdir)
        assert d["loss/progress"].item() == 0.0

    def test_disable_smooth_zeros_loss(self) -> None:
        """enable_smooth=false → loss/smooth is 0.0."""
        loss_fn = self._make_loss(enable_smooth=False)
        pred, gt, rdir = self._inputs()
        _, d = loss_fn(pred, gt, rdir)
        assert d["loss/smooth"].item() == 0.0

    def test_waypoint_only_total_equals_waypoint(self) -> None:
        """All auxiliary losses disabled → total == waypoint loss."""
        loss_fn = self._make_loss(
            enable_direction=False, enable_progress=False, enable_smooth=False
        )
        pred, gt, rdir = self._inputs()
        total, d = loss_fn(pred, gt, rdir)
        assert abs(total.item() - d["loss/waypoint"].item()) < 1e-6
