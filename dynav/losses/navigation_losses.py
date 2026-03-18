"""Navigation loss functions for Map Navigation Model.

Total loss:
    L_total = L_waypoint + λ_dir * L_direction + λ_prog * L_progress + λ_smooth * L_smooth

All loss functions operate on normalized waypoints in [-1, 1].
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig


# ── Individual loss functions ──────────────────────────────────────────────────

def compute_waypoint_loss(
    pred_waypoints: torch.Tensor,
    gt_waypoints: torch.Tensor,
) -> torch.Tensor:
    """L1 regression loss between predicted and ground-truth waypoints.

    L_waypoint = (1/H) * Σ_i ||â_i - a*_i||₁

    Args:
        pred_waypoints: Predicted waypoints of shape (B, H, 2).
        gt_waypoints: Ground-truth waypoints of shape (B, H, 2).

    Returns:
        Scalar loss tensor.
    """
    # (B, H, 2) → mean over H and (x,y)
    return torch.abs(pred_waypoints - gt_waypoints).mean()


def compute_direction_loss(
    pred_waypoints: torch.Tensor,
    route_direction: torch.Tensor,
) -> torch.Tensor:
    """Penalize deviation of predicted trajectory direction from route direction.

    L_direction = 1 - cos(α̂, α*_route)

    The predicted direction is the direction of the total displacement vector
    (sum of all waypoint steps). This encourages the net trajectory to align
    with the intended route direction.

    Args:
        pred_waypoints: Predicted waypoints of shape (B, H, 2), each entry
            is a (Δx, Δy) step in robot body frame.
        route_direction: Route heading in radians (body frame) of shape (B,).

    Returns:
        Scalar loss tensor in [0, 2].
    """
    # Sum waypoint displacements to get total trajectory vector
    total_disp = pred_waypoints.sum(dim=1)                     # (B, 2)

    # Normalize predicted direction (clamp denom to avoid div-by-zero)
    pred_norm = total_disp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    pred_dir = total_disp / pred_norm                          # (B, 2) unit vector

    # Convert route angle (radians) to unit vector
    route_vec = torch.stack(
        [torch.cos(route_direction), torch.sin(route_direction)], dim=-1
    )                                                          # (B, 2)

    cos_sim = (pred_dir * route_vec).sum(dim=-1)               # (B,)
    return (1.0 - cos_sim).mean()


def compute_progress_loss(
    pred_waypoints: torch.Tensor,
    route_direction: torch.Tensor,
) -> torch.Tensor:
    """Incentivize forward progress along the route direction.

    L_progress = -(1/H) * Σ_i (â_i · d̂_route)

    Maximizing dot product with the route direction unit vector means each
    waypoint step contributes positive progress along the route.

    Args:
        pred_waypoints: Predicted waypoints of shape (B, H, 2).
        route_direction: Route heading in radians (body frame) of shape (B,).

    Returns:
        Scalar loss tensor (negative → minimizing increases progress).
    """
    # Route direction as unit vector, broadcast over H
    route_vec = torch.stack(
        [torch.cos(route_direction), torch.sin(route_direction)], dim=-1
    )                                                          # (B, 2)
    route_vec = route_vec.unsqueeze(1)                         # (B, 1, 2)

    # Per-waypoint progress dot product
    progress = (pred_waypoints * route_vec).sum(dim=-1)        # (B, H)
    return -progress.mean()


def compute_smooth_loss(pred_waypoints: torch.Tensor) -> torch.Tensor:
    """Penalize jerkiness in the predicted waypoint trajectory.

    L_smooth = (1/(H-1)) * Σ_i ||â_{i+1} - â_i||²

    Args:
        pred_waypoints: Predicted waypoints of shape (B, H, 2).

    Returns:
        Scalar loss tensor. Zero for H=1; undefined for H<1.
    """
    diff = pred_waypoints[:, 1:, :] - pred_waypoints[:, :-1, :]  # (B, H-1, 2)
    # L2 norm squared per step, then mean over (B, H-1)
    return (diff ** 2).sum(dim=-1).mean()


# ── Composite loss ─────────────────────────────────────────────────────────────

class NavigationLoss(nn.Module):
    """Weighted combination of all navigation loss terms.

    L_total = L_waypoint
            + λ_direction * L_direction
            + λ_progress  * L_progress
            + λ_smooth    * L_smooth

    Lambda values are read from ``cfg.loss`` (OmegaConf DictConfig).

    Args:
        cfg: OmegaConf DictConfig with a ``loss`` sub-key containing
            ``lambda_direction``, ``lambda_progress``, and ``lambda_smooth``.

    Example:
        >>> criterion = NavigationLoss(cfg)
        >>> pred = torch.randn(2, 5, 2).tanh()
        >>> gt   = torch.randn(2, 5, 2).tanh()
        >>> rdir = torch.zeros(2)
        >>> total, losses = criterion(pred, gt, rdir)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.lambda_direction: float = cfg.loss.lambda_direction
        self.lambda_progress: float  = cfg.loss.lambda_progress
        self.lambda_smooth: float    = cfg.loss.lambda_smooth

    def forward(
        self,
        pred_waypoints: torch.Tensor,
        gt_waypoints: torch.Tensor,
        route_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute total navigation loss and individual components.

        Args:
            pred_waypoints: Predicted waypoints of shape (B, H, 2).
            gt_waypoints: Ground-truth waypoints of shape (B, H, 2).
            route_direction: Route heading in radians (body frame), shape (B,).

        Returns:
            Tuple of:
                - ``total_loss``: Scalar weighted sum of all loss terms.
                - ``loss_dict``: Dict with keys ``"loss/total"``,
                  ``"loss/waypoint"``, ``"loss/direction"``,
                  ``"loss/progress"``, ``"loss/smooth"`` for wandb logging.
        """
        l_waypoint  = compute_waypoint_loss(pred_waypoints, gt_waypoints)
        l_direction = compute_direction_loss(pred_waypoints, route_direction)
        l_progress  = compute_progress_loss(pred_waypoints, route_direction)
        l_smooth    = compute_smooth_loss(pred_waypoints)

        total = (
            l_waypoint
            + self.lambda_direction * l_direction
            + self.lambda_progress  * l_progress
            + self.lambda_smooth    * l_smooth
        )

        loss_dict: dict[str, torch.Tensor] = {
            "loss/total":     total,
            "loss/waypoint":  l_waypoint,
            "loss/direction": l_direction,
            "loss/progress":  l_progress,
            "loss/smooth":    l_smooth,
        }
        return total, loss_dict
