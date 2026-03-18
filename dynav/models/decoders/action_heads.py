"""Action head modules for Map Navigation Model.

BaseActionHead defines the shared interface; WaypointHead is the default
regression-based implementation. A future DiffusionHead can extend
BaseActionHead without changing downstream code.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseActionHead(ABC, nn.Module):
    """Abstract base class for action prediction heads.

    All action heads consume a context vector (B, d) produced by a decoder
    and output a waypoint tensor of shape (B, H, 2).

    Subclasses must implement :meth:`forward`.
    """

    @abstractmethod
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Predict waypoints from a context vector.

        Args:
            context: Decoder output of shape (B, d).

        Returns:
            Waypoints of shape (B, H, 2) where each (Δx, Δy) pair is
            normalized to [-1, 1].
        """


class WaypointHead(BaseActionHead):
    """Regression-based waypoint prediction head.

    Predicts H relative waypoints (Δx, Δy) in the robot body frame.
    Output is bounded to [-1, 1] via tanh (waypoints are pre-normalized by
    max_waypoint_distance before training).

    Architecture:
        Linear(token_dim, hidden_dim) → ReLU → Linear(hidden_dim, H*2) → tanh
        → reshape (B, H, 2)

    Args:
        token_dim: Input context vector dimension (d).
        hidden_dim: Intermediate hidden dimension.
        prediction_horizon: Number of waypoints to predict (H).

    Example:
        >>> head = WaypointHead(token_dim=256, hidden_dim=128, prediction_horizon=5)
        >>> ctx = torch.randn(2, 256)
        >>> wps = head(ctx)
        >>> wps.shape    # (2, 5, 2)
        >>> wps.abs().max().item() <= 1.0
        True
    """

    def __init__(
        self,
        token_dim: int = 256,
        hidden_dim: int = 128,
        prediction_horizon: int = 5,
    ) -> None:
        super().__init__()

        self.prediction_horizon = prediction_horizon

        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prediction_horizon * 2),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Predict normalized relative waypoints from a context vector.

        Args:
            context: Decoder output of shape (B, d).

        Returns:
            Waypoints of shape (B, H, 2), values in [-1, 1].
        """
        B = context.shape[0]

        out = self.net(context)              # (B, H*2)
        out = torch.tanh(out)               # (B, H*2)  — bound to [-1, 1]
        waypoints = out.view(B, self.prediction_horizon, 2)  # (B, H, 2)

        return waypoints
