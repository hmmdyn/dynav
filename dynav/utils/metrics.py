"""Evaluation metrics for Map Navigation Model.

Normalized-space val loss depends on clipping/normalization constants and is
not interpretable; reporting metrics are ADE/FDE in meters:

    ADE = (1/H) Σ_i ||â_i - a*_i||₂ · d_norm      (average displacement error)
    FDE = ||â_H - a*_H||₂ · d_norm                 (final displacement error)
"""

import torch


@torch.no_grad()
def compute_ade_fde(
    pred_waypoints: torch.Tensor,
    gt_waypoints: torch.Tensor,
    waypoint_norm_m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample ADE/FDE in meters.

    Args:
        pred_waypoints: Predicted waypoints (B, H, 2), normalized [-1, 1].
        gt_waypoints: Ground-truth waypoints (B, H, 2), normalized [-1, 1].
        waypoint_norm_m: Normalization radius in meters, shape (B,) or scalar.

    Returns:
        Tuple of (ade_m, fde_m), each of shape (B,).
    """
    dist = (pred_waypoints - gt_waypoints).norm(dim=-1)   # (B, H)
    ade = dist.mean(dim=-1) * waypoint_norm_m             # (B,)
    fde = dist[:, -1] * waypoint_norm_m                   # (B,)
    return ade, fde


class StratifiedMeter:
    """Accumulates per-sample metric values grouped by a string label.

    Used to report val ADE/FDE stratified by maneuver class — mean metrics on
    straight-dominated data hide turning failures.

    Example:
        >>> meter = StratifiedMeter()
        >>> meter.update(ade, ["straight", "turn_left"])
        >>> meter.means()   # {"straight": ..., "turn_left": ..., "all": ...}
    """

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, values: torch.Tensor, labels: list[str]) -> None:
        """Accumulate a batch of per-sample values with their labels.

        Args:
            values: Per-sample metric values, shape (B,).
            labels: Per-sample string labels, length B.
        """
        for v, lab in zip(values.tolist(), labels):
            self._sums[lab] = self._sums.get(lab, 0.0) + v
            self._counts[lab] = self._counts.get(lab, 0) + 1

    def means(self) -> dict[str, float]:
        """Per-label means plus the overall mean under key ``"all"``."""
        out = {lab: self._sums[lab] / self._counts[lab] for lab in self._sums}
        total_n = sum(self._counts.values())
        if total_n:
            out["all"] = sum(self._sums.values()) / total_n
        return out
