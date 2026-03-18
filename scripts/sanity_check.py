"""Sanity check: overfit DummydynavDataset in 100 iterations.

Verifies that the entire training pipeline (model, loss, optimizer,
data → forward → backward → step) is correctly wired end-to-end and
that the model can learn a trivially simple mapping.

Pass criterion: waypoint_loss < --threshold after --iters gradient steps.

Exit codes:
    0  PASSED — training pipeline works and model can overfit.
    1  FAILED — loss did not decrease enough (pipeline or architecture bug).

Usage:
    python scripts/sanity_check.py                      # default settings
    python scripts/sanity_check.py --threshold 0.05 --iters 100
    python scripts/sanity_check.py --iters 200          # more iterations
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _build_lean_cfg():
    """Load default config and apply overrides for fast convergence.

    Changes from default:
        - pretrained=False  : no weight download, random init for all layers
        - n_layers=2        : half the decoder depth → faster forward pass
        - n_heads=2         : fewer heads → faster attention
        - d_ff=256          : narrower FFN
    """
    cfg = OmegaConf.load(_PROJECT_ROOT / "configs" / "default.yaml")
    OmegaConf.update(cfg, "encoder.pretrained", False)
    OmegaConf.update(cfg, "decoder.n_layers",   2)
    OmegaConf.update(cfg, "decoder.n_heads",    2)
    OmegaConf.update(cfg, "decoder.d_ff",       256)
    return cfg


def run_sanity_check(
    threshold: float = 0.05,
    n_iters: int = 100,
    lr: float = 1e-4,
    batch_size: int = 10,
    n_samples: int = 10,
    device_str: str = "cpu",
    verbose: bool = True,
) -> bool:
    """Run the overfitting sanity check and return True on success.

    Uses ``DummydynavDataset(fixed_targets=True)`` so all samples share the
    same straight-ahead gt_waypoints — the model only needs to learn the
    output bias, which converges very quickly.

    Args:
        threshold: Maximum acceptable waypoint_loss at the end of training.
        n_iters: Number of gradient steps to take.
        lr: Adam learning rate.
        batch_size: Samples per gradient step.
        n_samples: Total samples in the dummy dataset.
        verbose: Print progress every 20 iterations.

    Returns:
        True if final waypoint_loss < threshold, False otherwise.
    """
    from dynav.data.dataset import DummydynavDataset
    from dynav.losses.navigation_losses import compute_waypoint_loss
    from dynav.models.map_nav_model import dynavModel

    cfg    = _build_lean_cfg()
    device = torch.device(device_str)

    if verbose:
        print(
            f"[sanity_check] device={device}  iters={n_iters}  "
            f"lr={lr}  batch={batch_size}  samples={n_samples}  "
            f"threshold={threshold}"
        )

    # ── Dataset: fixed straight-ahead targets (trivially easy to overfit) ──────
    n_obs   = cfg.model.obs_context_length + 2
    dataset = DummydynavDataset(
        size=n_samples,
        n_obs=n_obs,
        image_size=cfg.data.image_size,
        horizon=cfg.model.prediction_horizon,
        fixed_targets=True,   # all samples → same gt_waypoints
        seed=42,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # ── Model setup ───────────────────────────────────────────────────────────
    # Strategy: freeze EfficientNet backbone, set the model to eval() mode for
    # stable BatchNorm (running stats = mean 0 / var 1 at init → ~identity),
    # then re-enable train() only for the lightweight trainable parts
    # (proj layers, decoder, head).
    #
    # Rationale: EfficientNet in train() mode with random-init BatchNorm +
    # small batch produces high-variance features that saturate tanh on the
    # first Adam step, stalling training. eval() mode for the backbone gives
    # deterministic, bounded features.
    model = dynavModel.from_config(cfg).to(device)
    model.freeze_encoders()
    model.eval()                            # backbone BN uses running stats

    # Re-enable train() for the parts we're actually optimizing
    model.decoder.train()
    model.waypoint_head.train()
    model.visual_encoder.proj.train()
    model.map_encoder.proj.train()

    # Only include trainable params in the optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    model.train()

    final_loss = float("inf")
    iteration  = 0
    loader_it  = iter(loader)

    while iteration < n_iters:
        # Cycle through the (small) dataset repeatedly
        try:
            batch = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            batch     = next(loader_it)

        obs   = batch["observations"].to(device)    # (B, N_obs, 3, H, W)
        mp    = batch["map_image"].to(device)        # (B, 3, H, W)
        gt_wp = batch["gt_waypoints"].to(device)    # (B, H, 2)

        out  = model(obs, mp)
        loss = compute_waypoint_loss(out["waypoints"], gt_wp)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        final_loss = loss.item()
        iteration += 1

        if verbose and iteration % 20 == 0:
            print(f"  iter {iteration:3d}/{n_iters}  waypoint_loss={final_loss:.4f}")

    passed = final_loss < threshold

    if verbose:
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(
            f"\n[sanity_check] {status}  "
            f"final waypoint_loss={final_loss:.4f}  threshold={threshold}"
        )
        if not passed:
            print(
                "  Hint: try --iters 200 or --lr 0.05 if loss is still decreasing.",
                file=sys.stderr,
            )

    return passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overfit DummydynavDataset to verify training pipeline correctness."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="Waypoint loss target (default: 0.05).",
    )
    parser.add_argument(
        "--iters", type=int, default=100,
        help="Number of gradient update steps (default: 100).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Adam learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Batch size per gradient step (default: 5).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="Dummy dataset size (default: 10).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device string (default: cpu). Use 'cuda' only if compatible.",
    )
    args = parser.parse_args()

    success = run_sanity_check(
        threshold=args.threshold,
        n_iters=args.iters,
        lr=args.lr,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        device_str=args.device,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
