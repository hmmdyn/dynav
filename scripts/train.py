"""Training script for Map Navigation Model.

Usage:
    # Dummy data overfitting test (no real dataset needed)
    python scripts/train.py training.dummy=true training.epochs=20

    # Real data training
    python scripts/train.py data.data_dir=/path/to/data

    # Override decoder type for ablation
    python scripts/train.py decoder.type=self_attention

Hydra writes outputs (logs, checkpoints) to outputs/<date>/<time>/.
"""

import logging
import math
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

# Optional wandb — degrades gracefully if not installed
try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """Instantiate optimizer from config."""
    if cfg.training.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: '{cfg.training.optimizer}'")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warm-up followed by cosine decay."""
    warmup_epochs = cfg.training.warmup_epochs
    total_epochs  = cfg.training.epochs

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - warmup_epochs, 1),
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def _save_checkpoint(state: dict, path: Path) -> None:
    torch.save(state, path)
    log.info(f"Checkpoint → {path}")


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch.

    Returns:
        Dict of averaged loss components for the epoch (keys: ``loss/total``,
        ``loss/waypoint``, ``loss/direction``, ``loss/progress``, ``loss/smooth``).
    """
    model.train()
    running: dict[str, float] = {}

    for batch in loader:
        obs  = batch["observations"].to(device)    # (B, N_obs, 3, H, W)
        mp   = batch["map_image"].to(device)        # (B, 3, H, W)
        gt   = batch["gt_waypoints"].to(device)    # (B, H, 2)
        rdir = batch["route_direction"].to(device)  # (B,)

        out = model(obs, mp)
        total, loss_dict = criterion(out["waypoints"], gt, rdir)

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in loss_dict.items():
            running[k] = running.get(k, 0.0) + v.item()

    n = len(loader)
    return {k: v / n for k, v in running.items()}


@torch.no_grad()
def _eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one validation epoch and return mean total loss."""
    model.eval()
    total_loss = 0.0
    for batch in loader:
        obs  = batch["observations"].to(device)
        mp   = batch["map_image"].to(device)
        gt   = batch["gt_waypoints"].to(device)
        rdir = batch["route_direction"].to(device)
        out  = model(obs, mp)
        loss, _ = criterion(out["waypoints"], gt, rdir)
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


# ── Main ───────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train dynavModel with config-driven hyperparameters.

    Args:
        cfg: Hydra-merged DictConfig from configs/default.yaml and CLI overrides.
    """
    log.info("\n" + OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Datasets ───────────────────────────────────────────────────────────────
    from dynav.data.dataset import DummydynavDataset, dynavDataset

    n_obs = cfg.model.obs_context_length + 2

    if cfg.training.dummy:
        log.info("Using DummydynavDataset (dummy=true)")
        train_ds = DummydynavDataset(
            size=cfg.training.dummy_size,
            n_obs=n_obs,
            image_size=cfg.data.image_size,
            horizon=cfg.model.prediction_horizon,
        )
        val_ds = DummydynavDataset(
            size=max(cfg.training.dummy_size // 5, 16),
            n_obs=n_obs,
            image_size=cfg.data.image_size,
            horizon=cfg.model.prediction_horizon,
            seed=99,
        )
    else:
        from hydra.utils import get_original_cwd
        data_dir = Path(get_original_cwd()) / cfg.data.data_dir
        train_ds = dynavDataset(data_dir, split="train", image_size=cfg.data.image_size)
        val_ds   = dynavDataset(data_dir, split="val",   image_size=cfg.data.image_size)

    num_workers = cfg.training.num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    log.info(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    # ── Model ──────────────────────────────────────────────────────────────────
    from dynav.models.map_nav_model import dynavModel

    model = dynavModel.from_config(cfg).to(device)
    model.print_parameter_summary()

    # ── Loss / optimizer / scheduler ───────────────────────────────────────────
    from dynav.losses.navigation_losses import NavigationLoss

    criterion = NavigationLoss(cfg)
    optimizer  = _build_optimizer(model, cfg)
    scheduler  = _build_scheduler(optimizer, cfg)

    # ── Encoder freeze schedule ────────────────────────────────────────────────
    model.freeze_encoders()
    log.info(f"Encoders frozen for first {cfg.encoder.freeze_epochs} epoch(s)")

    # ── WandB ──────────────────────────────────────────────────────────────────
    use_wandb = cfg.training.use_wandb and _WANDB_AVAILABLE
    if cfg.training.use_wandb and not _WANDB_AVAILABLE:
        log.warning("wandb not installed — logging disabled")
    if use_wandb:
        _wandb.init(
            project="dynav",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ── Output dir (Hydra sets cwd to outputs/<date>/<time>/) ──────────────────
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    best_val_loss = math.inf
    save_every    = cfg.training.save_every

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(cfg.training.epochs):

        # Unfreeze encoder backbone after freeze_epochs
        if epoch == cfg.encoder.freeze_epochs:
            model.unfreeze_encoders()
            log.info(f"Epoch {epoch}: encoder backbone unfrozen")

        train_losses = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss     = _eval_one_epoch(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        log.info(
            f"[{epoch:3d}/{cfg.training.epochs}] "
            f"train={train_losses['loss/total']:.4f}  "
            f"val={val_loss:.4f}  "
            f"lr={current_lr:.2e}"
        )

        if use_wandb:
            _wandb.log({
                "epoch": epoch,
                "lr":    current_lr,
                "val/loss": val_loss,
                **{f"train/{k.split('/')[1]}": v for k, v in train_losses.items()},
            })

        # Checkpoint state
        state = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss":             val_loss,
            "cfg":                  OmegaConf.to_container(cfg, resolve=True),
        }

        if (epoch + 1) % save_every == 0:
            _save_checkpoint(state, ckpt_dir / f"epoch_{epoch:04d}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(state, ckpt_dir / "best.pt")

    if use_wandb:
        _wandb.finish()

    log.info(f"Training complete — best val loss: {best_val_loss:.4f}")
    log.info(f"Checkpoints in: {ckpt_dir.resolve()}")


if __name__ == "__main__":
    main()
