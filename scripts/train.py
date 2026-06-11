"""Training script for Map Navigation Model.

Usage:
    # Dummy data overfitting test (no real dataset needed)
    python scripts/train.py data.data_dir=/path/to/data

    # Override decoder type for ablation
    python scripts/train.py decoder.type=cross_attention

Hydra writes outputs (logs, checkpoints) to outputs/<date>/<time>/.
"""

import logging
import math
import time
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dynav.utils.metrics import StratifiedMeter, compute_ade_fde, compute_per_horizon_de

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


def _flatten_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    """Map epoch metric keys to wandb panel paths.

    ``loss/waypoint`` → ``{prefix}/waypoint``; metric keys without the
    ``loss/`` prefix (``ade_m``, ``grad_norm``, ``ade_m/turn`` …) pass through
    as ``{prefix}/{key}``.
    """
    out = {}
    for k, v in metrics.items():
        key = k[len("loss/"):] if k.startswith("loss/") else k
        out[f"{prefix}/{key}"] = v
    return out


@torch.no_grad()
def _trajectory_figure(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 16,
):
    """Pred-vs-GT waypoint trajectories (body frame, meters) for the first
    val batch — returns a matplotlib Figure, or None if matplotlib is missing.

    Axes: vertical = x (forward), horizontal = -y (so robot-left renders left).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    model.eval()
    batch = next(iter(loader))
    obs = batch["observations"].to(device)
    mp  = batch["map_image"].to(device)
    out = model(obs, mp)

    n = min(n_samples, obs.shape[0])
    norm_m = batch["waypoint_norm_m"][:n, None, None]            # (n, 1, 1)
    pred = out["waypoints"][:n].float().cpu() * norm_m            # (n, H, 2) m
    gt   = batch["gt_waypoints"][:n] * norm_m                     # (n, H, 2) m
    labels = batch["maneuver"][:n]

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        for wps, color, lab in ((gt[i], "tab:green", "GT"), (pred[i], "tab:red", "pred")):
            xs = torch.cat([torch.zeros(1), wps[:, 0]])
            ys = torch.cat([torch.zeros(1), wps[:, 1]])
            ax.plot(-ys, xs, "-o", color=color, markersize=3, label=lab)
        ax.set_title(labels[i], fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    grad_clip_norm: float = 1.0,
) -> dict[str, float]:
    """Run one training epoch.

    Returns:
        Dict of averaged loss components for the epoch (``loss/total``,
        ``loss/waypoint``, ``loss/direction``, ``loss/progress``,
        ``loss/smooth``) plus ``grad_norm`` (pre-clip, epoch mean),
        ``ade_m``/``fde_m`` (meters), and ``samples_per_sec``.
    """
    model.train()
    running: dict[str, float] = {}
    grad_norm_sum = 0.0
    ade_sum = fde_sum = 0.0
    n_samples = 0
    t0 = time.perf_counter()

    for batch in loader:
        obs  = batch["observations"].to(device)    # (B, N_obs, 3, H, W)
        mp   = batch["map_image"].to(device)        # (B, 3, H, W)
        gt   = batch["gt_waypoints"].to(device)    # (B, H, 2)
        rdir = batch["route_direction"].to(device)  # (B,)
        norm_m = batch["waypoint_norm_m"].to(device)  # (B,)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(obs, mp)
            total, loss_dict = criterion(out["waypoints"], gt, rdir)

        optimizer.zero_grad()
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        for k, v in loss_dict.items():
            running[k] = running.get(k, 0.0) + v.item()
        grad_norm_sum += grad_norm.item()

        ade, fde = compute_ade_fde(out["waypoints"].float(), gt, norm_m)
        ade_sum += ade.sum().item()
        fde_sum += fde.sum().item()
        n_samples += gt.shape[0]

    n = len(loader)
    elapsed = time.perf_counter() - t0
    result = {k: v / n for k, v in running.items()}
    result["grad_norm"] = grad_norm_sum / n
    result["ade_m"] = ade_sum / max(n_samples, 1)
    result["fde_m"] = fde_sum / max(n_samples, 1)
    result["samples_per_sec"] = n_samples / max(elapsed, 1e-9)
    return result


@torch.no_grad()
def _eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, float]:
    """Run one validation epoch.

    Returns:
        Dict of averaged loss components (``loss/total``, ``loss/waypoint``,
        ``loss/direction``, ``loss/progress``, ``loss/smooth``) plus metric
        keys ``ade_m``/``fde_m`` (overall, meters) and per-maneuver
        ``ade_m/{label}``/``fde_m/{label}`` when maneuver labels exist.
    """
    model.eval()
    running: dict[str, float] = {}
    ade_meter = StratifiedMeter()
    fde_meter = StratifiedMeter()
    horizon_sum: torch.Tensor | None = None   # (H,) accumulated DE per index
    n_samples = 0

    for batch in loader:
        obs  = batch["observations"].to(device)
        mp   = batch["map_image"].to(device)
        gt   = batch["gt_waypoints"].to(device)
        rdir = batch["route_direction"].to(device)
        norm_m = batch["waypoint_norm_m"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            out  = model(obs, mp)
            _, loss_dict = criterion(out["waypoints"], gt, rdir)
        for k, v in loss_dict.items():
            running[k] = running.get(k, 0.0) + v.item()

        ade, fde = compute_ade_fde(out["waypoints"].float(), gt, norm_m)
        ade_meter.update(ade.cpu(), batch["maneuver"])
        fde_meter.update(fde.cpu(), batch["maneuver"])

        de = compute_per_horizon_de(out["waypoints"].float(), gt, norm_m)  # (B, H)
        horizon_sum = de.sum(dim=0) if horizon_sum is None else horizon_sum + de.sum(dim=0)
        n_samples += de.shape[0]

    n = max(len(loader), 1)
    result = {k: v / n for k, v in running.items()}

    if horizon_sum is not None and n_samples:
        for i, v in enumerate((horizon_sum / n_samples).tolist()):
            result[f"de_m_h{i + 1}"] = v   # error growth along the horizon

    ade_means, fde_means = ade_meter.means(), fde_meter.means()
    result["ade_m"] = ade_means.pop("all", float("nan"))
    result["fde_m"] = fde_means.pop("all", float("nan"))
    # Per-maneuver breakdown — skip if labels are absent (single "unknown" bucket)
    if set(ade_means) != {"unknown"}:
        for lab, v in ade_means.items():
            result[f"ade_m/{lab}"] = v
        for lab, v in fde_means.items():
            result[f"fde_m/{lab}"] = v
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train DyNavModel with config-driven hyperparameters.

    Args:
        cfg: Hydra-merged DictConfig from configs/default.yaml and CLI overrides.
    """
    log.info("\n" + OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Datasets ───────────────────────────────────────────────────────────────
    from dynav.data.dataset import DyNavDataset
    from hydra.utils import get_original_cwd

    data_dir = Path(get_original_cwd()) / cfg.data.data_dir
    train_ds = DyNavDataset(data_dir, split="train", image_size=cfg.data.image_size)
    val_ds   = DyNavDataset(data_dir, split="val",   image_size=cfg.data.image_size)

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
    from dynav.models.map_nav_model import DyNavModel

    model = DyNavModel.from_config(cfg).to(device)
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
        run_name = cfg.training.get("wandb_run_name", None) or (
            f"{cfg.decoder.type}-{cfg.loss.waypoint_type}"
            f"{'-mapdrop' + str(cfg.model.get('map_dropout_p', 0.0)) if cfg.model.get('map_dropout_p', 0.0) else ''}"
            f"{'-maponly' if cfg.model.get('disable_obs', False) else ''}"
            f"{'-obsonly' if cfg.model.get('disable_map', False) else ''}"
        )
        _wandb.init(
            project=cfg.training.get("wandb_project", "dynav"),
            entity=cfg.training.get("wandb_entity", None),
            name=run_name,
            tags=list(cfg.training.get("wandb_tags", [])) or None,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        # Primary comparison metric across runs: val ADE in meters (lower=better)
        _wandb.define_metric("val/ade_m", summary="min")
        _wandb.define_metric("val/fde_m", summary="min")
        _wandb.define_metric("val/total", summary="min")
        _wandb.watch(model, log="gradients", log_freq=200)

    # ── AMP ────────────────────────────────────────────────────────────────────
    use_amp = cfg.training.get("amp", False) and device.type == "cuda"
    if use_amp:
        log.info("AMP enabled: BF16 autocast")

    # ── Output dir (Hydra sets cwd to outputs/<date>/<time>/) ──────────────────
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    best_val_loss = math.inf
    save_every    = cfg.training.save_every
    patience      = cfg.training.get("early_stopping_patience", 0)
    epochs_no_improve = 0

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(cfg.training.epochs):

        # Unfreeze encoder backbone after freeze_epochs
        if epoch == cfg.encoder.freeze_epochs:
            model.unfreeze_encoders()
            log.info(f"Epoch {epoch}: encoder backbone unfrozen")

        train_losses = _train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp,
            grad_clip_norm=cfg.training.get("grad_clip_norm", 1.0),
        )
        val_losses   = _eval_one_epoch(model, val_loader, criterion, device, use_amp)
        val_loss     = val_losses["loss/total"]

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        log.info(
            f"[{epoch:3d}/{cfg.training.epochs}] "
            f"train={train_losses['loss/total']:.4f}  "
            f"val={val_loss:.4f}  "
            f"val_ade={val_losses['ade_m']:.2f}m  "
            f"val_fde={val_losses['fde_m']:.2f}m  "
            f"lr={current_lr:.2e}"
        )

        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "lr":    current_lr,
                **_flatten_metrics("train", train_losses),
                **_flatten_metrics("val", val_losses),
            }

            viz_every = cfg.training.get("viz_every", 5)
            if viz_every and (epoch + 1) % viz_every == 0:
                fig = _trajectory_figure(model, val_loader, device)
                if fig is not None:
                    log_dict["val/trajectories"] = _wandb.Image(fig)
                    import matplotlib.pyplot as plt
                    plt.close(fig)

            _wandb.log(log_dict)

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
            epochs_no_improve = 0
            _save_checkpoint(state, ckpt_dir / "best.pt")
        else:
            epochs_no_improve += 1

        # Early stopping: halt if val loss has not improved for `patience` epochs
        if patience and epochs_no_improve >= patience:
            log.info(
                f"Early stopping at epoch {epoch}: "
                f"val loss did not improve for {patience} epochs "
                f"(best={best_val_loss:.4f})"
            )
            break

    if use_wandb:
        _wandb.finish()

    log.info(f"Training complete — best val loss: {best_val_loss:.4f}")
    log.info(f"Checkpoints in: {ckpt_dir.resolve()}")


if __name__ == "__main__":
    main()
