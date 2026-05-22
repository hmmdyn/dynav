"""Per-component val loss breakdown across all saved checkpoints.

Usage (run from dynav/ repo root):
    python scripts/eval_checkpoints.py --data_dir /path/to/dataset
    python scripts/eval_checkpoints.py --data_dir /home/hmmdyn/data/frodobots/dataset
    python scripts/eval_checkpoints.py --ckpt_dir scripts/checkpoints --data_dir /path/to/dataset
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@torch.no_grad()
def eval_checkpoint(ckpt_path: Path, data_dir: Path, device: torch.device, batch_size: int = 64):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = state["cfg"]
    cfg = OmegaConf.create(cfg_dict)

    from dynav.data.dataset import DyNavDataset
    from dynav.losses.navigation_losses import NavigationLoss
    from dynav.models.map_nav_model import DyNavModel

    val_ds = DyNavDataset(data_dir, split="val", image_size=cfg.data.image_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = DyNavModel.from_config(cfg).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    criterion = NavigationLoss(cfg)

    totals: dict[str, float] = {}
    for batch in val_loader:
        obs  = batch["observations"].to(device)
        mp   = batch["map_image"].to(device)
        gt   = batch["gt_waypoints"].to(device)
        rdir = batch["route_direction"].to(device)
        _, loss_dict = criterion(model(obs, mp)["waypoints"], gt, rdir)
        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + v.item()

    n = len(val_loader)
    return {k: v / n for k, v in totals.items()}, state.get("epoch", "?")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="scripts/checkpoints")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    data_dir = Path(args.data_dir)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Val data: {data_dir}\n")

    # Collect checkpoints: epoch_*.pt first (sorted), then best.pt
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt")) + [ckpt_dir / "best.pt"]
    ckpts = [p for p in ckpts if p.exists()]

    header = f"{'checkpoint':<22} {'epoch':>5}  {'total':>7}  {'waypoint':>8}  {'direction':>9}  {'progress':>8}  {'smooth':>7}"
    print(header)
    print("-" * len(header))

    best_label = ""
    for ckpt in ckpts:
        losses, epoch = eval_checkpoint(ckpt, data_dir, device, args.batch_size)
        label = ckpt.name
        if label == "best.pt":
            label += " ★"
        print(
            f"{label:<22} {str(epoch):>5}  "
            f"{losses['loss/total']:>7.4f}  "
            f"{losses['loss/waypoint']:>8.4f}  "
            f"{losses['loss/direction']:>9.4f}  "
            f"{losses['loss/progress']:>8.4f}  "
            f"{losses['loss/smooth']:>7.4f}"
        )


if __name__ == "__main__":
    main()
