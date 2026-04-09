#!/usr/bin/env python3
"""Config-driven rosbag recording script for dynav data collection.

Reads a topic list from a YAML config and launches ``ros2 bag record``
as a subprocess.  Optionally verifies that required topics are being
published before starting.

Requires ROS2 Humble sourced in the current shell.

Usage
-----
# Record until Ctrl-C:
python scripts/record_bag.py

# Record for 120 seconds:
python scripts/record_bag.py --duration 120

# Custom config and output directory:
python scripts/record_bag.py \\
    --config configs/record_topics.yaml \\
    --output ~/bags/campus_run_01

# Skip topic pre-check:
python scripts/record_bag.py --no-check
"""

from __future__ import annotations

import argparse
import datetime
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

try:
    from omegaconf import OmegaConf
except ImportError:
    print("ERROR: omegaconf not installed.  Run: pip install omegaconf", file=sys.stderr)
    sys.exit(1)

_REPO = Path(__file__).resolve().parent.parent


def _load_config(config_path: Path):
    """Load and return the OmegaConf config from *config_path*."""
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    return OmegaConf.load(config_path)


def _resolve_output_dir(cfg, cli_output: Optional[str]) -> Path:
    """Determine the output bag directory.

    Priority: CLI --output > config output.dir > default ~/bags/TIMESTAMP.
    """
    if cli_output:
        return Path(cli_output).expanduser().resolve()

    base_dir_str = OmegaConf.select(cfg, "output.dir", default="~/bags")
    base_dir = Path(str(base_dir_str)).expanduser()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / timestamp


def _check_topics(topics: List[str], warn_missing: bool, min_active: int) -> bool:
    """Check which topics are currently being published.

    Args:
        topics:       Topic list from the config.
        warn_missing: If True, warn about missing topics but do not abort.
        min_active:   Minimum number of active topics required to proceed.

    Returns:
        True if the check passes, False if it fails and warn_missing is False.
    """
    if min_active == 0:
        return True

    print("Checking active topics (ros2 topic list)...")
    try:
        result = subprocess.run(
            ["ros2", "topic", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        active = set(result.stdout.strip().splitlines())
    except FileNotFoundError:
        print("  [WARN] 'ros2' not found in PATH — skipping topic check.")
        return True
    except subprocess.TimeoutExpired:
        print("  [WARN] ros2 topic list timed out — skipping topic check.")
        return True

    missing = [t for t in topics if t not in active]
    found = [t for t in topics if t in active]

    for t in found:
        print(f"  [OK]   {t}")
    for t in missing:
        print(f"  [MISS] {t}")

    if len(found) < min_active:
        msg = (
            f"Only {len(found)}/{min_active} required topics are active. "
            "Is the robot / camera driver running?"
        )
        if warn_missing:
            print(f"  [WARN] {msg}")
        else:
            print(f"  [ERROR] {msg}", file=sys.stderr)
            return False

    return True


def _build_ros2_record_cmd(
    topics: List[str],
    output_dir: Path,
    compression: str,
    duration: Optional[int],
) -> List[str]:
    """Construct the ``ros2 bag record`` command.

    Args:
        topics:      Topics to record.
        output_dir:  Bag output directory.
        compression: Compression codec ("zstd", "lz4", "none").
        duration:    If set, record for this many seconds then stop.

    Returns:
        List of command tokens ready for subprocess.
    """
    cmd = ["ros2", "bag", "record"]
    cmd += ["--output", str(output_dir)]

    if compression and compression.lower() != "none":
        cmd += ["--compression-mode", "file", "--compression-format", compression]

    if duration is not None:
        cmd += ["--duration", str(duration)]

    cmd += topics
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Config-driven rosbag recording for dynav data collection"
    )
    parser.add_argument(
        "--config",
        default=str(_REPO / "configs" / "record_topics.yaml"),
        help="Path to record_topics.yaml (default: configs/record_topics.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help="Output bag directory (overrides config output.dir)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Stop recording after this many seconds (default: record until Ctrl-C)",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip topic availability pre-check",
    )
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))

    topics: List[str] = list(OmegaConf.to_container(cfg.topics, resolve=True))
    if not topics:
        print("ERROR: no topics defined in config", file=sys.stderr)
        sys.exit(1)

    output_dir = _resolve_output_dir(cfg, args.output)
    compression = str(OmegaConf.select(cfg, "output.compression", default="zstd"))
    warn_missing = bool(OmegaConf.select(cfg, "recording.warn_missing", default=True))
    min_active = int(OmegaConf.select(cfg, "recording.min_active_topics", default=2))

    print(f"Recording {len(topics)} topics → {output_dir}")
    if args.duration:
        print(f"Duration: {args.duration}s")

    if not args.no_check:
        ok = _check_topics(topics, warn_missing, min_active)
        if not ok:
            sys.exit(1)

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = _build_ros2_record_cmd(topics, output_dir, compression, args.duration)
    print("\nRunning:", " ".join(cmd))
    print("Press Ctrl-C to stop.\n")

    proc = subprocess.Popen(cmd)

    def _handle_sigint(sig, frame):  # noqa: ARG001
        proc.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, _handle_sigint)

    ret = proc.wait()
    if ret not in (0, -2):  # -2 = SIGINT
        print(f"ros2 bag record exited with code {ret}", file=sys.stderr)
        sys.exit(ret)

    print(f"\nBag saved to: {output_dir}")


if __name__ == "__main__":
    main()
