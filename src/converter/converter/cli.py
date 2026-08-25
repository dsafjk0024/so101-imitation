"""Offline CLI: rosbag episodes -> one LeRobot v3.0 dataset.

Not a ROS node. Needs a shell with ROS sourced and the pixi lerobot env:

    source /opt/ros/jazzy/setup.bash
    pixi run -e lerobot python -m converter.cli --bags bags/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from converter.config import DEFAULT_CONFIG_PATH, load_config

ROS_HINT = (
    "rosbag2_py is not importable. This CLI needs both environments:\n"
    "    source /opt/ros/jazzy/setup.bash\n"
    "    pixi run -e lerobot python -m converter.cli ...\n"
    "Sourcing ROS puts rosbag2_py on PYTHONPATH, which the pixi env inherits."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="converter",
        description="Convert recorded rosbag episodes into a LeRobot v3.0 dataset.",
    )
    parser.add_argument(
        "--bags",
        required=True,
        type=Path,
        help="directory holding one subdirectory per episode bag",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"conversion config YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="dataset repo id (default: repo_id from the config)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="dataset output directory (default: datasets/<repo_id>)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing dataset at the output path",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="store images as files instead of encoded video (debugging)",
    )
    parser.add_argument(
        "--max-interior-drop-frac",
        type=float,
        default=0.02,
        help="skip an episode whose interior dropped-tick fraction exceeds this",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="convert only the first N episodes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log every converted episode and per-feature sync statistics",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Raise only our own logger for --verbose. Turning the root logger up to
    # DEBUG makes Pillow log every PNG chunk it writes.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.verbose:
        logging.getLogger("converter").setLevel(logging.DEBUG)

    try:
        import rosbag2_py  # noqa: F401
    except ImportError:
        print(ROS_HINT, file=sys.stderr)
        return 2

    from converter.convert import convert_all

    cfg = load_config(args.config)
    repo_id = args.repo_id or cfg.repo_id
    root = args.root or Path("datasets") / repo_id

    report = convert_all(
        args.bags,
        cfg,
        root,
        use_videos=not args.no_videos,
        max_interior_drop_frac=args.max_interior_drop_frac,
        overwrite=args.overwrite,
        limit=args.limit,
    )

    print(
        f"\n{report['episodes_converted']} episode(s), {report['frames']} frames "
        f"-> {report['root']}"
    )
    if args.verbose:
        for entry in report["per_episode"]:
            print(
                f"  {entry['name']}: {entry['frames']} frames "
                f"(leading={entry['dropped_leading']} "
                f"interior={entry['dropped_interior']} "
                f"trailing={entry['dropped_trailing']})"
            )
            for key, summary in entry["features"].items():
                print(
                    f"    {key}: match={summary['match_rate']:.1%} "
                    f"mean_dt={summary['mean_dt_s'] * 1e3:.1f}ms "
                    f"max_dt={summary['dt_max_s'] * 1e3:.1f}ms"
                )

    if report["episodes_skipped"]:
        print(f"\nskipped {len(report['episodes_skipped'])} episode(s):")
        for name, reason in report["episodes_skipped"]:
            print(f"  {name}: {reason}")
        return 1

    if report["episodes_converted"] == 0:
        print("no episodes converted", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
