#!/usr/bin/env python3
"""Read-only quality audit for a LeRobotDataset collected via SO-101 leader-follower
teleop, aimed at catching two ACT failure modes before training:

  - dwell deadlock: long, position-varying stalls get recorded as "action: stop",
    so a trained policy freezes wherever it's ever seen a long stall.
  - reversal multimodality: if the same visual state gets corrected in opposite
    directions across episodes, ACT's CVAE latent (fixed to z=0 at inference)
    averages the modes into near-zero motion.

Never writes to --dataset-path. All outputs go under --out-dir (created fresh).

Run (inside the `lerobot` pixi env, which has pandas/pyarrow/numpy/matplotlib):

    pixi run -e lerobot python scripts/analyze_teleop_quality.py \
        --dataset-path outputs/datasets/<repo_id>

Deliberately does not use `LeRobotDataset` for reading frames: its default
indexing decodes video, and this script must never touch the image/video
columns. Episode/feature metadata comes straight from meta/info.json, and
per-frame action/state data comes from the data/ parquet files directly with
an explicit column selection -- this also means the same code path handles
both v2.x (one parquet per episode) and v3.x (multiple episodes concatenated
per parquet, via the `episode_index` column) with no version branch: episode
boundaries come from grouping by `episode_index`, not from file layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REQUIRED_COLUMNS = ["action", "observation.state", "timestamp", "frame_index", "episode_index"]


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_info(dataset_path: Path) -> dict:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        die(
            f"{info_path} not found -- this doesn't look like a LeRobotDataset root.\n"
            f"Actual tree under {dataset_path}:\n" + _tree(dataset_path)
        )
    return json.loads(info_path.read_text())


def _tree(root: Path, max_depth: int = 3) -> str:
    if not root.exists():
        return f"  (does not exist: {root})"
    lines = []
    for p in sorted(root.rglob("*")):
        depth = len(p.relative_to(root).parts)
        if depth > max_depth:
            continue
        lines.append("  " + "  " * (depth - 1) + p.name + ("/" if p.is_dir() else ""))
    return "\n".join(lines) if lines else "  (empty)"


def resolve_joint_layout(info: dict) -> tuple[list[str], list[int], list[int]]:
    names = info["features"]["action"]["names"]
    gripper_idx = [i for i, n in enumerate(names) if "gripper" in n.lower()]
    arm_idx = [i for i in range(len(names)) if i not in gripper_idx]
    if not gripper_idx:
        print(
            f"WARNING: no joint name contains 'gripper' in {names}; "
            "treating all joints as arm joints for dwell/reversal metrics.",
            file=sys.stderr,
        )
    return names, arm_idx, gripper_idx


def discover_data_files(dataset_path: Path) -> list[Path]:
    files = sorted((dataset_path / "data").rglob("*.parquet"))
    if not files:
        die(
            f"no parquet files under {dataset_path / 'data'}.\n"
            f"Actual tree under {dataset_path}:\n" + _tree(dataset_path)
        )
    return files


def load_frames(dataset_path: Path, files: list[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        table = pq.read_table(f, columns=REQUIRED_COLUMNS)
        frames.append(table.to_pandas())
    df = pd.concat(frames, ignore_index=True)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        die(f"parquet files under {dataset_path / 'data'} are missing columns: {missing}")
    df["action"] = df["action"].apply(np.asarray)
    df["observation.state"] = df["observation.state"].apply(np.asarray)
    return df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def severity(length_sec: float, warn_sec: float, critical_sec: float) -> str:
    if length_sec < 0.5:
        return "ignore"
    if length_sec < warn_sec:
        return "minor"
    if length_sec < critical_sec:
        return "warn"
    return "critical"


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end] index ranges (inclusive) where mask is True."""
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def analyze_episode(ep_df: pd.DataFrame, arm_idx, gripper_idx, eps, warn_sec, critical_sec):
    """Return (dwell_segments: list[dict], reversal_count: int, duration_sec: float,
    phase_dwell: np.ndarray[T-1] bool aligned to frames[1:])."""
    action = np.stack(ep_df["action"].to_numpy())
    frame_index = ep_df["frame_index"].to_numpy()
    timestamp = ep_df["timestamp"].to_numpy()
    T = len(ep_df)
    duration_sec = float(timestamp[-1] - timestamp[0]) if T > 1 else 0.0

    if T < 2:
        return [], 0, duration_sec, np.zeros(0, dtype=bool), []

    arm = action[:, arm_idx]
    grip = action[:, gripper_idx] if gripper_idx else np.zeros((T, 0))
    delta_arm = arm[1:] - arm[:-1]
    delta_norm = np.linalg.norm(delta_arm, axis=1)
    delta_grip_abs_sum_per_step = np.abs(grip[1:] - grip[:-1]).sum(axis=1) if grip.shape[1] else np.zeros(T - 1)

    is_dwell = delta_norm < eps

    segments = []
    for start, end in find_runs(is_dwell):
        # `start`/`end` index into the length-(T-1) delta arrays; the stationary
        # span they describe covers frames [start .. end+1] of the episode.
        f_start, f_end = start, end + 1
        length_sec = float(timestamp[f_end] - timestamp[f_start])
        sev = severity(length_sec, warn_sec, critical_sec)
        if sev == "ignore":
            continue
        gripper_moved = float(delta_grip_abs_sum_per_step[start : end + 1].sum())
        segments.append(
            {
                "start_frame_index": int(frame_index[f_start]),
                "end_frame_index": int(frame_index[f_end]),
                "start_timestamp": float(timestamp[f_start]),
                "end_timestamp": float(timestamp[f_end]),
                "length_sec": length_sec,
                "phase_start": f_start / (T - 1),
                "severity": sev,
                "gripper_moved_abs_sum": gripper_moved,
                "benign_gripper": gripper_moved > eps,
                "leading": f_start == 0,
                "trailing": f_end == T - 1,
            }
        )

    reversal_count = 0
    reversal_phases = []
    for d in range(len(arm_idx)):
        v = delta_arm[:, d]
        active = np.abs(v) > eps
        signs = np.sign(v[active])
        positions = np.nonzero(active)[0]
        if len(signs) < 2:
            continue
        flips = np.nonzero(np.diff(signs) != 0)[0]
        reversal_count += len(flips)
        reversal_phases.extend((positions[flips] + 1) / (T - 1))

    return segments, reversal_count, duration_sec, is_dwell, reversal_phases


def resample_phase_bins(is_dwell: np.ndarray, n_bins: int = 100) -> np.ndarray:
    """is_dwell is length T-1 (aligned to frames[1:]); return length-n_bins bool
    array of whether the episode was dwelling in each normalized-phase bin,
    filling any bin with no frame (very short episodes) from its neighbor."""
    out = np.full(n_bins, np.nan)
    T1 = len(is_dwell)
    if T1 == 0:
        return np.zeros(n_bins, dtype=bool)
    phases = (np.arange(T1) + 1) / (T1 + 1 - 1) if T1 > 1 else np.array([1.0])
    bin_idx = np.clip((phases * n_bins).astype(int), 0, n_bins - 1)
    for b in range(n_bins):
        vals = is_dwell[bin_idx == b]
        if len(vals):
            out[b] = vals.any()
    idx = np.arange(n_bins)
    valid = ~np.isnan(out)
    if not valid.any():
        return np.zeros(n_bins, dtype=bool)
    out = np.interp(idx, idx[valid], out[valid].astype(float))
    return out > 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-path", required=True, type=Path)
    ap.add_argument("--eps", type=float, default=None, help="dwell/reversal threshold on ||delta_action_arm||; default is data-driven, see report")
    ap.add_argument("--fps-override", type=float, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("analysis_output"))
    ap.add_argument("--dwell-warn-sec", type=float, default=1.0)
    ap.add_argument("--dwell-critical-sec", type=float, default=2.0)
    args = ap.parse_args()

    dataset_path = args.dataset_path.expanduser().resolve()
    info = load_info(dataset_path)
    fps = args.fps_override or info.get("fps")
    if not fps:
        die("fps not found in meta/info.json; pass --fps-override")
    names, arm_idx, gripper_idx = resolve_joint_layout(info)

    files = discover_data_files(dataset_path)
    df = load_frames(dataset_path, files)
    episodes = sorted(df["episode_index"].unique().tolist())

    print(f"dataset: {dataset_path}")
    print(f"codebase_version: {info.get('codebase_version')}")
    print(f"fps: {fps}  episodes: {len(episodes)}  frames: {len(df)}")
    print(f"joint names: {names}")
    print(f"arm_idx: {arm_idx}  gripper_idx: {gripper_idx}")

    # ---- Stage 0: scale ----
    action_all = np.stack(df["action"].to_numpy())
    print("\n=== Stage 0: action scale (per joint) ===")
    for i, n in enumerate(names):
        col = action_all[:, i]
        print(f"  {n:20s} min={col.min():+8.4f} max={col.max():+8.4f} mean={col.mean():+8.4f} std={col.std():.4f}")

    delta_norms = []
    for ep in episodes:
        ep_df = df[df.episode_index == ep]
        arm = np.stack(ep_df["action"].to_numpy())[:, arm_idx]
        if len(arm) > 1:
            delta_norms.append(np.linalg.norm(arm[1:] - arm[:-1], axis=1))
    delta_norms = np.concatenate(delta_norms) if delta_norms else np.zeros(0)

    pcts = [1, 5, 10, 25, 50, 75, 90, 99]
    pct_vals = np.percentile(delta_norms, pcts) if len(delta_norms) else np.zeros(len(pcts))
    print("\n||delta_action_arm|| percentiles:")
    for p, v in zip(pcts, pct_vals):
        print(f"  p{p:02d} = {v:.5f}")

    nonzero = delta_norms[delta_norms > 0]
    p10_nonzero = float(np.percentile(nonzero, 10)) if len(nonzero) else 0.0
    noise_floor = float(np.percentile(nonzero, 1)) if len(nonzero) else 0.0
    eps = args.eps if args.eps is not None else max(p10_nonzero, noise_floor)
    print(f"\np10(nonzero) = {p10_nonzero:.5f}  noise_floor(p1 nonzero) = {noise_floor:.5f}")
    print(f"EPS used = {eps:.5f}  (source: {'--eps override' if args.eps is not None else 'data-driven max(p10_nonzero, noise_floor)'})")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.hist(delta_norms, bins=200, log=True)
    plt.axvline(eps, color="red", linestyle="--", label=f"EPS={eps:.4f}")
    plt.xlabel("||delta_action_arm|| per frame")
    plt.ylabel("count (log)")
    plt.title("Frame-to-frame arm action delta distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "delta_hist.png", dpi=120)
    plt.close()

    # ---- Stage 1 & 2: per-episode dwell + reversal ----
    all_segments = []
    ep_rows = []
    phase_bins_per_ep = {}
    reversal_phase_all = []

    for ep in episodes:
        ep_df = df[df.episode_index == ep].reset_index(drop=True)
        segments, reversal_count, duration_sec, is_dwell, reversal_phases = analyze_episode(
            ep_df, arm_idx, gripper_idx, eps, args.dwell_warn_sec, args.dwell_critical_sec
        )
        for s in segments:
            s["episode_index"] = ep
            all_segments.append(s)
        reversal_phase_all.extend(reversal_phases)
        phase_bins_per_ep[ep] = resample_phase_bins(is_dwell)

        max_dwell = max((s["length_sec"] for s in segments), default=0.0)
        n_critical = sum(1 for s in segments if s["severity"] == "critical")
        total_dwell_frames = sum(s["end_frame_index"] - s["start_frame_index"] for s in segments)
        reversal_rate = reversal_count / duration_sec if duration_sec > 0 else 0.0
        ep_rows.append(
            {
                "episode_index": ep,
                "n_frames": len(ep_df),
                "duration_sec": duration_sec,
                "total_dwell_frames": total_dwell_frames,
                "max_dwell_sec": max_dwell,
                "n_critical": n_critical,
                "reversal_rate": reversal_rate,
            }
        )

    ep_summary = pd.DataFrame(ep_rows)
    median_dur = ep_summary["duration_sec"].median()
    ep_summary["is_outlier"] = ep_summary["duration_sec"] > 1.5 * median_dur
    reversal_p90 = ep_summary["reversal_rate"].quantile(0.9) if len(ep_summary) else 0.0

    dwell_df = pd.DataFrame(all_segments)

    # ---- Stage 3: cross-episode phase consistency ----
    if phase_bins_per_ep:
        stall_matrix = np.stack([phase_bins_per_ep[ep] for ep in episodes])
        stall_fraction = stall_matrix.mean(axis=0)
    else:
        stall_fraction = np.zeros(100)

    plt.figure(figsize=(8, 3))
    plt.plot(np.linspace(0, 1, 100), stall_fraction)
    plt.axhline(0.7, color="green", linestyle="--", linewidth=1, label="benign >0.7")
    plt.axhspan(0.2, 0.6, color="orange", alpha=0.15, label="risky 0.2-0.6")
    plt.xlabel("normalized episode phase")
    plt.ylabel("fraction of episodes dwelling")
    plt.title("Phase-stall consistency profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "phase_stall_profile.png", dpi=120)
    plt.close()

    plt.figure(figsize=(9, max(3, 0.15 * len(episodes))))
    for row_i, ep in enumerate(episodes):
        segs = dwell_df[dwell_df.episode_index == ep] if len(dwell_df) else dwell_df
        ep_df = df[df.episode_index == ep]
        T = len(ep_df)
        for _, s in segs.iterrows():
            width = (s["end_frame_index"] - s["start_frame_index"]) / max(T - 1, 1)
            color = {"minor": "gold", "warn": "orange", "critical": "red"}[s["severity"]]
            plt.barh(row_i, width, left=s["phase_start"], color=color, height=0.8)
    plt.yticks(range(len(episodes)), [str(e) for e in episodes], fontsize=6)
    plt.xlabel("normalized episode phase")
    plt.ylabel("episode_index")
    plt.title("Per-episode dwell timeline (gold=minor, orange=warn, red=critical)")
    plt.tight_layout()
    plt.savefig(args.out_dir / "episode_timeline.png", dpi=120)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(ep_summary["duration_sec"], bins=30)
    plt.axvline(median_dur, color="black", linestyle="--", label=f"median={median_dur:.2f}s")
    plt.axvline(1.5 * median_dur, color="red", linestyle="--", label="1.5x median (outlier)")
    plt.xlabel("episode duration (sec)")
    plt.ylabel("count")
    plt.title("Episode length distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "episode_length_hist.png", dpi=120)
    plt.close()

    dwell_df.to_csv(args.out_dir / "dwell_segments.csv", index=False)
    ep_summary.to_csv(args.out_dir / "episode_summary.csv", index=False)

    # ---- report.md ----
    rerecord = ep_summary[(ep_summary.n_critical > 0) | (ep_summary.reversal_rate >= reversal_p90)]
    trim_eps = set()
    critical_or_warn_interior_eps = set()
    if len(dwell_df):
        for ep in episodes:
            segs = dwell_df[dwell_df.episode_index == ep]
            interior_bad = segs[(~segs.leading) & (~segs.trailing) & (segs.severity.isin(["warn", "critical"]))]
            edge_bad = segs[(segs.leading | segs.trailing) & (segs.severity.isin(["minor", "warn", "critical"]))]
            if len(interior_bad):
                critical_or_warn_interior_eps.add(ep)
            elif len(edge_bad):
                trim_eps.add(ep)

    benign_ranges = []
    in_range = False
    for i, frac in enumerate(stall_fraction):
        if frac > 0.7 and not in_range:
            start_i, in_range = i, True
        elif frac <= 0.7 and in_range:
            benign_ranges.append((start_i, i - 1))
            in_range = False
    if in_range:
        benign_ranges.append((start_i, len(stall_fraction) - 1))

    sev_counts = dwell_df["severity"].value_counts().to_dict() if len(dwell_df) else {}

    lines = []
    lines.append("# Teleop dataset quality report\n")
    lines.append(f"- dataset: `{dataset_path}`")
    lines.append(f"- codebase_version: {info.get('codebase_version')}, fps: {fps}")
    lines.append(f"- episodes: {len(episodes)}, frames: {len(df)}")
    lines.append(f"- EPS used: **{eps:.5f}** (`{'--eps override' if args.eps is not None else 'max(p10 nonzero, p1 nonzero noise floor)'}`)")
    lines.append(f"- dwell-warn-sec: {args.dwell_warn_sec}, dwell-critical-sec: {args.dwell_critical_sec}\n")

    lines.append("## Dwell segment counts by severity\n")
    for sev in ["minor", "warn", "critical"]:
        lines.append(f"- {sev}: {sev_counts.get(sev, 0)}")
    lines.append("")

    lines.append("## Recommended for re-recording\n")
    if len(rerecord):
        for _, r in rerecord.iterrows():
            reasons = []
            if r.n_critical > 0:
                crit_segs = dwell_df[(dwell_df.episode_index == r.episode_index) & (dwell_df.severity == "critical")]
                ts = ", ".join(f"{s.start_timestamp:.2f}-{s.end_timestamp:.2f}s" for _, s in crit_segs.iterrows())
                reasons.append(f"{r.n_critical} critical dwell(s) at [{ts}]")
            if r.reversal_rate >= reversal_p90:
                reasons.append(f"reversal_rate={r.reversal_rate:.2f}/s (top 10%)")
            lines.append(f"- episode {r.episode_index}: " + "; ".join(reasons))
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Trim-only fixable (leading/trailing dwell only)\n")
    if trim_eps:
        for ep in sorted(trim_eps):
            segs = dwell_df[(dwell_df.episode_index == ep) & (dwell_df.leading | dwell_df.trailing)]
            for _, s in segs.iterrows():
                tag = "leading" if s.leading else "trailing"
                lines.append(f"- episode {ep}: {tag} dwell {s.start_timestamp:.2f}-{s.end_timestamp:.2f}s ({s.severity})")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Consistent stop phases (benign)\n")
    if benign_ranges:
        for lo, hi in benign_ranges:
            phase_lo, phase_hi = lo / 100, (hi + 1) / 100
            overlapping = dwell_df[(dwell_df.phase_start >= phase_lo - 0.05) & (dwell_df.phase_start <= phase_hi + 0.05)] if len(dwell_df) else dwell_df
            gripper_frac = overlapping["benign_gripper"].mean() if len(overlapping) else 0.0
            cause = "gripper 동작으로 추정" if gripper_frac > 0.5 else "원인 불명, 확인 필요"
            lines.append(f"- phase {phase_lo:.2f}-{phase_hi:.2f}: {stall_fraction[lo:hi+1].mean()*100:.0f}% of episodes dwell here ({cause})")
    else:
        lines.append("- (none found >70% consistency)")
    lines.append("")

    lines.append("## Episode length outliers (>1.5x median)\n")
    outliers = ep_summary[ep_summary.is_outlier]
    if len(outliers):
        for _, r in outliers.iterrows():
            lines.append(f"- episode {r.episode_index}: {r.duration_sec:.2f}s (median {median_dur:.2f}s)")
    else:
        lines.append("- (none)")
    lines.append("")

    n_critical_eps = int((ep_summary.n_critical > 0).sum())
    if n_critical_eps > 0:
        verdict = f"품질: 재녹화 필요 에피소드 {len(rerecord)}개 존재 — 학습 전 재녹화/제외 권장."
    elif len(trim_eps) > 0:
        verdict = "품질: 경계 수준 — 트리밍으로 해결 가능한 에피소드가 있음, 검토 후 학습 진행."
    else:
        verdict = "품질: 양호 — 학습 진행 가능."
    lines.append(f"## 결론\n\n{verdict}\n")

    report_path = args.out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print("\n" + "=" * 60)
    print(report_path.read_text())


if __name__ == "__main__":
    main()
