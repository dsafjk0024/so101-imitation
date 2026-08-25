"""Train Flow-matching BC (FBC) on a LeRobotDataset v3.0 dataset.

Deliberately a near-copy of `tools/train_so101.py`: same dataset plumbing, same
`delta_timestamps` chunking, same mean/std normalization, same checkpoint
format. Only the policy differs (`FBCPolicy` instead of `ACTPolicy`), so a run
of this script and a run of `train_so101.py` over the same dataset differ in the
training objective and nothing else -- which is the point, since the two are
meant to be compared.

Note on normalization: actions are mean/std normalized (as for ACT), not
min-max'd into [-1, 1] as the reference JAX implementation does. Unit-ish
variance is what the flow's N(0, I) source distribution expects, and it keeps
the comparison against ACT on identical inputs.

Run:
    python tools/train_fbc_so101.py --data-dir /path/to/dataset --steps 200000 \
        --batch-size 32 --save-freq 25000 --out outputs/fbc_so101
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from act import (
    ACTION,
    OBS_STATE,
    FBCConfig,
    FBCPolicy,
    build_normalizer,
    load_checkpoint,
    normalize_batch,
    reduce_lerobot_stats,
    save_checkpoint,
    save_stats,
)

VECTOR_KEYS = (OBS_STATE, ACTION)


def train(
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    repo_id: str = "local/so101",
    steps: int = 200000,
    batch_size: int = 32,
    chunk_size: int = 100,
    flow_steps: int = 10,
    lr: float | None = None,
    device: str | None = None,
    num_workers: int = 4,
    log_freq: int = 50,
    save_freq: int = 0,
    video_backend: str | None = "pyav",
    init_from: str | Path | None = None,
) -> Path:
    """Train FBC on a LeRobotDataset root. Returns the path to the final
    checkpoint. Importable so tests/other scripts don't need to shell out."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=data_dir)
    image_keys = tuple(sorted(meta.camera_keys))
    action_dim = meta.shapes[ACTION][0]
    robot_state_dim = meta.shapes[OBS_STATE][0]

    stats = reduce_lerobot_stats(meta.stats, VECTOR_KEYS, image_keys)
    save_stats(out_dir / "stats.json", stats)
    norm = build_normalizer(stats, VECTOR_KEYS, image_keys, device)

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=data_dir,
        delta_timestamps={ACTION: [i / meta.fps for i in range(chunk_size)]},
        video_backend=video_backend,
    )
    print(f"train: {len(dataset)} frames, {dataset.num_episodes} episodes, image_keys={image_keys}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    config = FBCConfig(
        action_dim=action_dim,
        robot_state_dim=robot_state_dim,
        image_keys=image_keys,
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        flow_steps=flow_steps,
    )
    if lr is not None:
        config.optimizer_lr = lr
    policy = FBCPolicy(config).to(device)
    policy.train()

    # Continue from an earlier run's weights. Only the model is restored -- the
    # checkpoint carries no optimizer state, so AdamW's moments rebuild over the
    # first ~1k steps. Harmless because the LR is constant (no schedule to
    # resume). The step counter picks up where the checkpoint left off so
    # `save_freq` filenames stay on one continuous series.
    start_step = 0
    if init_from is not None:
        ckpt = load_checkpoint(init_from, device=device)
        policy.load_state_dict(ckpt["model"])
        start_step = int(ckpt["step"])
        print(f"init from {init_from} at step {start_step}")

    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=config.optimizer_lr,
        weight_decay=config.optimizer_weight_decay,
    )

    def save(name: str) -> Path:
        ckpt = out_dir / name
        save_checkpoint(ckpt, policy, config.__dict__, step)
        tqdm.write(f"saved {ckpt}")
        return ckpt

    keep_keys = (OBS_STATE, ACTION, "action_is_pad", *image_keys)
    step = start_step
    running = 0.0
    done = False
    pbar = tqdm(total=steps, initial=step, desc="train", dynamic_ncols=True)
    while not done:
        for batch in loader:
            batch = {k: batch[k].to(device, non_blocking=True) for k in keep_keys}
            batch = normalize_batch(batch, norm)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running += loss.item()
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if step % log_freq == 0:
                tqdm.write(f"step {step:5d}  loss {running / log_freq:.4f}  {loss_dict}")
                running = 0.0
            if save_freq and step % save_freq == 0:
                save(f"step_{step:07d}.pt")
            if step >= steps:
                done = True
                break
    pbar.close()

    return save("fbc_so101.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="LeRobotDataset v3.0 root (converter output)")
    ap.add_argument("--repo-id", default="local/so101", help="only used for LeRobotDataset bookkeeping")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--flow-steps", type=int, default=10, help="Euler steps at sampling time (inference only)")
    ap.add_argument("--lr", type=float, default=None, help="override config.optimizer_lr")
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-freq", type=int, default=50)
    ap.add_argument("--save-freq", type=int, default=0, help="checkpoint every N steps (0 = only at end)")
    ap.add_argument("--video-backend", default="pyav", help="lerobot video decode backend (torchcodec fails on our av1 mp4s)")
    ap.add_argument("--init-from", default=None, help="checkpoint to continue from (model weights + step; no optimizer state)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train(
        args.data_dir,
        args.out,
        repo_id=args.repo_id,
        steps=args.steps,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        flow_steps=args.flow_steps,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        video_backend=args.video_backend,
        init_from=args.init_from,
    )


if __name__ == "__main__":
    main()
