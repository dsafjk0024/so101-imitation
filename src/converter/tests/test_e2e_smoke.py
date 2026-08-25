"""Seam 1 -- tracer-bullet smoke test for the full software chain.

fixture 5-episode rosbags -> converter -> LeRobot v3.0 dataset -> act/'s own
training loop (act/tools/train_so101.py, a couple of steps) -> checkpoint
saved in act/'s own {"model", "config", "step"} torch.save format + stats.json
-> reload -> one dummy-observation inference -> sane action output.

No real hardware, no real training signal expected: this only proves the
plumbing connects end to end. Needs both environments, like test_convert.py:

    source /opt/ros/jazzy/setup.bash
    cd <repo root>
    pixi run -e lerobot python -m pytest src/converter/tests -v
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rosbag2_py", reason="source /opt/ros/jazzy/setup.bash before running Seam 1")

from test_convert import IMG_H, IMG_W, REPO_ID, make_test_cfg, write_fixture_bag  # noqa: E402

from converter.convert import convert_all  # noqa: E402

NUM_EPISODES = 5


@pytest.mark.slow
def test_fixture_to_checkpoint_to_inference(tmp_path):
    import torch

    from act.checkpoint import build_normalizer, load_stats, normalize_batch
    from act.config import ACTION, OBS_STATE
    from act.policy import ACTPolicy
    from tools.train_so101 import train

    bags_root = tmp_path / "bags"
    bags_root.mkdir()
    for i in range(NUM_EPISODES):
        write_fixture_bag(bags_root / f"ep_{i:03d}", duration_s=1.0)

    dataset_dir = tmp_path / "dataset"
    report = convert_all(bags_root, make_test_cfg(), dataset_dir, use_videos=False)
    assert report["episodes_converted"] == NUM_EPISODES

    train_out = tmp_path / "train_out"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # chunk_size well under the ~31-frame fixture episodes; num_workers=0 to
    # avoid multiprocessing overhead/segfault risk (see DatasetReader._query_videos
    # docstring) for a 2-step smoke run.
    ckpt_path = train(
        dataset_dir,
        train_out,
        repo_id=REPO_ID,
        steps=2,
        batch_size=4,
        chunk_size=10,
        num_workers=0,
        device=device,
    )

    assert ckpt_path.is_file()
    stats_path = train_out / "stats.json"
    assert stats_path.exists()

    saved = torch.load(ckpt_path, weights_only=False)
    assert set(saved.keys()) == {"model", "config", "step"}

    policy = ACTPolicy.from_checkpoint(ckpt_path, device=torch.device(device))

    stats = load_stats(stats_path)
    norm = build_normalizer(stats, (OBS_STATE, ACTION), policy.config.image_keys, torch.device(device))

    rng = np.random.default_rng(0)
    observation = {
        "observation.state": torch.tensor(
            np.linspace(0.1, 0.6, 6, dtype=np.float32)
        ).unsqueeze(0),
        "observation.images.wrist": torch.tensor(
            rng.random((1, 3, IMG_H, IMG_W), dtype=np.float32)
        ),
        "observation.images.top": torch.tensor(
            rng.random((1, 3, IMG_H, IMG_W), dtype=np.float32)
        ),
    }
    observation = {k: v.to(device) for k, v in observation.items()}
    observation = normalize_batch(observation, norm)

    action = policy.select_action(observation)
    action = action.squeeze(0).cpu().numpy()

    assert action.shape == (6,)
    assert np.isfinite(action).all()
    assert np.all(np.abs(action) < 1e3)
