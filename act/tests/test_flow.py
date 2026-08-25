"""FBC test: flow-matching loss, ODE sampling, action queue, checkpoint round-trip.

Runs on CPU with tiny inputs and no pretrained weights (offline-safe).
Standalone-runnable: `python tests/test_flow.py` (or via pytest).
"""

import torch

from act.checkpoint import save_checkpoint
from act.config import ACTION, OBS_STATE
from act.flow import FBCConfig, SinusoidalTimeEmbedding
from act.flow_policy import FBCPolicy

B, CHUNK, STATE, ACT_DIM = 2, 8, 6, 6
H = W = 64
CAMS = ("observation.images.wrist", "observation.images.external")


def _cfg(**kw):
    base = dict(
        action_dim=ACT_DIM,
        robot_state_dim=STATE,
        image_keys=CAMS,
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        pretrained_backbone_weights=None,
        flow_steps=4,
    )
    base.update(kw)
    return FBCConfig(**base)


def _batch(cfg, with_action):
    b = {OBS_STATE: torch.randn(B, STATE)}
    for key in cfg.image_keys:
        b[key] = torch.rand(B, 3, H, W)
    if with_action:
        b[ACTION] = torch.randn(B, CHUNK, ACT_DIM)
        b["action_is_pad"] = torch.zeros(B, CHUNK, dtype=torch.bool)
    return b


def test_time_embedding_shape_and_distinctness():
    embed = SinusoidalTimeEmbedding(128)
    out = embed(torch.tensor([0.0, 0.5, 1.0]))
    assert out.shape == (3, 128), out.shape
    # Distinct times must give distinct embeddings, else the velocity field
    # cannot tell where it is along the path.
    assert not torch.allclose(out[0], out[1])
    assert not torch.allclose(out[1], out[2])


def test_forward_returns_loss():
    cfg = _cfg()
    policy = FBCPolicy(cfg).train()
    loss, loss_dict = policy(_batch(cfg, with_action=True))
    assert loss.ndim == 0 and loss.requires_grad, loss
    assert "flow_loss" in loss_dict
    assert loss.item() > 0.0


def test_loss_ignores_padded_steps():
    """A fully padded chunk contributes nothing, so the loss must be 0."""
    cfg = _cfg()
    policy = FBCPolicy(cfg).train()
    batch = _batch(cfg, with_action=True)
    batch["action_is_pad"] = torch.ones(B, CHUNK, dtype=torch.bool)
    loss, _ = policy(batch)
    assert loss.item() == 0.0, loss.item()


def test_gradients_reach_backbone_and_head():
    cfg = _cfg()
    policy = FBCPolicy(cfg).train()
    loss, _ = policy(_batch(cfg, with_action=True))
    loss.backward()
    head_grad = policy.model.velocity_head.weight.grad
    assert head_grad is not None and head_grad.abs().sum() > 0
    # The backbone must be in the graph, otherwise images are being ignored.
    backbone_grads = [
        p.grad for p in policy.model.backbone.parameters() if p.requires_grad and p.grad is not None
    ]
    assert backbone_grads and any(g.abs().sum() > 0 for g in backbone_grads)


def test_sampling_shape_and_finiteness():
    cfg = _cfg()
    policy = FBCPolicy(cfg).eval()
    chunk = policy.predict_action_chunk(_batch(cfg, with_action=False))
    assert chunk.shape == (B, CHUNK, ACT_DIM), chunk.shape
    assert torch.isfinite(chunk).all()


def test_sampling_is_stochastic():
    """Two samples from the same observation must differ -- that is the whole
    point of a generative head versus L1 regression."""
    cfg = _cfg()
    policy = FBCPolicy(cfg).eval()
    batch = _batch(cfg, with_action=False)
    a = policy.predict_action_chunk(batch)
    b = policy.predict_action_chunk(batch)
    assert not torch.allclose(a, b)


def test_flow_steps_control_integration_count():
    """`flow_steps` must actually drive the number of velocity evaluations."""
    cfg = _cfg(flow_steps=3)
    policy = FBCPolicy(cfg).eval()
    calls = []
    inner = policy.model.forward

    def counting_forward(batch, x_t, t):
        calls.append(float(t[0]))
        return inner(batch, x_t, t)

    policy.model.forward = counting_forward
    policy.predict_action_chunk(_batch(cfg, with_action=False))
    assert len(calls) == 3, calls
    # Times must sweep 0 -> (flow_steps-1)/flow_steps (float32, so approximate).
    expected = [0.0, 1 / 3, 2 / 3]
    assert all(abs(a - b) < 1e-6 for a, b in zip(calls, expected)), calls


def test_action_queue_drains_then_refills():
    cfg = _cfg(n_action_steps=CHUNK)
    policy = FBCPolicy(cfg).eval()
    batch = _batch(cfg, with_action=False)
    for _ in range(CHUNK):
        action = policy.select_action(batch)
        assert action.shape == (B, ACT_DIM), action.shape
    assert len(policy._action_queue) == 0
    # Next call must repopulate rather than raise.
    assert policy.select_action(batch).shape == (B, ACT_DIM)


def test_checkpoint_round_trip(tmp_path=None):
    import tempfile
    from pathlib import Path

    cfg = _cfg()
    policy = FBCPolicy(cfg).eval()
    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    path = d / "fbc.pt"
    save_checkpoint(path, policy, cfg.__dict__, 123)

    restored = FBCPolicy.from_checkpoint(path)
    assert restored.config.chunk_size == cfg.chunk_size
    assert restored.config.flow_steps == cfg.flow_steps
    # Weights must match, so a restored policy is the same function.
    for (n1, p1), (n2, p2) in zip(
        policy.state_dict().items(), restored.state_dict().items()
    ):
        assert n1 == n2
        assert torch.equal(p1, p2), n1


def test_overfits_single_batch():
    """The real check: can the objective learn at all? Fit one fixed batch and
    require the loss to drop substantially. Catches sign errors, detached
    graphs, and conditioning that never reaches the head."""
    torch.manual_seed(0)
    cfg = _cfg(chunk_size=4, n_action_steps=4)
    policy = FBCPolicy(cfg).train()
    batch = _batch(cfg, with_action=True)
    batch[ACTION] = torch.randn(B, 4, ACT_DIM)
    batch["action_is_pad"] = torch.zeros(B, 4, dtype=torch.bool)

    opt = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)
    # Fix the noise/time draw so the objective is deterministic; otherwise the
    # per-step loss is dominated by resampling and this test is just noise.
    torch.manual_seed(1)
    x_0 = torch.randn_like(batch[ACTION])
    t = torch.rand(B)
    target = batch[ACTION] - x_0
    x_t = x_0 + t[:, None, None] * target
    prepared = policy._prepare_images(batch)

    losses = []
    for _ in range(60):
        pred = policy.model(prepared, x_t, t)
        loss = (pred - target).pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, (losses[0], losses[-1])


if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            if fn.__name__ == "test_checkpoint_round_trip":
                fn(tmp_path=Path(tempfile.mkdtemp()))
            else:
                fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
