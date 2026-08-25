"""FBC policy wrapper (flow-matching loss, Euler sampling, action queue).

Mirrors `act.policy.ACTPolicy` so the two are drop-in interchangeable for
training (`forward` -> `(loss, loss_dict)`) and inference (`select_action` /
`predict_action_chunk`). The action queue and temporal ensembler are reused from
`act.policy` unchanged.

Training objective (rectified flow / conditional flow matching):

    x_0 ~ N(0, I)                       noise, shaped like an action chunk
    x_1 = batch[ACTION]                 the demonstrated chunk (normalized)
    t   ~ U(0, 1)                       one time per batch element
    x_t = (1 - t) * x_0 + t * x_1       straight-line probability path
    loss = MSE( v(x_t, t | obs),  x_1 - x_0 )

The target velocity `x_1 - x_0` is constant along each path, which is what makes
the learned paths near-straight and lets sampling use few Euler steps.

Sampling integrates the ODE from t=0 to t=1:

    x <- x + v(x, t) / flow_steps       repeated `flow_steps` times

Padding: chunks near the end of an episode are padded by LeRobot, flagged in
`batch["action_is_pad"]`. Those entries are excluded from the loss (as ACT
does), but they are still fed to the network as part of `x_t` -- masking the
input would change the sequence the transformer sees between training and
inference.
"""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor, nn

from act.checkpoint import load_checkpoint
from act.config import ACTION, OBS_IMAGES
from act.flow import FBC, FBCConfig
from act.policy import ACTTemporalEnsembler


class FBCPolicy(nn.Module):
    """Flow-matching Behavior Cloning policy: wraps the `FBC` velocity network
    for training (loss computation) and inference (ODE sampling + action queue)."""

    def __init__(self, config: FBCConfig):
        super().__init__()
        config.validate_architecture()
        config.validate_features()
        self.config = config

        self.model = FBC(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )

        self.reset()

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None) -> "FBCPolicy":
        """Load a policy saved via `act.checkpoint.save_checkpoint`."""
        saved = load_checkpoint(checkpoint_path, device=device)
        policy = cls(FBCConfig(**saved["config"]))
        if device is not None:
            policy = policy.to(device)
        policy.load_state_dict(saved["model"])
        policy.eval()
        return policy

    def get_optim_params(self) -> list[dict]:
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def _prepare_images(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.has_images:
            batch = dict(batch)  # shallow copy so adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_keys]
        return batch

    @torch.no_grad()
    def sample_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Integrate the flow ODE to draw one action chunk per batch element.

        `batch` must already carry OBS_IMAGES if the config uses cameras.
        Returns (B, chunk_size, action_dim).
        """
        ref = (
            batch[OBS_IMAGES][0]
            if OBS_IMAGES in batch
            else next(iter(batch.values()))
        )
        batch_size = ref.shape[0]
        x = torch.randn(
            (batch_size, self.config.chunk_size, self.config.action_dim),
            dtype=torch.float32,
            device=ref.device,
        )
        for i in range(self.config.flow_steps):
            t = torch.full(
                (batch_size,),
                i / self.config.flow_steps,
                dtype=torch.float32,
                device=x.device,
            )
            x = x + self.model(batch, x, t) / self.config.flow_steps
        return x

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        Wraps `predict_action_chunk` to return one action at a time, refilling a
        queue whenever it empties.
        """
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # The queue is effectively (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()
        return self.sample_action_chunk(self._prepare_images(batch))

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the flow-matching loss."""
        batch = self._prepare_images(batch)

        x_1 = batch[ACTION]
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], dtype=x_1.dtype, device=x_1.device)

        target_velocity = x_1 - x_0
        x_t = x_0 + t[:, None, None] * target_velocity

        pred_velocity = self.model(batch, x_t, t)

        sq_err = (pred_velocity - target_velocity).pow(2)
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * sq_err.shape[-1]
        flow_loss = (sq_err * valid_mask).sum() / num_valid.clamp_min(1)

        return flow_loss, {"flow_loss": flow_loss.item()}
