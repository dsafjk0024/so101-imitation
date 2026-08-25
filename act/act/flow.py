"""Flow-matching Behavior Cloning (FBC) — config + network.

FBC is ACT with the generative head swapped: the CVAE (latent token + KL term)
is replaced by a rectified-flow velocity field, and the L1 action regression by
an MSE velocity regression. Everything upstream of the head -- ResNet backbone,
transformer encoder over [conditioning token, robot-state token, image feature
tokens], transformer decoder with `chunk_size` queries -- is reused verbatim
from `act.model` so a FBC-vs-ACT comparison isolates the objective and not the
architecture.

Why the objective matters: ACT regresses a chunk with L1, so when the
demonstrations are multimodal (the same observation preceding two different
valid motions) it is pulled toward the average of those motions. Flow matching
learns a velocity field transporting noise to the action distribution, so
sampling commits to one mode instead of averaging.

The two structural changes versus `act.model.ACT`:

  encoder token 0    latent sample z  ->  timestep embedding of t
  decoder input      zeros            ->  the noised action chunk x_t

`t` also gets added to every decoder input token, so the velocity field sees the
flow time directly rather than only through cross-attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import chain

import einops
import torch
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from act.config import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from act.model import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
)


@dataclass
class FBCConfig:
    """Configuration for the Flow-matching Behavior Cloning policy.

    Architecture fields mirror `ACTConfig` field-for-field (the shared
    transformer modules read them by name), minus the CVAE block and plus the
    flow block. Defaults are ACT's defaults so the two policies differ only in
    the objective.
    """

    # --- input / output structure (the only per-robot part) ---
    action_dim: int = 6
    robot_state_dim: int | None = 6
    env_state_dim: int | None = None
    image_keys: tuple[str, ...] = ()

    # --- chunking ---
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    # --- vision backbone ---
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False

    # --- transformer ---
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Kept at 1 to match ACT: the original ACT sets 7 but a bug means only the
    # first decoder layer runs, and `act.model` matches that. Holding it equal
    # keeps the comparison about the objective.
    n_decoder_layers: int = 1

    # --- flow matching ---
    # Euler steps used to integrate the ODE at sampling time. Inference-only:
    # it does not enter training, so it can be changed on a trained checkpoint.
    # Rectified-flow paths are close to straight, so 10 is generally enough;
    # raise it if sampled chunks look under-integrated.
    flow_steps: int = 10
    # Width of the sinusoidal timestep embedding before it is projected to
    # `dim_model`.
    time_embed_dim: int = 128

    # --- inference ---
    temporal_ensemble_coeff: float | None = None

    # --- training / loss ---
    dropout: float = 0.1
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        self.image_keys = tuple(self.image_keys)
        self.validate_architecture()

    @property
    def has_images(self) -> bool:
        return len(self.image_keys) > 0

    @property
    def has_robot_state(self) -> bool:
        return self.robot_state_dim is not None

    @property
    def has_env_state(self) -> bool:
        return self.env_state_dim is not None

    def validate_architecture(self) -> None:
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be a ResNet variant. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling: the policy "
                "must be queried every step to form the ensemble."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})."
            )
        if self.n_obs_steps != 1:
            raise ValueError(f"Only n_obs_steps=1 is supported. Got {self.n_obs_steps}.")
        if self.flow_steps < 1:
            raise ValueError(f"`flow_steps` must be >= 1. Got {self.flow_steps}.")

    def validate_features(self) -> None:
        if not self.has_images and not self.has_env_state:
            raise ValueError("Provide at least one of `image_keys` or `env_state_dim`.")


class SinusoidalTimeEmbedding(nn.Module):
    """Transformer-style sinusoidal embedding of a continuous flow time in [0, 1].

    `t` is scaled by 1000 before the usual frequency ladder so that the range
    actually used (a unit interval) spans a useful part of the ladder, which is
    what diffusion/flow implementations conventionally do.
    """

    def __init__(self, dimension: int):
        super().__init__()
        if dimension % 2 != 0:
            raise ValueError(f"`dimension` must be even. Got {dimension}.")
        self.dimension = dimension

    def forward(self, t: Tensor) -> Tensor:
        """(B,) float times in [0, 1] -> (B, dimension)."""
        half = self.dimension // 2
        exponents = torch.arange(half, dtype=torch.float32, device=t.device) / half
        freqs = torch.exp(-math.log(10000.0) * exponents)
        angles = t.unsqueeze(-1).float() * 1000.0 * freqs.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class FBC(nn.Module):
    """Conditional velocity field v(x_t, t | observation) over action chunks.

    A forward pass predicts the flow velocity for one noised chunk at one time.
    It does not sample -- integration lives in `FBCPolicy` so that the module
    stays a plain single-pass network (and so the sampler is testable on its
    own).
    """

    def __init__(self, config: FBCConfig):
        super().__init__()
        self.config = config

        # Backbone for image feature extraction. Identical to act.model.ACT.
        if self.config.has_images:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[
                    False,
                    False,
                    config.replace_final_stride_with_dilation,
                ],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(
                backbone_model, return_layers={"layer4": "feature_map"}
            )

        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Flow-time conditioning. The embedding feeds both encoder token 0
        # (global conditioning, occupying the slot ACT used for the VAE latent)
        # and every decoder input token (so the velocity head sees t directly).
        self.time_embed = SinusoidalTimeEmbedding(config.time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embed_dim, config.dim_model),
            nn.SiLU(),
            nn.Linear(config.dim_model, config.dim_model),
        )

        # Transformer encoder input projections. Tokens are structured like
        # [time, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.config.has_robot_state:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.robot_state_dim, config.dim_model
            )
        if self.config.has_env_state:
            self.encoder_env_state_input_proj = nn.Linear(
                config.env_state_dim, config.dim_model
            )
        if self.config.has_images:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )

        n_1d_tokens = 1  # for the flow time
        if self.config.has_robot_state:
            n_1d_tokens += 1
        if self.config.has_env_state:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.has_images:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(
                config.dim_model // 2
            )

        # Decoder: learnable positional embedding per chunk step (as in ACT),
        # but the content of each decoder input token is now the corresponding
        # noised action rather than a zero vector.
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.decoder_action_input_proj = nn.Linear(config.action_dim, config.dim_model)

        # Velocity head, in place of ACT's action head.
        self.velocity_head = nn.Linear(config.dim_model, config.action_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform init of the transformer parameters, as ACT does."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor], x_t: Tensor, t: Tensor) -> Tensor:
        """Predict the flow velocity at `(x_t, t)` given the observation.

        Args:
            batch: observation dict, same convention as `act.model.ACT`:
                [OBS_STATE] (optional): (B, state_dim)
                [OBS_IMAGES]: list of (B, C, H, W), one per camera
                AND/OR [OBS_ENV_STATE]: (B, env_dim)
                The [ACTION] key is not read -- the chunk enters through `x_t`.
            x_t: (B, chunk_size, action_dim) noised action chunk.
            t: (B,) flow times in [0, 1].

        Returns:
            (B, chunk_size, action_dim) predicted velocity.
        """
        batch_size = x_t.shape[0]
        time_token = self.time_mlp(self.time_embed(t))  # (B, dim_model)

        encoder_in_tokens = [time_token]
        encoder_in_pos_embed = list(
            self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)
        )
        if self.config.has_robot_state:
            encoder_in_tokens.append(
                self.encoder_robot_state_input_proj(batch[OBS_STATE])
            )
        if self.config.has_env_state:
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(batch[OBS_ENV_STATE])
            )

        if self.config.has_images:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(
                    dtype=cam_features.dtype
                )
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        # (B, S, A) -> (S, B, D), plus the flow time broadcast over the chunk.
        decoder_in = self.decoder_action_input_proj(x_t).transpose(0, 1)
        decoder_in = decoder_in + time_token.unsqueeze(0)

        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)
        return self.velocity_head(decoder_out)
