"""Sync inference node: observation in, joint-command action out.

Loads a standalone `act/` checkpoint -- the `{"model", "config", "step"}`
torch.save produced by `act/tools/train_so101.py` (ACT) or
`act/tools/train_fbc_so101.py` (FBC), with a `stats.json` sibling in the same
directory holding MEAN_STD normalization stats -- then subscribes to the
follower's joint state and two cameras, and at a fixed control rate assembles
an observation (via `inference.obs_builder`, sharing the exact decode path the
converter uses to build the training dataset -- see Seam 4), normalizes it the
same way training did, runs `select_action`, and publishes the resulting joint
command in canonical order.

Both policies work here unchanged: FBCPolicy mirrors ACTPolicy's inference
surface (`from_checkpoint`, `select_action`, `reset`, and the `chunk_size` /
`n_action_steps` config fields), so only the class differs. Which one to build
is read off the checkpoint's own config rather than a parameter -- see the
`flow_steps` check below.

Checkpoint loading and MEAN_STD normalization reuse act/'s shared helpers
(`from_checkpoint`, `act.checkpoint.build_normalizer`/`normalize_batch`) -- the
same ones the training scripts use -- so this node and the training scripts
can't silently drift on the on-disk format.

This node straddles two environments: rclpy comes from the system ROS jazzy
install, while torch/act come from the pixi `lerobot` env. Run it as
(verified: importing `inference`/`converter`/`act` this way works once all
environments' sys.path are combined; `python -m inference.sync_inference_node`
is the invocation confirmed to resolve the package -- running the file
directly by path would not, since it uses an absolute `inference.` import):

    source /opt/ros/jazzy/setup.bash
    source install/setup.bash   # after colcon build, so `converter` and `inference` are on PYTHONPATH
    pixi run -e lerobot python -m inference.sync_inference_node --ros-args -p checkpoint_path:=/path/to/act_so101.pt

`checkpoint_path` must point at the checkpoint file itself (e.g.
`.../act_so101.pt`); its `stats.json` sibling is loaded from the same
directory automatically.

Not verified end-to-end (needs a real checkpoint + real ROS topics, neither of
which exist yet in this repo) -- treat the invocation above as documented but
unverified beyond the two envs' sys.path composing correctly.
"""

from __future__ import annotations

from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

from converter.config import DEFAULT_CONFIG_PATH, load_config

from inference.obs_builder import STATE_KEY, build_observation


class SyncInferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("sync_inference_node")

        cfg = load_config(DEFAULT_CONFIG_PATH)
        by_key = {spec.key: spec for spec in cfg.features}

        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("state_topic", by_key["observation.state"].topic)
        self.declare_parameter("wrist_topic", by_key["observation.images.wrist"].topic)
        self.declare_parameter("top_topic", by_key["observation.images.top"].topic)
        self.declare_parameter("action_topic", by_key["action"].topic)
        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("stale_timeout_s", 0.25)
        self.declare_parameter("task", cfg.task)
        # act/tools/train_so101.py bakes n_action_steps == chunk_size into the
        # checkpoint (100 here): select_action predicts once, then blindly
        # executes the whole queued chunk (~3.3s at 30Hz) before observing
        # again. Any drift from the training distribution during that window
        # -- e.g. the harder, more variable final-placement phase -- can't be
        # corrected until the queue drains, and the next replan starts from
        # an already-off-distribution observation. Overriding it here is
        # purely a runtime queue-length knob (see ACTPolicy.select_action) --
        # it does not touch the trained weights, no retraining needed.
        self.declare_parameter("n_action_steps", 10)
        # Euler integration steps for FBC's ODE sampler. 0 keeps whatever the
        # checkpoint was trained with. No effect on ACT checkpoints.
        self.declare_parameter("flow_steps", 0)

        checkpoint_path = self.get_parameter("checkpoint_path").value
        if not checkpoint_path:
            raise ValueError("checkpoint_path parameter is required")
        checkpoint_path = Path(checkpoint_path)

        state_topic = self.get_parameter("state_topic").value
        wrist_topic = self.get_parameter("wrist_topic").value
        top_topic = self.get_parameter("top_topic").value
        action_topic = self.get_parameter("action_topic").value
        rate = float(self.get_parameter("control_rate_hz").value)
        self.stale_timeout_s = float(self.get_parameter("stale_timeout_s").value)
        self.task = self.get_parameter("task").value

        self.cfg = cfg
        self.joint_names = cfg.joint_names

        # Imported here, not at module scope: rclpy runs under system ROS
        # jazzy's Python, but torch/act only exist in the pixi lerobot env
        # this node is launched from. Deferring the import keeps the module
        # importable (e.g. for `--help`) even if that env is misconfigured.
        import torch

        from act.checkpoint import build_normalizer, load_checkpoint, load_stats, normalize_batch
        from act.config import ACTION

        self._ACTION = ACTION

        self._torch = torch
        self._normalize_batch = normalize_batch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ACT and FBC checkpoints share the {"model", "config", "step"} format,
        # so the policy class is chosen from the config's contents rather than
        # from a parameter the caller could get wrong: `flow_steps` exists only
        # on FBCConfig. Feeding an FBC checkpoint to ACTPolicy would otherwise
        # fail deep inside ACTConfig(**config) with an unexpected-keyword error.
        saved_config = load_checkpoint(checkpoint_path, device=self.device)["config"]
        self.is_flow_policy = "flow_steps" in saved_config
        if self.is_flow_policy:
            from act.flow_policy import FBCPolicy as PolicyClass
        else:
            from act.policy import ACTPolicy as PolicyClass

        self.policy = PolicyClass.from_checkpoint(checkpoint_path, device=self.device)
        n_action_steps = int(self.get_parameter("n_action_steps").value)
        self.policy.config.n_action_steps = n_action_steps

        # FBC samples by integrating an ODE, so its cost per replan scales with
        # the Euler step count. Like n_action_steps this is inference-only and
        # touches no weights, so it stays a runtime knob: lower it if the
        # control loop misses its deadline, raise it if sampled chunks look
        # under-integrated. Ignored for ACT, which is a single forward pass.
        flow_steps = int(self.get_parameter("flow_steps").value)
        if self.is_flow_policy and flow_steps > 0:
            self.policy.config.flow_steps = flow_steps

        self.policy.reset()  # rebuilds the action queue with the new maxlen

        stats = load_stats(checkpoint_path.parent / "stats.json")
        self._norm = build_normalizer(stats, (STATE_KEY, ACTION), self.policy.config.image_keys, self.device)

        self._latest_state: JointState | None = None
        self._latest_wrist: Image | None = None
        self._latest_top: Image | None = None
        self._state_stamp: float | None = None
        self._wrist_stamp: float | None = None
        self._top_stamp: float | None = None

        self.create_subscription(JointState, state_topic, self._on_state, 10)
        self.create_subscription(Image, wrist_topic, self._on_wrist, 10)
        self.create_subscription(Image, top_topic, self._on_top, 10)
        self.pub = self.create_publisher(Float64MultiArray, action_topic, 10)
        self.timer = self.create_timer(1.0 / rate, self._tick)

        policy_desc = type(self.policy).__name__
        if self.is_flow_policy:
            policy_desc += f" (flow_steps={self.policy.config.flow_steps})"
        self.get_logger().info(
            f"sync_inference_node: policy={policy_desc}, checkpoint={checkpoint_path}, "
            f"inputs=({state_topic}, {wrist_topic}, {top_topic}), "
            f"action_topic={action_topic} @ {rate:g}Hz, device={self.device}, "
            f"n_action_steps={n_action_steps} (chunk_size={self.policy.config.chunk_size})"
        )

    def _on_state(self, msg: JointState) -> None:
        self._latest_state = msg
        self._state_stamp = self._now_s()

    def _on_wrist(self, msg: Image) -> None:
        self._latest_wrist = msg
        self._wrist_stamp = self._now_s()

    def _on_top(self, msg: Image) -> None:
        self._latest_top = msg
        self._top_stamp = self._now_s()

    def _tick(self) -> None:
        if self._latest_state is None or self._latest_wrist is None or self._latest_top is None:
            return
        now = self._now_s()
        stamps = (self._state_stamp, self._wrist_stamp, self._top_stamp)
        if any(now - s > self.stale_timeout_s for s in stamps):
            return  # an input is stale -- stop commanding rather than act on old data

        observation = build_observation(
            self._latest_state, self._latest_wrist, self._latest_top, self.cfg
        )
        torch = self._torch
        batch = {STATE_KEY: torch.from_numpy(observation[STATE_KEY]).float().unsqueeze(0).to(self.device)}
        for key in self.policy.config.image_keys:
            img = torch.from_numpy(observation[key]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            batch[key] = img.to(self.device)
        batch = self._normalize_batch(batch, self._norm)

        action = self.policy.select_action(batch)  # already @torch.no_grad()
        # The model was trained to predict actions in the same normalized
        # (mean/std) space normalize_batch put them in during training (see
        # train_so101.py: ACTION is in VECTOR_KEYS, so ground truth was
        # normalized before the loss). select_action's output is therefore
        # still normalized here and must be mapped back to physical joint
        # radians before publishing -- forgetting this step previously sent
        # raw z-scored values straight to the follower's position controller,
        # which is harmless for joints whose mean sits near 0 but is a large,
        # sudden offset for one whose mean doesn't (e.g. wrist_roll, mean
        # ~1.34 rad here) -- exactly the "last joint snaps" symptom seen live.
        mean, std = self._norm[self._ACTION]
        action = action * std + mean
        action = action.squeeze(0).cpu().numpy()
        self.pub.publish(Float64MultiArray(data=[float(v) for v in action]))

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def main() -> None:
    rclpy.init()
    node = SyncInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
