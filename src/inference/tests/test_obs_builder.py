"""Seam 4 -- obs_builder output must exactly match the training dataset
contract in `so101.yaml`. Messages are stand-in objects (see
`converter/tests/test_decoders.py`'s FakeImage/FakeJointState convention) so
this runs without a ROS environment.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pytest

from converter.config import DEFAULT_CONFIG_PATH, load_config
from converter.decoders import IMAGE_MSG_TYPE

from inference.obs_builder import STATE_KEY, TOP_KEY, WRIST_KEY, build_observation

CANONICAL = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Deliberately not in canonical order -- proves reordering isn't
# accidentally-correct-by-luck (same convention as test_convert.py's SCRAMBLED).
SCRAMBLED = ("gripper", "elbow_flex", "shoulder_pan", "wrist_roll", "shoulder_lift", "wrist_flex")

IMG_H, IMG_W = 480, 640


@dataclass
class FakeImage:
    height: int
    width: int
    encoding: str
    step: int
    data: bytes


@dataclass
class FakeJointState:
    name: list[str] = field(default_factory=list)
    position: list[float] = field(default_factory=list)


def make_image(h=IMG_H, w=IMG_W, fill_value=0):
    arr = np.full((h, w, 3), fill_value, dtype=np.uint8)
    return FakeImage(height=h, width=w, encoding="rgb8", step=w * 3, data=arr.tobytes())


def make_scrambled_joint_state():
    canonical_positions = {name: float(i) / 10 for i, name in enumerate(CANONICAL)}
    return FakeJointState(
        name=list(SCRAMBLED),
        position=[canonical_positions[n] for n in SCRAMBLED],
    )


@pytest.fixture
def cfg():
    return load_config(DEFAULT_CONFIG_PATH)


def test_observation_has_exactly_the_expected_keys(cfg):
    obs = build_observation(make_scrambled_joint_state(), make_image(), make_image(), cfg)
    assert set(obs) == {STATE_KEY, WRIST_KEY, TOP_KEY}


def test_state_is_canonical_order_not_scrambled_order(cfg):
    obs = build_observation(make_scrambled_joint_state(), make_image(), make_image(), cfg)
    state = obs[STATE_KEY]
    assert state.dtype == np.float32
    assert state.shape == (6,)
    expected = [i / 10 for i in range(6)]  # canonical-order values
    np.testing.assert_allclose(state, expected, atol=1e-6)


def test_images_have_dataset_shape_and_dtype(cfg):
    wrist = make_image(fill_value=1)
    top = make_image(fill_value=2)
    obs = build_observation(make_scrambled_joint_state(), wrist, top, cfg)

    for key in (WRIST_KEY, TOP_KEY):
        assert obs[key].dtype == np.uint8
        assert obs[key].shape == (IMG_H, IMG_W, 3)

    # not just shape/dtype-equal -- actually the right image in the right slot
    assert obs[WRIST_KEY][0, 0, 0] == 1
    assert obs[TOP_KEY][0, 0, 0] == 2


def test_default_cfg_loads_real_so101_yaml_contract():
    # No cfg passed -> falls back to converter/config/so101.yaml, the single
    # source of truth. If this ever needs a hardcoded fallback, that's the bug.
    obs = build_observation(make_scrambled_joint_state(), make_image(), make_image())
    assert obs[STATE_KEY].shape == (6,)


# --- contract-driven, not hardcoded -----------------------------------------


def test_joint_order_is_driven_by_config_not_hardcoded(cfg):
    """Swap two joint names in the config's observation.state feature and
    confirm build_observation's output changes accordingly for the same raw
    JointState message -- proving the reorder logic reads the config rather
    than an internal copy of the canonical list."""
    state_spec = next(f for f in cfg.features if f.key == STATE_KEY)
    swapped_names = list(state_spec.names)
    swapped_names[0], swapped_names[-1] = swapped_names[-1], swapped_names[0]  # pan <-> gripper
    swapped_spec = dataclasses.replace(state_spec, names=tuple(swapped_names))
    swapped_features = tuple(
        swapped_spec if f.key == STATE_KEY else f for f in cfg.features
    )
    swapped_cfg = dataclasses.replace(cfg, features=swapped_features)

    msg = make_scrambled_joint_state()
    obs_real = build_observation(msg, make_image(), make_image(), cfg)
    obs_swapped = build_observation(msg, make_image(), make_image(), swapped_cfg)

    assert obs_real[STATE_KEY][0] != obs_swapped[STATE_KEY][0]
    assert obs_real[STATE_KEY][-1] != obs_swapped[STATE_KEY][-1]
    # slot 0 of the swapped output now holds the value for "gripper"
    assert obs_swapped[STATE_KEY][0] == obs_real[STATE_KEY][-1]
    assert obs_swapped[STATE_KEY][-1] == obs_real[STATE_KEY][0]


def test_image_shape_mismatch_with_config_fails_loudly(cfg):
    """A wrong-shaped image must raise, not silently corrupt the observation --
    this is the '불일치 시 테스트 실패' guarantee for the image half of the contract."""
    wrong_shape_wrist = make_image(h=240, w=320)
    with pytest.raises(ValueError, match="!="):
        build_observation(make_scrambled_joint_state(), wrong_shape_wrist, make_image(), cfg)


def test_config_image_shape_feeds_the_check(cfg):
    """Mutate the configured shape itself (not the input image) and confirm a
    previously-valid image now fails -- proves the check is config-driven."""
    wrist_spec = next(f for f in cfg.features if f.key == WRIST_KEY)
    mutated_spec = dataclasses.replace(wrist_spec, shape=(240, 320, 3))
    mutated_features = tuple(
        mutated_spec if f.key == WRIST_KEY else f for f in cfg.features
    )
    mutated_cfg = dataclasses.replace(cfg, features=mutated_features)

    with pytest.raises(ValueError, match="!="):
        build_observation(make_scrambled_joint_state(), make_image(), make_image(), mutated_cfg)


def test_wrist_and_top_feature_specs_are_real_image_features(cfg):
    by_key = {f.key: f for f in cfg.features}
    assert by_key[WRIST_KEY].msg_type == IMAGE_MSG_TYPE
    assert by_key[TOP_KEY].msg_type == IMAGE_MSG_TYPE
    assert by_key[WRIST_KEY].shape == (IMG_H, IMG_W, 3)
    assert by_key[TOP_KEY].shape == (IMG_H, IMG_W, 3)
