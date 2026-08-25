"""Grid core tests. No ROS, no lerobot — messages are plain tuples.

The tick arithmetic and the emit/push ordering are the easiest things in this
converter to get subtly wrong, so they are pinned here rather than only in the
slower integration test.
"""

from __future__ import annotations

import pytest

from converter.config import Config, FeatureSpec
from converter.grid import NS_PER_S, GridStats, LastBuffer, iter_frames, tick_ns

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

STATE_TOPIC = "/follower/joint_states"
ACTION_TOPIC = "/follower/forward_controller/commands"
WRIST_TOPIC = "/follower/camera/wrist/image_raw"
TOP_TOPIC = "/follower/camera/top/image_raw"


def make_cfg(fps: int = 30) -> Config:
    cfg = Config(
        robot_type="so101",
        fps=fps,
        task="t",
        repo_id="local/test",
        joint_names=JOINTS,
        features=(
            FeatureSpec("observation.state", STATE_TOPIC,
                        "sensor_msgs/msg/JointState", 0.05, names=JOINTS),
            FeatureSpec("action", ACTION_TOPIC,
                        "std_msgs/msg/Float64MultiArray", 0.05, names=JOINTS),
            FeatureSpec("observation.images.wrist", WRIST_TOPIC,
                        "sensor_msgs/msg/Image", 0.1, shape=(4, 4, 3)),
            FeatureSpec("observation.images.top", TOP_TOPIC,
                        "sensor_msgs/msg/Image", 0.1, shape=(4, 4, 3)),
        ),
    )
    cfg.validate()
    return cfg


def two_feature_cfg() -> Config:
    """State + action only. Smaller trace, same machinery."""
    cfg = Config(
        robot_type="so101",
        fps=30,
        task="t",
        repo_id="local/test",
        joint_names=JOINTS,
        features=(
            FeatureSpec("observation.state", STATE_TOPIC,
                        "sensor_msgs/msg/JointState", 0.05, names=JOINTS),
            FeatureSpec("action", ACTION_TOPIC,
                        "std_msgs/msg/Float64MultiArray", 0.05, names=JOINTS),
        ),
    )
    cfg.validate()
    return cfg


# --- tick arithmetic ------------------------------------------------------


def test_tick_zero_is_t0():
    assert tick_ns(1234, 0, 30) == 1234


def test_tick_60_at_30fps_is_exactly_two_seconds():
    assert tick_ns(0, 60, 30) == 2 * NS_PER_S


def test_tick_does_not_accumulate_rounding_error():
    """Computing from k must not drift the way `t += period` does.

    1/30 s is 33_333_333.33... ns, so an integer period loses a third of a
    nanosecond every step. Over an episode that becomes microseconds.
    """
    fps = 30
    period = NS_PER_S // fps  # 33_333_333
    naive = period * 10_000
    assert tick_ns(0, 10_000, fps) - naive > 3_000


def test_tick_spacing_stays_within_one_ns_of_the_ideal_period():
    """Consecutive ticks differ by 33_333_333 or 33_333_334 ns, never less."""
    gaps = {tick_ns(0, k + 1, 30) - tick_ns(0, k, 30) for k in range(1_000)}
    assert gaps <= {33_333_333, 33_333_334}
    assert tick_ns(0, 900, 30) == 30 * NS_PER_S  # 30s lands exactly


# --- LastBuffer as-of semantics -------------------------------------------


def test_empty_buffer_misses():
    buf = LastBuffer(50_000_000)
    assert buf.filled is False
    assert buf.asof(0) is None
    assert buf.miss_empty == 1


def test_buffer_returns_sample_at_or_before_reference():
    buf = LastBuffer(50_000_000)
    buf.push(10_000_000, "v10")
    assert buf.asof(10_000_000) == "v10"
    assert buf.asof(40_000_000) == "v10"


def test_buffer_refuses_future_sample():
    """Causality: a frame must never contain data the robot had not yet seen."""
    buf = LastBuffer(50_000_000)
    buf.push(10_000_000, "v10")
    assert buf.asof(5_000_000) is None
    assert buf.miss_stale == 1


def test_buffer_refuses_stale_sample():
    buf = LastBuffer(50_000_000)
    buf.push(0, "v0")
    assert buf.asof(50_000_000) == "v0"       # exactly at the window edge
    assert buf.asof(50_000_001) is None       # one ns past it
    assert buf.miss_stale == 1


def test_buffer_keeps_newest_and_ignores_older_push():
    buf = LastBuffer(NS_PER_S)
    buf.push(20_000_000, "new")
    buf.push(10_000_000, "old")
    assert buf.asof(30_000_000) == "new"


def test_buffer_summary_reports_match_rate_and_dt():
    buf = LastBuffer(50_000_000)
    buf.push(0, "v")
    buf.asof(10_000_000)
    buf.asof(1_000_000_000)  # stale
    s = buf.summary()
    assert s["tried"] == 2
    assert s["matched"] == 1
    assert s["match_rate"] == pytest.approx(0.5)
    assert s["dt_max_s"] == pytest.approx(0.01)
    assert s["mean_dt_s"] == pytest.approx(0.01)


# --- the emit/push ordering -----------------------------------------------


def test_tick_uses_last_sample_before_it_not_the_one_that_triggered_emission():
    """Regression guard for the ordering trap.

    Ticks must be emitted BEFORE the triggering message is pushed. If a message
    at 40ms were pushed first, the 33.33ms tick would see it as a future sample,
    miss, and drop the frame -- even though the correct 30ms sample was
    available. The single-slot buffer is only sound with emit-before-push.
    """
    cfg = two_feature_cfg()
    stats = GridStats()
    messages = [
        (STATE_TOPIC, 0, "s0"),
        (ACTION_TOPIC, 0, "a0"),
        (STATE_TOPIC, 30_000_000, "s30"),
        (ACTION_TOPIC, 30_000_000, "a30"),
        (STATE_TOPIC, 40_000_000, "s40"),
        (ACTION_TOPIC, 40_000_000, "a40"),
        (STATE_TOPIC, 70_000_000, "s70"),
        (ACTION_TOPIC, 70_000_000, "a70"),
    ]
    frames = list(iter_frames(messages, cfg, stats))
    stats.finalize()

    # tick 0 = 0ns, tick 1 = 33_333_333ns, tick 2 = 66_666_667ns
    assert len(frames) == 3
    assert frames[0] == {"observation.state": "s0", "action": "a0"}
    assert frames[1] == {"observation.state": "s30", "action": "a30"}
    assert frames[2] == {"observation.state": "s40", "action": "a40"}
    assert stats.dropped_interior == 0


def test_duplicate_timestamps_are_all_visible_to_the_tick():
    """A tick at exactly ts must see every message stamped ts, so emission
    waits until a strictly later message arrives."""
    cfg = two_feature_cfg()
    stats = GridStats()
    messages = [
        (STATE_TOPIC, 0, "s0"),
        (ACTION_TOPIC, 0, "a0"),          # same ts as the state sample
        (STATE_TOPIC, 100_000_000, "s100"),
        (ACTION_TOPIC, 100_000_000, "a100"),
    ]
    frames = list(iter_frames(messages, cfg, stats))
    stats.finalize()
    assert frames[0] == {"observation.state": "s0", "action": "a0"}


def test_non_ascending_timestamps_raise():
    cfg = two_feature_cfg()
    stats = GridStats()
    messages = [
        (STATE_TOPIC, 0, "s0"),
        (ACTION_TOPIC, 0, "a0"),
        (STATE_TOPIC, 50_000_000, "s50"),
        (ACTION_TOPIC, 40_000_000, "a40"),  # goes backwards
    ]
    with pytest.raises(ValueError, match="ascending"):
        list(iter_frames(messages, cfg, stats))


# --- frame count on the pinned fixture timing -----------------------------


def pinned_messages(duration_s: float = 2.0):
    """Every topic starts at ts=0 and ends at ts=duration, so t0 == 0."""
    end_ns = int(duration_s * NS_PER_S)
    out: list[tuple[str, int, str]] = []
    for k in range(61):  # cameras at 30Hz, 0..2s inclusive
        ts = tick_ns(0, k, 30)
        if ts > end_ns:
            break
        out.append((WRIST_TOPIC, ts, f"w{k}"))
        out.append((TOP_TOPIC, ts, f"t{k}"))
    for k in range(201):  # joint_states at 100Hz
        out.append((STATE_TOPIC, k * 10_000_000, f"s{k}"))
    for k in range(101):  # commands at 50Hz
        out.append((ACTION_TOPIC, k * 20_000_000, f"a{k}"))
    out.sort(key=lambda m: m[1])
    return out


def test_pinned_two_second_fixture_yields_exactly_61_frames():
    cfg = make_cfg()
    stats = GridStats()
    frames = list(iter_frames(pinned_messages(), cfg, stats))
    stats.finalize()
    assert len(frames) == 61
    assert stats.emitted == 61
    assert stats.dropped_leading == 0
    assert stats.dropped_interior == 0
    assert stats.first_emitted_k == 0
    assert stats.last_emitted_k == 60
    assert set(frames[0]) == {
        "observation.state",
        "action",
        "observation.images.wrist",
        "observation.images.top",
    }


# --- drop classification --------------------------------------------------


def test_interior_gap_is_counted_and_attributed():
    """A 300ms camera gap exceeds the 100ms image window, so those ticks drop
    and are classified interior because emission resumes afterwards."""
    cfg = make_cfg()
    stats = GridStats()
    messages = [m for m in pinned_messages() if not (
        m[0] == WRIST_TOPIC and 700_000_000 < m[1] < 1_000_000_000
    )]
    frames = list(iter_frames(messages, cfg, stats))
    stats.finalize()

    assert stats.dropped_interior > 0
    assert stats.last_emitted_k == 60          # emission resumed after the gap
    assert len(frames) == stats.emitted
    assert all(key == "observation.images.wrist" for _, key in stats.interior_drops)
    assert stats.interior_drop_frac > 0.02


def test_trailing_drops_are_not_counted_interior():
    """Cutting a camera short at the end trims the episode; it does not distort
    the timeline of what was kept."""
    cfg = make_cfg()
    stats = GridStats()
    messages = [m for m in pinned_messages() if not (
        m[0] == WRIST_TOPIC and m[1] > 1_500_000_000
    )]
    list(iter_frames(messages, cfg, stats))
    stats.finalize()

    assert stats.dropped_interior == 0
    assert stats.dropped_trailing > 0
    assert stats.interior_drop_frac == 0.0


def test_leading_drops_are_not_counted_interior():
    """t0 is the instant every buffer first holds something, but the other
    buffers may already be stale there, so the earliest ticks can still drop.
    Those are leading drops: nothing had been emitted yet.

    Here action arrives 200ms after the only state sample, so tick 0 (at t0 =
    200ms) finds state 200ms old, past its 50ms window. Emission starts at
    tick 1 once fresh samples of both have arrived.
    """
    cfg = two_feature_cfg()
    stats = GridStats()
    messages = [
        (STATE_TOPIC, 0, "s0"),
        (ACTION_TOPIC, 200_000_000, "a200"),   # t0 = 200ms
        (STATE_TOPIC, 210_000_000, "s210"),
        (ACTION_TOPIC, 220_000_000, "a220"),
        (STATE_TOPIC, 240_000_000, "s240"),
        (ACTION_TOPIC, 260_000_000, "a260"),
        (STATE_TOPIC, 280_000_000, "s280"),
        (ACTION_TOPIC, 300_000_000, "a300"),
    ]
    frames = list(iter_frames(messages, cfg, stats))
    stats.finalize()

    assert stats.dropped_leading == 1
    assert stats.dropped_interior == 0
    assert stats.first_emitted_k == 1
    assert len(frames) == 3


def test_no_frames_when_a_feature_never_arrives():
    cfg = two_feature_cfg()
    stats = GridStats()
    messages = [(STATE_TOPIC, k * 10_000_000, f"s{k}") for k in range(50)]
    frames = list(iter_frames(messages, cfg, stats))
    stats.finalize()
    assert frames == []
    assert stats.emitted == 0
    assert stats.span_ticks == 0
    assert stats.interior_drop_frac == 0.0


def test_unconfigured_topics_are_ignored():
    cfg = two_feature_cfg()
    stats = GridStats()
    messages = [
        ("/tf", 0, "ignored"),
        (STATE_TOPIC, 0, "s0"),
        (ACTION_TOPIC, 0, "a0"),
        ("/rosout", 5_000_000, "ignored"),
        (STATE_TOPIC, 100_000_000, "s100"),
        (ACTION_TOPIC, 100_000_000, "a100"),
    ]
    frames = list(iter_frames(messages, cfg, stats))
    stats.finalize()
    assert frames[0] == {"observation.state": "s0", "action": "a0"}
