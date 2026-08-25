"""Fixed-rate grid resampling with as-of sampling.

Imports neither ROS nor lerobot: it consumes an iterable of
`(topic, ts_ns, decoded_value)` tuples, so the tick arithmetic and the drop
accounting are unit-testable on plain Python data.

Why a fixed grid rather than one frame per message on a reference topic:
lerobot 0.5.1 synthesises `timestamp = frame_index / fps` and discards the real
bag times. A jittery reference stream would therefore be recorded as uniformly
spaced, quietly distorting the joint velocities a policy learns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

from converter.config import Config, FeatureSpec

NS_PER_S = 1_000_000_000


def tick_ns(t0_ns: int, k: int, fps: int) -> int:
    """Absolute time of grid tick `k`.

    Computed from `k` every time rather than accumulated, so rounding cannot
    build up over an episode (1/fps is not an integer number of nanoseconds).
    """
    return t0_ns + round(k * NS_PER_S / fps)


class LastBuffer:
    """Newest sample for one feature, answering as-of queries."""

    def __init__(self, max_age_ns: int) -> None:
        self.max_age_ns = int(max_age_ns)
        self.ts_ns: int | None = None
        self.value = None

        self.tried = 0
        self.matched = 0
        self.miss_empty = 0
        self.miss_stale = 0
        self.dt_max_ns = 0
        self._dt_sum_ns = 0

    @property
    def filled(self) -> bool:
        return self.ts_ns is not None

    def push(self, ts_ns: int, value) -> None:
        if self.ts_ns is None or ts_ns >= self.ts_ns:
            self.ts_ns = int(ts_ns)
            self.value = value

    def asof(self, ref_ns: int):
        """Newest sample at or before `ref_ns`, if within `max_age_ns`.

        `ts_ns <= ref_ns` is causality: a frame must never carry a sample the
        robot had not yet observed at that instant. With emit-before-push in
        `iter_frames` a future sample cannot normally be present, but the check
        stays so the invariant is enforced rather than assumed.
        """
        self.tried += 1
        if self.ts_ns is None:
            self.miss_empty += 1
            return None

        dt = ref_ns - self.ts_ns
        if dt < 0 or dt > self.max_age_ns:
            self.miss_stale += 1
            return None

        self.matched += 1
        self._dt_sum_ns += dt
        self.dt_max_ns = max(self.dt_max_ns, dt)
        return self.value

    def summary(self) -> dict:
        return {
            "max_age_s": self.max_age_ns / NS_PER_S,
            "tried": self.tried,
            "matched": self.matched,
            "match_rate": self.matched / self.tried if self.tried else 0.0,
            "miss_empty": self.miss_empty,
            "miss_stale": self.miss_stale,
            "mean_dt_s": (self._dt_sum_ns / self.matched / NS_PER_S)
            if self.matched
            else float("nan"),
            "dt_max_s": self.dt_max_ns / NS_PER_S,
        }


@dataclass
class GridStats:
    """Per-episode emission and drop accounting.

    A drop after the first emitted tick cannot be classified until the episode
    ends -- it is interior only if some later tick was emitted. `finalize()`
    resolves the pending drops once the last emitted tick is known.
    """

    emitted: int = 0
    dropped_leading: int = 0
    dropped_interior: int = 0
    dropped_trailing: int = 0
    first_emitted_k: int | None = None
    last_emitted_k: int | None = None
    interior_drops: list[tuple[int, str]] = field(default_factory=list)
    buffers: dict[str, LastBuffer] = field(default_factory=dict)
    _pending_drops: list[tuple[int, str]] = field(default_factory=list)

    def record_emit(self, k: int) -> None:
        self.emitted += 1
        if self.first_emitted_k is None:
            self.first_emitted_k = k
        self.last_emitted_k = k

    def record_drop(self, k: int, missing_key: str) -> None:
        if self.first_emitted_k is None:
            self.dropped_leading += 1
        else:
            self._pending_drops.append((k, missing_key))

    def finalize(self) -> None:
        if self.last_emitted_k is None:
            self.interior_drops = []
            self.dropped_interior = 0
            self.dropped_trailing = len(self._pending_drops)
            return
        last = self.last_emitted_k
        self.interior_drops = [(k, key) for k, key in self._pending_drops if k < last]
        self.dropped_interior = len(self.interior_drops)
        self.dropped_trailing = len(self._pending_drops) - self.dropped_interior

    @property
    def span_ticks(self) -> int:
        """Ticks from the first to the last emitted tick, inclusive."""
        if self.first_emitted_k is None or self.last_emitted_k is None:
            return 0
        return self.last_emitted_k - self.first_emitted_k + 1

    @property
    def interior_drop_frac(self) -> float:
        span = self.span_ticks
        return self.dropped_interior / span if span else 0.0


def _sample(
    buffers: dict[str, LastBuffer],
    by_topic: dict[str, FeatureSpec],
    tick_time_ns: int,
) -> tuple[dict | None, str | None]:
    """Sample every feature as of `tick_time_ns`.

    Returns `(frame, None)` when all features matched, else `(None, missing_key)`.
    """
    frame: dict = {}
    for topic, spec in by_topic.items():
        value = buffers[topic].asof(tick_time_ns)
        if value is None:
            return None, spec.key
        frame[spec.key] = value
    return frame, None


def iter_frames(
    messages: Iterable[tuple[str, int, object]],
    cfg: Config,
    stats: GridStats,
) -> Iterator[dict]:
    """Resample `messages` onto a fixed `cfg.fps` grid.

    `messages` must be ordered by ascending `ts_ns`. Yields one dict per tick
    where every configured feature had a fresh sample.
    """
    by_topic = cfg.by_topic()
    buffers = {topic: LastBuffer(spec.max_age_ns) for topic, spec in by_topic.items()}
    stats.buffers = buffers

    t0: int | None = None
    k = 0
    last_ts: int | None = None

    def flush_until(limit_ns: int, inclusive: bool) -> Iterator[dict]:
        nonlocal k
        while t0 is not None:
            tick_time = tick_ns(t0, k, cfg.fps)
            if tick_time > limit_ns or (tick_time == limit_ns and not inclusive):
                return
            frame, missing = _sample(buffers, by_topic, tick_time)
            if frame is None:
                stats.record_drop(k, missing)
            else:
                stats.record_emit(k)
                yield frame
            k += 1

    for topic, ts_ns, value in messages:
        if topic not in buffers:
            continue
        if last_ts is not None and ts_ns < last_ts:
            raise ValueError(
                f"bag timestamps are not ascending: {ts_ns} follows {last_ts} on "
                f"topic {topic!r}. Grid alignment requires ascending order."
            )
        last_ts = ts_ns

        # Emit before pushing. Every tick strictly before this message is now
        # final, and nothing newer than the tick has entered the buffers yet --
        # which is what keeps the single-slot buffer free of future samples.
        yield from flush_until(ts_ns, inclusive=False)

        buffers[topic].push(ts_ns, value)
        if t0 is None and all(b.filled for b in buffers.values()):
            t0 = ts_ns

    if last_ts is not None:
        yield from flush_until(last_ts, inclusive=True)
