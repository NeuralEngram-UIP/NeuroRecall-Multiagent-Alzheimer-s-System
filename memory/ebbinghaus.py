# memory/ebbinghaus.py

"""
Production-Grade Adaptive Ebbinghaus Forgetting Engine

Features
────────
- immutable retention state
- mathematically validated parameters
- adaptive reinforcement
- diminishing returns
- bounded stability
- deterministic clocks
- functional compatibility API
- production-safe validation
- maintenance-safe performance
"""

import math
import time

from dataclasses import dataclass
from typing import Callable


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

DEFAULT_RETENTION_THRESHOLD = 0.30

BASE_STABILITY_HOURS = 24.0

# backward compatibility alias
DEFAULT_STABILITY_HOURS = (
    BASE_STABILITY_HOURS
)

MAX_STABILITY_HOURS = 8760.0

DEFAULT_REINFORCEMENT_BOOST = 1.8

DEFAULT_DIMINISHING_FACTOR = 0.3

MIN_IMPORTANCE = 0.1

MAX_IMPORTANCE = 5.0

MIN_PRIORITY = 1

MAX_PRIORITY = 10


# backward compatibility alias
BASE_STABLE_HOURS = (
    BASE_STABILITY_HOURS
)


# ─────────────────────────────────────────────────────────────
# Retention State
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RetentionState:
    """
    Immutable retention state.
    """

    stability_hours: float

    retention: float

    last_accessed: float

    access_count: int

    importance: float

    priority: int

    @staticmethod
    def create(
        stability_hours: float = (
            BASE_STABILITY_HOURS
        ),
        importance: float = 1.0,
        priority: int = 1,
        clock: Callable[[], float] = (
            time.time
        )
    ) -> "RetentionState":

        if stability_hours <= 0:

            raise ValueError(
                "stability_hours must be positive"
            )

        if not (
            MIN_IMPORTANCE
            <= importance
            <= MAX_IMPORTANCE
        ):

            raise ValueError(
                "importance must be between "
                f"{MIN_IMPORTANCE} and "
                f"{MAX_IMPORTANCE}"
            )

        if not (
            MIN_PRIORITY
            <= priority
            <= MAX_PRIORITY
        ):

            raise ValueError(
                "priority must be between "
                f"{MIN_PRIORITY} and "
                f"{MAX_PRIORITY}"
            )

        stability_hours = min(
            stability_hours,
            MAX_STABILITY_HOURS
        )

        return RetentionState(
            stability_hours=(
                stability_hours
            ),

            retention=1.0,

            last_accessed=clock(),

            access_count=0,

            importance=importance,

            priority=priority
        )


# ─────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────

class EbbinghausEngine:
    """
    Adaptive forgetting engine.

    Formula:
        R = e^(-t/S)
    """

    def __init__(
        self,
        retention_threshold: float = (
            DEFAULT_RETENTION_THRESHOLD
        ),

        reinforcement_boost: float = (
            DEFAULT_REINFORCEMENT_BOOST
        ),

        diminishing_factor: float = (
            DEFAULT_DIMINISHING_FACTOR
        ),

        clock: Callable[[], float] = (
            time.time
        )
    ):

        if not (
            0.0
            < retention_threshold
            < 1.0
        ):

            raise ValueError(
                "retention_threshold "
                "must be between 0 and 1"
            )

        if reinforcement_boost <= 1.0:

            raise ValueError(
                "reinforcement_boost "
                "must be > 1.0"
            )

        if diminishing_factor < 0:

            raise ValueError(
                "diminishing_factor "
                "must be >= 0"
            )

        self.retention_threshold = (
            retention_threshold
        )

        self.reinforcement_boost = (
            reinforcement_boost
        )

        self.diminishing_factor = (
            diminishing_factor
        )

        self.clock = clock

    # ─────────────────────────────────────────────────────────

    def create_state(
        self,
        stability_hours: float = (
            BASE_STABILITY_HOURS
        ),

        importance: float = 1.0,

        priority: int = 1
    ) -> RetentionState:
        """
        Create initial memory state.

        IMPORTANT
        ─────────
        Stability weighting is
        intentional and part of the
        adaptive cognition model.

        weighted_stability =
            stability
            × importance
            × priority
        """

        weighted_stability = (
            stability_hours
            * importance
            * priority
        )

        weighted_stability = min(
            weighted_stability,
            MAX_STABILITY_HOURS
        )

        return RetentionState.create(
            stability_hours=(
                weighted_stability
            ),

            importance=importance,

            priority=priority,

            clock=self.clock
        )

    # ─────────────────────────────────────────────────────────

    def compute_retention(
        self,
        elapsed_hours: float,
        stability_hours: float
    ) -> float:
        """
        Compute retention score.
        """

        if stability_hours <= 0:

            raise ValueError(
                "stability_hours must be positive"
            )

        elapsed_hours = max(
            0.0,
            elapsed_hours
        )

        stability_hours = min(
            stability_hours,
            MAX_STABILITY_HOURS
        )

        retention = math.exp(
            -elapsed_hours
            / stability_hours
        )

        return min(
            max(retention, 0.0),
            1.0
        )

    # ─────────────────────────────────────────────────────────

    def decay(
        self,
        state: RetentionState
    ) -> RetentionState:
        """
        Apply forgetting decay.
        """

        now = self.clock()

        elapsed_hours = max(
            0.0,
            (
                now
                - state.last_accessed
            ) / 3600.0
        )

        retention = (
            self.compute_retention(
                elapsed_hours=(
                    elapsed_hours
                ),

                stability_hours=(
                    state.stability_hours
                )
            )
        )

        return RetentionState(
            stability_hours=(
                state.stability_hours
            ),

            retention=retention,

            last_accessed=(
                state.last_accessed
            ),

            access_count=(
                state.access_count
            ),

            importance=(
                state.importance
            ),

            priority=(
                state.priority
            )
        )

    # ─────────────────────────────────────────────────────────

    def reinforce(
        self,
        state: RetentionState,
        quality: float = 1.0
    ) -> RetentionState:
        """
        Reinforce memory after recall.
        """

        if not (
            0.0
            <= quality
            <= 1.0
        ):

            raise ValueError(
                "quality must be between 0 and 1"
            )

        decay_factor = (
            1.0
            / (
                1.0
                + (
                    self.diminishing_factor
                    * math.log1p(
                        state.access_count
                    )
                )
            )
        )

        effective_boost = (
            1.0
            + quality
            * (
                self.reinforcement_boost
                - 1.0
            )
            * decay_factor
        )

        new_stability = (
            state.stability_hours
            * effective_boost
        )

        new_stability = min(
            new_stability,
            MAX_STABILITY_HOURS
        )

        return RetentionState(
            stability_hours=(
                new_stability
            ),

            retention=1.0,

            last_accessed=self.clock(),

            access_count=(
                state.access_count
                + 1
            ),

            importance=(
                state.importance
            ),

            priority=(
                state.priority
            )
        )

    # ─────────────────────────────────────────────────────────

    def forgotten(
        self,
        state: RetentionState
    ) -> bool:
        """
        Determine whether memory
        is forgotten.
        """

        decayed = self.decay(
            state
        )

        return (
            decayed.retention
            < self.retention_threshold
        )

    # ─────────────────────────────────────────────────────────

    def time_until_forgotten(
        self,
        state: RetentionState
    ) -> float:
        """
        Estimate remaining hours
        before forgetting.
        """

        threshold_hours = (
            -state.stability_hours
            * math.log(
                self.retention_threshold
            )
        )

        elapsed_hours = (
            (
                self.clock()
                - state.last_accessed
            ) / 3600.0
        )

        remaining = (
            threshold_hours
            - elapsed_hours
        )

        return max(
            remaining,
            0.0
        )


# ─────────────────────────────────────────────────────────────
# Functional Compatibility Layer
# ─────────────────────────────────────────────────────────────

_default_engine = (
    EbbinghausEngine()
)


def compute_retention(
    last_reviewed_at,
    stability_hours,
    now=None
):
    """
    Compute retention from timestamps.
    """

    if now is None:

        now = time.time()

    if hasattr(
        last_reviewed_at,
        "timestamp"
    ):

        last_reviewed_at = (
            last_reviewed_at.timestamp()
        )

    if hasattr(
        now,
        "timestamp"
    ):

        now = now.timestamp()

    elapsed_hours = (
        now - last_reviewed_at
    ) / 3600.0

    return (
        _default_engine
        .compute_retention(
            elapsed_hours=(
                elapsed_hours
            ),

            stability_hours=(
                stability_hours
            )
        )
    )


def is_memory_forgotten(
    retention: float
) -> bool:
    """
    Uses default threshold.
    """

    return (
        retention
        < _default_engine
        .retention_threshold
    )


def reinforce_memory(
    stability_hours: float,
    quality: float = 1.0,
    review_count: int = 1
) -> float:
    """
    Reinforce memory stability.
    """

    state = RetentionState(
        stability_hours=(
            stability_hours
        ),

        retention=1.0,

        last_accessed=time.time(),

        access_count=review_count,

        importance=1.0,

        priority=1
    )

    reinforced = (
        _default_engine
        .reinforce(
            state,
            quality=quality
        )
    )

    return (
        reinforced.stability_hours
    )


def time_until_forgotten(
    last_reviewed_at,
    stability_hours,
    now=None
):
    """
    Estimate remaining hours
    before forgetting.
    """

    if now is None:

        now = time.time()

    if hasattr(
        last_reviewed_at,
        "timestamp"
    ):

        last_reviewed_at = (
            last_reviewed_at.timestamp()
        )

    if hasattr(
        now,
        "timestamp"
    ):

        now = now.timestamp()

    threshold_hours = (
        -stability_hours
        * math.log(
            _default_engine
            .retention_threshold
        )
    )

    elapsed_hours = (
        now - last_reviewed_at
    ) / 3600.0

    remaining = (
        threshold_hours
        - elapsed_hours
    )

    return max(
        remaining,
        0.0
    )


# ─────────────────────────────────────────────────────────────
# Backward Compatibility
# ─────────────────────────────────────────────────────────────

def create_memory_state(
    stability_hours: float = (
        DEFAULT_STABILITY_HOURS
    ),

    importance: float = 1.0,

    priority: int = 1
):
    """
    Backward compatibility wrapper.

    IMPORTANT
    ─────────
    Uses adaptive weighting from
    create_state() intentionally.
    """

    return (
        _default_engine
        .create_state(
            stability_hours=(
                stability_hours
            ),

            importance=importance,

            priority=priority
        )
    )