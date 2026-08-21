# memory_scheduler.py

"""
FINAL ENTERPRISE-GRADE
MEMORY SCHEDULER

Runs periodic background memory maintenance.

Responsibilities
────────────────
- apply Ebbinghaus decay
- remove forgotten memories
- graceful shutdown
- resilient execution
- lifecycle callbacks
- failure escalation
- observability
- distributed diagnostics
"""

import logging
import threading
import time
import uuid

from dataclasses import dataclass

from typing import (
    Callable,
    Dict,
    List,
    Optional
)

from memory.memory_store import (
    MemoryStore
)


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

FAILURE_ALERT_THRESHOLD = 5


# ─────────────────────────────────────────────────────────────
# Cleanup Result
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CleanupResult:
    """
    Structured cleanup result.

    NOTE:
    Legacy cleanup() implementations
    may return only an integer count.

    In compatibility mode,
    removed_ids cannot be recovered
    and will remain empty.
    """

    removed_count: int

    removed_ids: List[str]


# ─────────────────────────────────────────────────────────────
# Maintenance Event
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MaintenanceEvent:
    """
    Immutable maintenance event.
    """

    cycle_id: str

    removed_count: int

    removed_ids: List[str]

    duration: float

    average_duration: float

    timestamp: float

    total_cycles: int

    consecutive_failures: int

    scheduler_metrics: Dict

    memory_store_metrics: Dict

    memory_snapshot: Dict


# ─────────────────────────────────────────────────────────────
# Memory Scheduler
# ─────────────────────────────────────────────────────────────

class MemoryScheduler:
    """
    Background scheduler for
    periodic memory maintenance.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        interval_seconds: int = 300,

        on_cycle_complete: Optional[
            Callable[
                [MaintenanceEvent],
                None
            ]
        ] = None,

        on_failure_alert: Optional[
            Callable[
                [Dict],
                None
            ]
        ] = None
    ):

        if interval_seconds <= 0:

            raise ValueError(
                "interval_seconds "
                "must be > 0"
            )

        self.memory_store = (
            memory_store
        )

        self.interval_seconds = (
            interval_seconds
        )

        self.on_cycle_complete = (
            on_cycle_complete
        )

        self.on_failure_alert = (
            on_failure_alert
        )

        self.running = False

        self.thread = None

        # IMPORTANT:
        # RLock is intentionally used
        # instead of Lock because some
        # internal helper methods may
        # be called from already-locked
        # contexts.
        #
        # Replacing this with a plain
        # Lock can introduce deadlocks.

        self.lock = threading.RLock()

        self.stop_event = (
            threading.Event()
        )

        # observability metrics

        self.total_cycles = 0

        self.total_failures = 0

        self.total_removed = 0

        self.last_cycle_time = None

        self.total_cycle_duration = 0.0

        self.max_cycle_duration = 0.0

        self.consecutive_failures = 0

        logger.info(
            "MemoryScheduler initialized "
            "interval=%ss",
            interval_seconds
        )

    # ─────────────────────────────────────────────────────────
    # Average Duration
    # ─────────────────────────────────────────────────────────

    def _average_duration_unsafe(
        self
    ) -> float:
        """
        Lock-free average duration.

        MUST only be called from
        contexts that already hold
        self.lock.
        """

        if self.total_cycles == 0:
            return 0.0

        return (
            self.total_cycle_duration
            / self.total_cycles
        )

    def _average_duration(
        self
    ) -> float:

        with self.lock:

            return (
                self._average_duration_unsafe()
            )

    # ─────────────────────────────────────────────────────────
    # Safe Memory Metrics
    # ─────────────────────────────────────────────────────────

    def _safe_memory_metrics(
        self
    ) -> Dict:

        try:

            return (
                self.memory_store
                .metrics()
            )

        except Exception:

            logger.exception(
                "failed retrieving "
                "memory metrics"
            )

            return {}

    # ─────────────────────────────────────────────────────────
    # Safe Memory Snapshot
    # ─────────────────────────────────────────────────────────

    def _safe_memory_snapshot(
        self
    ) -> Dict:

        snapshot = {}

        try:

            if hasattr(
                self.memory_store,
                "working_memory"
            ):

                snapshot[
                    "working_memory_size"
                ] = len(
                    self.memory_store
                    .working_memory
                )

        except Exception:

            logger.exception(
                "failed collecting "
                "memory snapshot"
            )

        return snapshot

    # ─────────────────────────────────────────────────────────
    # System Health
    # ─────────────────────────────────────────────────────────

    def _system_health(
        self
    ) -> Dict:

        return {
            "thread_alive": (
                self.thread is not None
                and self.thread.is_alive()
            ),

            "scheduler_running": (
                self.running
            ),

            "active_threads": (
                threading.active_count()
            )
        }

    # ─────────────────────────────────────────────────────────
    # Maintenance Cycle
    # ─────────────────────────────────────────────────────────

    def _maintenance_cycle(
        self
    ):

        cycle_id = str(
            uuid.uuid4()
        )

        start_time = time.time()

        try:

            logger.debug(
                "maintenance cycle started "
                "cycle_id=%s",
                cycle_id
            )

            self.memory_store.apply_decay()

            cleanup_result = (
                self.memory_store
                .cleanup()
            )

            # backward compatibility

            if isinstance(
                cleanup_result,
                int
            ):

                cleanup_result = CleanupResult(
                    removed_count=(
                        cleanup_result
                    ),

                    removed_ids=[]
                )

            end_time = time.time()

            duration = (
                end_time
                - start_time
            )

            with self.lock:

                self.total_cycles += 1

                self.total_removed += (
                    cleanup_result
                    .removed_count
                )

                self.last_cycle_time = (
                    end_time
                )

                self.total_cycle_duration += (
                    duration
                )

                self.max_cycle_duration = max(
                    self.max_cycle_duration,
                    duration
                )

                self.consecutive_failures = 0

                average_duration = (
                    self._average_duration_unsafe()
                )

            scheduler_metrics = (
                self.metrics()
            )

            memory_store_metrics = (
                self._safe_memory_metrics()
            )

            memory_snapshot = (
                self._safe_memory_snapshot()
            )

            event = MaintenanceEvent(
                cycle_id=cycle_id,

                removed_count=(
                    cleanup_result
                    .removed_count
                ),

                removed_ids=(
                    cleanup_result
                    .removed_ids
                ),

                duration=duration,

                average_duration=(
                    average_duration
                ),

                timestamp=end_time,

                total_cycles=(
                    self.total_cycles
                ),

                consecutive_failures=(
                    self.consecutive_failures
                ),

                scheduler_metrics=(
                    scheduler_metrics
                ),

                memory_store_metrics=(
                    memory_store_metrics
                ),

                memory_snapshot=(
                    memory_snapshot
                )
            )

            logger.info(
                "maintenance cycle complete "
                "cycle_id=%s "
                "removed=%s "
                "duration=%.4fs "
                "avg_duration=%.4fs",
                cycle_id,
                cleanup_result.removed_count,
                duration,
                average_duration
            )

            if (
                self.on_cycle_complete
                is not None
            ):

                try:

                    self.on_cycle_complete(
                        event
                    )

                except Exception:

                    logger.exception(
                        "maintenance callback "
                        "failed cycle_id=%s",
                        cycle_id
                    )

        except Exception:

            with self.lock:

                self.total_failures += 1

                self.consecutive_failures += 1

            logger.exception(
                "maintenance cycle failed "
                "cycle_id=%s "
                "consecutive_failures=%s",
                cycle_id,
                self.consecutive_failures
            )

            if (
                self.consecutive_failures
                >= FAILURE_ALERT_THRESHOLD
            ):

                logger.error(
                    "failure alert threshold "
                    "reached failures=%s",
                    self.consecutive_failures
                )

                if (
                    self.on_failure_alert
                    is not None
                ):

                    try:

                        self.on_failure_alert(
                            {
                                "cycle_id": (
                                    cycle_id
                                ),

                                "consecutive_failures": (
                                    self
                                    .consecutive_failures
                                ),

                                "total_failures": (
                                    self.total_failures
                                ),

                                "scheduler_metrics": (
                                    self.metrics()
                                ),

                                "memory_store_metrics": (
                                    self
                                    ._safe_memory_metrics()
                                ),

                                "memory_snapshot": (
                                    self
                                    ._safe_memory_snapshot()
                                ),

                                "system_health": (
                                    self
                                    ._system_health()
                                ),

                                "timestamp": (
                                    time.time()
                                )
                            }
                        )

                    except Exception:

                        logger.exception(
                            "failure alert callback "
                            "failed cycle_id=%s",
                            cycle_id
                        )

    # ─────────────────────────────────────────────────────────
    # Manual Trigger
    # ─────────────────────────────────────────────────────────

    def run_once(
        self
    ):

        logger.info(
            "manual maintenance trigger"
        )

        self._maintenance_cycle()

    # ─────────────────────────────────────────────────────────
    # Failure Backoff
    # ─────────────────────────────────────────────────────────

    def _compute_backoff(
        self
    ) -> int:

        with self.lock:

            if (
                self.consecutive_failures
                == 0
            ):

                return self.interval_seconds

            multiplier = min(
                2 ** self.consecutive_failures,
                32
            )

            return (
                self.interval_seconds
                * multiplier
            )

    # ─────────────────────────────────────────────────────────
    # Run Loop
    # ─────────────────────────────────────────────────────────

    def _run(
        self
    ):

        logger.info(
            "memory scheduler loop started"
        )

        # immediate startup cycle

        self._maintenance_cycle()

        while not self.stop_event.is_set():

            wait_time = (
                self._compute_backoff()
            )

            if self.stop_event.wait(
                wait_time
            ):
                break

            if not self.running:
                break

            self._maintenance_cycle()

        logger.info(
            "memory scheduler loop exited"
        )

    # ─────────────────────────────────────────────────────────
    # Start
    # ─────────────────────────────────────────────────────────

    def start(
        self
    ):

        with self.lock:

            if self.running:

                logger.warning(
                    "memory scheduler "
                    "already running"
                )

                return

            self.running = True

            self.stop_event.clear()

            self.thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="MemoryScheduler"
            )

            self.thread.start()

        logger.info(
            "memory scheduler started"
        )

    # ─────────────────────────────────────────────────────────
    # Stop
    # ─────────────────────────────────────────────────────────

    def stop(
        self
    ):

        with self.lock:

            if not self.running:
                return

            self.running = False

            self.stop_event.set()

            thread = self.thread

        # NOTE:
        # Lock intentionally released
        # before thread.join().
        #
        # Holding the scheduler lock
        # during join could deadlock
        # if the worker thread tries
        # to acquire the same lock.

        if thread is not None:

            thread.join(timeout=5)

            if thread.is_alive():

                logger.warning(
                    "memory scheduler "
                    "shutdown timeout"
                )

        logger.info(
            "memory scheduler stopped"
        )

    # ─────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────

    def is_running(
        self
    ) -> bool:

        with self.lock:

            return self.running

    # ─────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────

    def metrics(
        self
    ) -> Dict[str, object]:

        with self.lock:

            return {
                "running": (
                    self.running
                ),

                "interval_seconds": (
                    self.interval_seconds
                ),

                "total_cycles": (
                    self.total_cycles
                ),

                "total_failures": (
                    self.total_failures
                ),

                "consecutive_failures": (
                    self.consecutive_failures
                ),

                "total_removed": (
                    self.total_removed
                ),

                "last_cycle_time": (
                    self.last_cycle_time
                ),

                "average_cycle_duration": (
                    self
                    ._average_duration_unsafe()
                ),

                "max_cycle_duration": (
                    self.max_cycle_duration
                )
            }