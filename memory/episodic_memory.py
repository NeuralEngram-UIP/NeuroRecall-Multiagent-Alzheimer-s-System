# memory/episodic_memory.py

"""
ALZHEIMER'S-AWARE EPISODIC MEMORY STORE

Drop-in replacement for the generic multi-agent episodic store.
Public interface (class names, method names/signatures) is unchanged
so memory_orchestrator.py, scheduler.py, and api/main.py keep working
without modification. What changed is internal:

  - Memories are typed by MemoryCategory (critical / person / routine
    / episodic), each with different, clinically-motivated handling —
    see memory/models.py for the reasoning.
  - Nothing is ever hard-deleted by decay. "Forgotten" means archived
    (excluded from active recall); the row stays in SQLite for the
    caregiver/clinician's full history. Only an explicit `purge()`
    call, with an actor and a reason, permanently removes data.
  - No per-patient special-casing lives in code. If a patient needs a
    different decay rate, that's set via `set_patient_profile()` and
    stored as auditable data, not a hardcoded name in this file.
  - Every meaningful write is recorded in an append-only audit log.
"""

import hashlib
import json
import logging
import math
import sqlite3
import threading
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import anthropic

from sentence_transformers import SentenceTransformer

from memory.ebbinghaus import (
    DEFAULT_STABILITY_HOURS,
    compute_retention,
    reinforce_memory,
)
from memory.models import (
    BASE_STABILITY_HOURS,
    EPISODIC_ARCHIVE_THRESHOLD,
    PERSON_AT_RISK_THRESHOLD,
    MemoryCategory,
    valid_category,
)


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Lazy Dependency Loading (unchanged from original)
# ─────────────────────────────────────────────────────────────

_client = None
_model = None
_client_lock = threading.Lock()
_model_lock = threading.Lock()


def get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = anthropic.Anthropic()
    return _client


def get_embedding_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.20
DB_PATH = "multi_agent_memory.db"

REINFORCEMENT_BOOST = 1.6
DIMINISHING_FACTOR = 0.3
MAX_STABILITY_HOURS = 24 * 365 * 5  # 5-year sanity ceiling


# ─────────────────────────────────────────────────────────────
# Thread-Local SQLite Connections
# ─────────────────────────────────────────────────────────────

_thread_local = threading.local()


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    attr = f"connection_{db_path}".replace("/", "_").replace(".", "_")
    conn = getattr(_thread_local, attr, None)
    if conn is None:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        setattr(_thread_local, attr, conn)
    return conn


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return json.dumps(str(obj))


def safe_json_loads(s: Any) -> Any:
    try:
        return json.loads(s)
    except (TypeError, ValueError, json.JSONDecodeError):
        return s


def simple_embedding(text: str) -> List[float]:
    model = get_embedding_model()
    return model.encode(text).tolist()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _heuristic_importance(text: str) -> float:
    text_lower = text.lower()
    high = ["i am", "i'm", "i love", "i hate", "my goal", "i need"]
    low = ["okay", "thanks", "hello", "hi"]
    if any(s in text_lower for s in high):
        return 0.75
    if any(s in text_lower for s in low):
        return 0.15
    return 0.45


def should_store_memory(text: str, importance: float) -> bool:
    if importance < 0.15:
        return False
    if len(text.split()) <= 2 and importance < 0.20:
        return False
    return True


# ─────────────────────────────────────────────────────────────
# Enums (RecallStatus preserved — referenced by tests)
# ─────────────────────────────────────────────────────────────

class RecallStatus(Enum):
    UPDATED = "updated"
    FORGOTTEN = "forgotten"
    NOT_FOUND = "not_found"


# ─────────────────────────────────────────────────────────────
# Episode
# ─────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """
    A single memory belonging to one patient (agent_id == patient_id
    in this deployment).

    category drives every decay/visibility rule — see models.py.
    """

    content: Any
    context: Dict[str, Any]
    agent_id: str

    tags: List[str] = field(default_factory=list)
    shared: bool = False

    category: MemoryCategory = MemoryCategory.EPISODIC

    stability_hours: float = DEFAULT_STABILITY_HOURS

    last_reviewed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    review_count: int = 0

    # Soft-state flags. Neither ever triggers a hard delete.
    archived: bool = False   # EPISODIC only: excluded from active recall
    at_risk: bool = False    # PERSON only: caregiver attention flag

    cross_agent_recall_count: int = 0

    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    importance: float = 0.6
    agent_feedback: float = 0.5
    task_success_rate: float = 0.5

    # ── Compatibility alias ──────────────────────────────────
    # planner_agent.py currently reads `.episode_id` on fused results;
    # semantic-tier results expose `.memory_id`. Adding this alias lets
    # a future one-line fix in planner_agent.py (getattr(mem,
    # "memory_id", None)) work uniformly across both tiers without any
    # further change here.
    @property
    def memory_id(self) -> str:
        return self.episode_id

    # ── Retention ─────────────────────────────────────────────

    def retention(self, now: Optional[datetime] = None) -> float:
        """
        CRITICAL / ROUTINE: always 1.0 — never decays.
        PERSON / EPISODIC: normal Ebbinghaus retention curve.
        """
        if self.category in (MemoryCategory.CRITICAL, MemoryCategory.ROUTINE):
            return 1.0
        return compute_retention(
            self.last_reviewed_at, self.stability_hours, now
        )

    def is_forgotten(self, now: Optional[datetime] = None) -> bool:
        """
        Only EPISODIC memories are ever considered "forgotten"
        (i.e. excluded from active recall via archiving).

        CRITICAL / ROUTINE: never forgotten.
        PERSON: never "forgotten" in this sense — a fading PERSON
                memory is surfaced to caregivers via `at_risk`
                instead of being hidden from the patient.
        """
        if self.category != MemoryCategory.EPISODIC:
            return False
        return self.retention(now) < EPISODIC_ARCHIVE_THRESHOLD

    def access_frequency_score(self) -> float:
        return min(math.log1p(self.review_count) / math.log1p(100), 1.0)

    def consensus_score(self) -> float:
        if not self.shared:
            return 0.0
        return min(math.log1p(self.cross_agent_recall_count) / math.log1p(20), 1.0)

    def priority_score(self, now: Optional[datetime] = None) -> float:
        """Replay/dashboard priority score (not safety-critical)."""
        af = self.access_frequency_score()
        ti = max(0.0, min(self.importance, 1.0))
        fb = max(0.0, min(self.agent_feedback, 1.0))
        ts = max(0.0, min(self.task_success_rate, 1.0))
        cs = self.consensus_score() if self.shared else 0.0

        weighted = (
            0.15 * af + 0.18 * ti + 0.32 * fb + 0.27 * ts + 0.08 * cs
        )
        return weighted * self.retention(now)


# ─────────────────────────────────────────────────────────────
# EpisodicMemoryStore
# ─────────────────────────────────────────────────────────────

class EpisodicMemoryStore:
    """
    SQLite-backed episodic store, scoped per patient (agent_id).

    Category rules
    ──────────────
    CRITICAL  never archived, never index-cleaned, always in recall.
    ROUTINE   same as CRITICAL for this store (reminders live
              alongside episodic memory but are schedule-driven).
    PERSON    never archived; flagged at_risk instead when fading.
    EPISODIC  gently decays; archived (not deleted) when stale.

    prune_forgotten() only ever returns ARCHIVED EPISODIC ids — never
    CRITICAL, ROUTINE, or PERSON. Those ids are used upstream by
    MemoryOrchestrator.cleanup() to remove the *search index* entry in
    the semantic tier (so archived content stops surfacing in
    similarity search) — the durable SQLite row is untouched and stays
    available for caregiver/clinician review.

    Permanent deletion only ever happens through purge(), which
    requires a human actor and a reason and is logged.
    """

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._episodes: List[Episode] = []
        self._lock = threading.RLock()
        self._init_db()
        self._load_from_db()

    # ── Schema ────────────────────────────────────────────────

    def _init_db(self):
        conn = get_connection(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                agent_id TEXT,
                content TEXT,
                context TEXT,
                shared INTEGER,
                category TEXT DEFAULT 'episodic',
                stability_hours REAL,
                last_reviewed_at TEXT,
                review_count INTEGER,
                importance REAL,
                agent_feedback REAL,
                task_success_rate REAL,
                archived INTEGER DEFAULT 0,
                at_risk INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)
        # Lightweight migration for older DB files created before the
        # category/archived/at_risk columns existed.
        for stmt in (
            "ALTER TABLE episodes ADD COLUMN category TEXT DEFAULT 'episodic'",
            "ALTER TABLE episodes ADD COLUMN archived INTEGER DEFAULT 0",
            "ALTER TABLE episodes ADD COLUMN at_risk INTEGER DEFAULT 0",
        ):
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # column already exists

        conn.execute("""
            CREATE TABLE IF NOT EXISTS patient_profiles (
                patient_id TEXT PRIMARY KEY,
                stability_multiplier REAL DEFAULT 1.0,
                notes TEXT DEFAULT '',
                updated_by TEXT,
                updated_at TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                event_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                detail TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()

    def _load_from_db(self):
        conn = get_connection(self._db_path)
        cursor = conn.execute("SELECT * FROM episodes")
        cols = [d[0] for d in cursor.description]
        for row in cursor.fetchall():
            r = dict(zip(cols, row))
            ep = Episode(
                episode_id=r["episode_id"],
                agent_id=r["agent_id"],
                content=r["content"],
                context=safe_json_loads(r["context"]),
                shared=bool(r["shared"]),
                category=valid_category(r.get("category") or "episodic"),
                stability_hours=r["stability_hours"],
                last_reviewed_at=datetime.fromisoformat(r["last_reviewed_at"]),
                review_count=r["review_count"],
                importance=r["importance"],
                agent_feedback=r["agent_feedback"],
                task_success_rate=r["task_success_rate"],
                archived=bool(r.get("archived") or 0),
                at_risk=bool(r.get("at_risk") or 0),
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            self._episodes.append(ep)

    def _save_episode(self, ep: Episode):
        conn = get_connection(self._db_path)
        conn.execute("""
            INSERT OR REPLACE INTO episodes VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            ep.episode_id, ep.agent_id, str(ep.content),
            safe_json_dumps(ep.context), int(ep.shared), ep.category.value,
            ep.stability_hours, ep.last_reviewed_at.isoformat(),
            ep.review_count, ep.importance, ep.agent_feedback,
            ep.task_success_rate, int(ep.archived), int(ep.at_risk),
            ep.created_at.isoformat(),
        ))
        conn.commit()

    def _delete_episode_row(self, episode_id: str):
        conn = get_connection(self._db_path)
        conn.execute("DELETE FROM episodes WHERE episode_id = ?", (episode_id,))
        conn.commit()

    # ── Audit log (append-only) ─────────────────────────────────

    def _log(self, patient_id: str, actor: str, action: str, detail: str):
        conn = get_connection(self._db_path)
        conn.execute(
            "INSERT INTO audit_log VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), patient_id, actor, action, detail,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()

    def audit_trail(self, patient_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        conn = get_connection(self._db_path)
        rows = conn.execute(
            "SELECT actor, action, detail, timestamp FROM audit_log "
            "WHERE patient_id = ? ORDER BY timestamp DESC LIMIT ?",
            (patient_id, limit),
        ).fetchall()
        return [{"actor": r[0], "action": r[1], "detail": r[2], "timestamp": r[3]} for r in rows]

    # ── Per-patient clinical parameters (data-driven, auditable) ─

    def set_patient_profile(
        self,
        patient_id: str,
        stability_multiplier: float = 1.0,
        notes: str = "",
        actor: str = "caregiver",
    ):
        """
        The clinically-legitimate replacement for hardcoding a named
        patient's decay rate in source code: an explicit, logged,
        adjustable-by-a-human parameter. Applies uniformly through the
        same code path for every patient — nothing in this module
        branches on a specific patient_id.
        """
        if stability_multiplier <= 0:
            raise ValueError("stability_multiplier must be positive")

        conn = get_connection(self._db_path)
        conn.execute(
            "INSERT OR REPLACE INTO patient_profiles VALUES (?, ?, ?, ?, ?)",
            (patient_id, stability_multiplier, notes, actor,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        self._log(patient_id, actor, "set_patient_profile",
                   f"stability_multiplier={stability_multiplier} notes={notes!r}")

    def _patient_multiplier(self, patient_id: str) -> float:
        conn = get_connection(self._db_path)
        row = conn.execute(
            "SELECT stability_multiplier FROM patient_profiles WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()
        return row[0] if row else 1.0

    # ── Write ─────────────────────────────────────────────────

    def add(
        self,
        memory_id: str,
        agent_id: str,
        content: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        context = context or {}
        category = valid_category(context.get("category", "episodic"))

        importance = _heuristic_importance(str(content))
        base_stability = BASE_STABILITY_HOURS[category]

        if base_stability == float("inf"):
            stability = base_stability
        else:
            multiplier = self._patient_multiplier(agent_id)
            stability = min(
                base_stability * (1 + importance) * multiplier,
                MAX_STABILITY_HOURS,
            )

        with self._lock:
            ep = Episode(
                episode_id=memory_id,
                agent_id=agent_id,
                content=content,
                context=context,
                shared=context.get("shared", False),
                category=category,
                stability_hours=stability,
                importance=importance,
            )
            self._episodes.append(ep)
            self._save_episode(ep)
            self._log(agent_id, context.get("source_agent", "system"),
                       "add_memory", f"category={category.value} id={memory_id}")
        return ep.episode_id

    # ── Read ──────────────────────────────────────────────────

    def get_all(self) -> List[Episode]:
        return list(self._episodes)

    def get_by_id(self, episode_id: str) -> Optional[Episode]:
        for ep in self._episodes:
            if ep.episode_id == episode_id:
                return ep
        return None

    def critical_info(self, agent_id: str) -> List[Episode]:
        """Always-surfaced safety info for this patient."""
        return [
            ep for ep in self._episodes
            if ep.agent_id == agent_id and ep.category == MemoryCategory.CRITICAL
        ]

    def people_at_risk(self, agent_id: str) -> List[Episode]:
        """People whose recognition is fading — for caregiver attention."""
        return [
            ep for ep in self._episodes
            if ep.agent_id == agent_id and ep.category == MemoryCategory.PERSON
            and ep.at_risk
        ]

    # ── Remove (in-memory only; kept for interface parity) ─────

    def remove(self, episode_id: str) -> bool:
        before = len(self._episodes)
        self._episodes = [ep for ep in self._episodes if ep.episode_id != episode_id]
        return len(self._episodes) < before

    def delete(self, episode_id: str) -> bool:
        """
        NOTE: despite the name (kept for interface compatibility with
        MemoryOrchestrator.store()'s rollback path and .cleanup()),
        this only removes the in-memory + SQLite row for genuinely
        transient rollback cases (e.g. a failed multi-tier write that
        never should have been recorded). It is NOT used by the decay
        pipeline — apply_decay()/prune_forgotten() archive, they never
        call this. Prefer purge() for any human-initiated permanent
        deletion, since purge() requires an actor and a reason and is
        logged.
        """
        self._delete_episode_row(episode_id)
        return self.remove(episode_id)

    def purge(self, episode_id: str, actor: str, reason: str) -> bool:
        """
        The ONLY sanctioned way to permanently destroy a memory.
        Requires a human actor and a reason. Always logged. The
        system itself never calls this.
        """
        ep = self.get_by_id(episode_id)
        if ep is None:
            return False
        self._delete_episode_row(episode_id)
        self.remove(episode_id)
        self._log(ep.agent_id, actor, "purge", f"id={episode_id} reason={reason!r}")
        return True

    # ── Reinforcement ────────────────────────────────────────

    def recall(self, episode_id: str) -> Optional[Episode]:
        ep = self.get_by_id(episode_id)
        if ep:
            with self._lock:
                ep.review_count += 1
                ep.last_reviewed_at = datetime.now(timezone.utc)
                if ep.category in (MemoryCategory.PERSON, MemoryCategory.EPISODIC):
                    ep.stability_hours = reinforce_memory(
                        stability_hours=ep.stability_hours,
                        quality=1.0,
                        review_count=ep.review_count,
                    )
                # Successful recall clears both soft-state flags:
                # a recognized person is no longer at risk, and a
                # recalled episodic memory is no longer archived.
                ep.at_risk = False
                ep.archived = False
                self._save_episode(ep)
                self._log(ep.agent_id, "system", "reinforce", f"id={episode_id}")
        return ep

    # ── Retrieval ─────────────────────────────────────────────

    def grounded_retrieve(
        self,
        query: str,
        embedding: List[float],
        top_k: int = 5,
        agent_id: Optional[str] = None,
    ) -> List[Episode]:
        candidates = [
            ep for ep in self._episodes
            if not ep.is_forgotten()
            and (agent_id is None or ep.agent_id == agent_id or ep.shared)
        ]
        if not candidates or not embedding:
            return []

        scored = []
        for ep in candidates:
            ep_embedding = simple_embedding(str(ep.content))
            score = cosine_similarity(embedding, ep_embedding)
            if score >= SIMILARITY_THRESHOLD:
                scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    # ── Maintenance sweep ─────────────────────────────────────

    def apply_decay(self) -> None:
        """
        Sweep every episode and update archived/at_risk flags based
        on current retention. CRITICAL and ROUTINE are skipped
        entirely. Nothing is deleted here.
        """
        with self._lock:
            for ep in self._episodes:
                if ep.category in (MemoryCategory.CRITICAL, MemoryCategory.ROUTINE):
                    continue

                if ep.category == MemoryCategory.EPISODIC:
                    should_archive = ep.is_forgotten()
                    if should_archive != ep.archived:
                        ep.archived = should_archive
                        self._save_episode(ep)

                elif ep.category == MemoryCategory.PERSON:
                    should_flag = ep.retention() < PERSON_AT_RISK_THRESHOLD
                    if should_flag != ep.at_risk:
                        ep.at_risk = should_flag
                        self._save_episode(ep)
                        if should_flag:
                            self._log(
                                ep.agent_id, "system", "person_at_risk",
                                f"id={ep.episode_id} content={str(ep.content)[:60]!r}",
                            )

    def prune_forgotten(self) -> List[str]:
        """
        Returns the ids of EPISODIC memories currently archived, so
        the caller (MemoryOrchestrator.cleanup()) can remove the
        matching *search-index* entry in the semantic tier.

        IMPORTANT: this never returns CRITICAL, ROUTINE, or PERSON
        ids, and it never deletes the SQLite row — the durable record
        stays available for caregiver/clinician review. Nothing in
        this store is ever hard-deleted except through purge().
        """
        self.apply_decay()
        return [
            ep.episode_id for ep in self._episodes
            if ep.category == MemoryCategory.EPISODIC and ep.archived
        ]