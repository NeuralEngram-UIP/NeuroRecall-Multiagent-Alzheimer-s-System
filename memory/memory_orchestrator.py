"""
FINAL HARDENED
ENTERPRISE-GRADE
MULTI-AGENT MEMORY ORCHESTRATOR
"""

import logging
import threading
import uuid

from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

EXPECTED_EMBEDDING_DIM = 384


class MemoryOrchestrator:
    """
    Coordinates:
    - working memory
    - episodic memory
    - semantic memory
    - reinforcement
    - cleanup
    - retrieval fusion
    - token budgeting

    Design Goals
    ────────────
    - thread safety
    - scoped memory isolation
    - rollback handling
    - bounded retrieval
    - scalable concurrency
    - production-safe validation
    """

    def __init__(
        self,
        working_memory,
        episodic_memory,
        semantic_memory
    ):

        self.working_memory = (
            working_memory
        )

        self.episodic_memory = (
            episodic_memory
        )

        self.semantic_memory = (
            semantic_memory
        )

        # tier-level locks

        self.working_lock = (
            threading.RLock()
        )

        self.episodic_lock = (
            threading.RLock()
        )

        self.semantic_lock = (
            threading.RLock()
        )

    # ─────────────────────────────────────────────────────
    # Store
    # ─────────────────────────────────────────────────────

    def store(
        self,
        agent_id: str,
        content: str,
        embedding: List[float],
        context: Optional[
            Dict[str, Any]
        ] = None
    ) -> Dict[str, str]:

        context = context or {}

        self._validate_embedding(
            embedding
        )

        role = context.get(
            "role",
            "user"
        )

        memory_id = str(
            uuid.uuid4()
        )

        episodic_id = None

        try:

            # ─────────────────────────────────────────
            # Working Memory
            # ─────────────────────────────────────────

            # NOTE:
            # Working memory intentionally
            # does not participate in rollback.
            #
            # It is:
            # - transient
            # - bounded
            # - non-persistent
            # - eventually overwritten

            with self.working_lock:

                self.working_memory.add(
                    # BUGFIX: this was previously commented out, so
                    # every patient/agent's working-memory writes fell
                    # through to the same "default" buffer while reads
                    # (get_context/as_messages) were scoped per
                    # agent_id -- meaning patient A's recent context
                    # could bleed into patient B's conversation. This
                    # is a real cross-patient privacy issue, not just
                    # a bookkeeping bug, so it's restored here.
                    agent_id=agent_id,
                    role=role,
                    content=content
                )

            # ─────────────────────────────────────────
            # Episodic Memory
            # ─────────────────────────────────────────

            with self.episodic_lock:

                episodic_id = (
                    self.episodic_memory.add(
                        memory_id=memory_id,
                        agent_id=agent_id,
                        content=content,
                        context={
                            **context,

                            "source_agent": (
                                agent_id
                            ),

                            "confidence": (
                                context.get(
                                    "confidence",
                                    1.0
                                )
                            ),

                            "shared": (
                                context.get(
                                    "shared",
                                    False
                                )
                            )
                        }
                    )
                )

            # ─────────────────────────────────────────
            # Semantic Memory
            # ─────────────────────────────────────────

            with self.semantic_lock:

                semantic_id = (
                    self.semantic_memory.add(
                        memory_id=memory_id,
                        content=content,
                        embedding=embedding,
                        metadata={

                            "agent_id": (
                                agent_id
                            ),

                            "episodic_id": (
                                episodic_id
                            ),

                            "shared": (
                                context.get(
                                    "shared",
                                    False
                                )
                            ),

                            "confidence": (
                                context.get(
                                    "confidence",
                                    1.0
                                )
                            ),

                            "role": role
                        }
                    )
                )

            logger.info(
                "stored memory "
                "memory_id=%s "
                "agent_id=%s",
                memory_id,
                agent_id
            )

            return {
                "memory_id": memory_id,
                "episodic_id": episodic_id,
                "semantic_id": semantic_id
            }

        except Exception:

            logger.exception(
                "store failed "
                "memory_id=%s",
                memory_id
            )

            # NOTE:
            # Per-tier locking allows
            # higher concurrency than
            # a global orchestrator lock.
            #
            # This creates a small TOCTOU
            # window where another thread
            # may briefly observe an
            # episodic entry before
            # rollback removes it.
            #
            # This is considered acceptable
            # because:
            # - rollback events are rare
            # - eventual consistency is
            #   sufficient for cognition
            # - semantic tier acts as the
            #   authoritative retrieval layer

            if episodic_id is not None:

                try:

                    with self.episodic_lock:

                        self.episodic_memory.delete(
                            episodic_id
                        )

                except Exception:

                    logger.exception(
                        "rollback failed "
                        "episodic_id=%s",
                        episodic_id
                    )

            raise

    # ─────────────────────────────────────────────────────
    # Retrieve
    # ─────────────────────────────────────────────────────

    def retrieve(
        self,
        agent_id: str,
        query: str,
        embedding: List[float],
        top_k: int = 5,
        token_budget: int = 4000
    ) -> Dict[str, Any]:

        self._validate_embedding(
            embedding
        )

        # ─────────────────────────────────────────
        # Semantic Retrieval
        # ─────────────────────────────────────────

        with self.semantic_lock:

            semantic_results = (
                self.semantic_memory.search_multi(
                    embedding=embedding,
                    limit=top_k,
                    agent_id=agent_id
                )
            )

        # ─────────────────────────────────────────
        # Episodic Retrieval
        # ─────────────────────────────────────────

        with self.episodic_lock:

            episodic_results = (
                self.episodic_memory
                .grounded_retrieve(
                    query=query,
                    embedding=embedding,
                    top_k=top_k,
                    agent_id=agent_id
                )
            )

        # ─────────────────────────────────────────
        # Working Context
        # ─────────────────────────────────────────

        with self.working_lock:

            working_context = (
                self.working_memory
                .as_messages(
                    agent_id=agent_id,
                    limit=5
                )
            )

        # ─────────────────────────────────────────
        # Fusion Layer
        # ─────────────────────────────────────────

        fused_results = (
            self._budgeted_fusion(
                episodic_results,
                semantic_results,
                token_budget
            )
        )

        return {
            "query": query,

            "working_memory": (
                working_context
            ),

            "episodic_memory": (
                episodic_results
            ),

            "semantic_memory": (
                semantic_results
            ),

            "fused_memory": (
                fused_results
            )
        }

    # ─────────────────────────────────────────────────────
    # Context
    # ─────────────────────────────────────────────────────

    def get_context(
        self,
        agent_id: str,
        limit: int = 10
    ):

        with self.working_lock:

            return (
                self.working_memory
                .as_messages(
                    agent_id=agent_id,
                    limit=limit
                )
            )

    # ─────────────────────────────────────────────────────
    # Reinforcement
    # ─────────────────────────────────────────────────────

    def reinforce(
        self,
        agent_id: str,
        memory_id: str
    ):

        with self.episodic_lock:

            self.episodic_memory.recall(
                episode_id=memory_id
            )

        with self.semantic_lock:

            self.semantic_memory.reinforce(
                memory_id
            )

        logger.info(
            "reinforced memory "
            "agent_id=%s "
            "memory_id=%s",
            agent_id,
            memory_id
        )

    # ─────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────

    def cleanup(self) -> int:
        """
        Cleanup forgotten memories
        across all tiers.

        EXPECTS:
            prune_forgotten()
            -> List[str]
        """

        with self.episodic_lock:

            removed_ids = (
                self.episodic_memory
                .prune_forgotten()
            )

        if not isinstance(
            removed_ids,
            list
        ):

            raise TypeError(
                "prune_forgotten() "
                "must return List[str]"
            )

        # NOTE:
        # Episodic and semantic cleanup
        # are intentionally non-atomic.
        #
        # A concurrent store using a
        # recycled memory_id between
        # these lock boundaries could
        # theoretically create an
        # orphaned semantic entry.
        #
        # This is considered acceptable
        # because:
        # - memory_id uses UUID4
        # - collision probability is
        #   astronomically negligible

        with self.semantic_lock:

            for memory_id in removed_ids:

                try:

                    self.semantic_memory.delete(
                        memory_id
                    )

                except Exception:

                    logger.exception(
                        "semantic cleanup failed "
                        "memory_id=%s",
                        memory_id
                    )

        logger.info(
            "cleanup complete "
            "removed=%s",
            len(removed_ids)
        )

        return len(removed_ids)

    # ─────────────────────────────────────────────────────
    # Decay
    # ─────────────────────────────────────────────────────

    def apply_decay(self):

        with self.episodic_lock:

            self.episodic_memory.apply_decay()

    # ─────────────────────────────────────────────────────
    # Fusion Layer
    # ─────────────────────────────────────────────────────

    def _budgeted_fusion(
        self,
        episodic_results,
        semantic_results,
        token_budget: int
    ):
        """
        Deduplicated memory fusion
        with token budgeting.

        Design Choice
        ─────────────
        Retrieval order is assumed
        to already reflect relevance.

        Therefore:
        - higher-ranked memories
          are prioritised
        - lower-ranked memories are
          ignored once budget fills

        We intentionally BREAK
        instead of CONTINUE when
        the budget is exceeded.

        This preserves ranking fidelity
        over maximum coverage.
        """

        combined = []

        seen = set()

        for memory in (
            episodic_results
            + semantic_results
        ):

            memory_id = getattr(
                memory,
                "memory_id",
                None
            )

            content = getattr(
                memory,
                "content",
                ""
            )

            dedup_key = (
                memory_id
                or content.strip()
            )

            if dedup_key in seen:
                continue

            seen.add(dedup_key)

            combined.append(memory)

        current_tokens = 0

        filtered = []

        for memory in combined:

            content = getattr(
                memory,
                "content",
                ""
            )

            estimated_tokens = (
                self._estimate_tokens(
                    content
                )
            )

            if (
                current_tokens
                + estimated_tokens
                > token_budget
            ):

                # preserve relevance ordering

                break

            filtered.append(memory)

            current_tokens += (
                estimated_tokens
            )

        return filtered

    # ─────────────────────────────────────────────────────
    # Token Estimation
    # ─────────────────────────────────────────────────────

    def _estimate_tokens(
        self,
        text: str
    ) -> int:
        """
        Rough token estimation.

        Future upgrade path:
        - tiktoken
        - Anthropic tokenizer
        """

        text = text.strip()

        if not text:
            return 0

        words = len(
            text.split()
        )

        return int(words * 1.3)

    # ─────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────

    def _validate_embedding(
        self,
        embedding: List[float]
    ):

        if not isinstance(
            embedding,
            list
        ):

            raise ValueError(
                "embedding must be a list"
            )

        if not embedding:

            raise ValueError(
                "embedding cannot be empty"
            )

        if len(embedding) != (
            EXPECTED_EMBEDDING_DIM
        ):

            raise ValueError(
                f"expected embedding "
                f"dimension "
                f"{EXPECTED_EMBEDDING_DIM}, "
                f"got {len(embedding)}"
            )

        if not all(
            isinstance(
                value,
                (int, float)
            )
            for value in embedding
        ):

            raise ValueError(
                "embedding must contain "
                "numeric values"
            )
    def metrics(self, agent_id: str = None) -> Dict[str, Any]:

        """
    Returns live memory panel data.
    """
        with self.episodic_lock:
            all_episodes = self.episodic_memory.get_all()

    # filter by agent if specified
        episodes = [
        ep for ep in all_episodes
        if (
        agent_id is None
        or ep.agent_id == agent_id
        or ep.shared
    )
        and ep.context.get("role") != "assistant"
]

        total = len(episodes)
        if total == 0:
            return {
            "total_memories": 0,
            "avg_retention": 0.0,
            "avg_stability": 0.0,
            "avg_priority": 0.0,
            "forgotten_count": 0,
            "memories": []
        }

        retentions = []
        stabilities = []
        priorities = []
        forgotten = 0
        memory_list = []

        for ep in episodes:
            r = ep.retention()        # live R(t)
            p = ep.priority_score()   # live DMS score
            s = ep.stability_hours

            retentions.append(r)
            stabilities.append(s)
            priorities.append(p)

            if ep.is_forgotten():
                forgotten += 1

            memory_list.append({
            "id": ep.episode_id,
            "content": str(ep.content)[:80],
            "retention": round(r, 3),
            "stability_hours": round(s, 2),
            "priority_score": round(p, 3),
            "access_count": ep.review_count,
            "agent_id": ep.agent_id,
            "shared": ep.shared,
           })

        return {
        "total_memories": total,
        "avg_retention": round(
            sum(retentions) / total, 3
        ),
        "avg_stability": round(
            sum(stabilities) / total, 2
        ),
        "avg_priority": round(
            sum(priorities) / total, 3
        ),
        "forgotten_count": forgotten,
        "memories": memory_list
    }

    # ─────────────────────────────────────────────────────
    # Alzheimer's-specific passthroughs (additive only —
    # nothing above this point changed its signature)
    # ─────────────────────────────────────────────────────

    def critical_info(self, agent_id: str):
        """Always-surfaced safety info (medications, allergies, etc)."""
        with self.episodic_lock:
            return self.episodic_memory.critical_info(agent_id)

    def people_at_risk(self, agent_id: str):
        """People whose recognition is fading — caregiver alert list."""
        with self.episodic_lock:
            return self.episodic_memory.people_at_risk(agent_id)

    def set_patient_profile(
        self,
        agent_id: str,
        stability_multiplier: float = 1.0,
        notes: str = "",
        actor: str = "caregiver",
    ):
        """
        Data-driven, auditable per-patient decay tuning — the sanctioned
        replacement for ever hardcoding a patient's name/id in code.
        """
        with self.episodic_lock:
            self.episodic_memory.set_patient_profile(
                agent_id, stability_multiplier, notes, actor
            )

    def purge_memory(self, episode_id: str, actor: str, reason: str) -> bool:
        """The only sanctioned path for permanent deletion. Logged."""
        with self.episodic_lock:
            return self.episodic_memory.purge(episode_id, actor, reason)

    def audit_trail(self, agent_id: str, limit: int = 100):
        with self.episodic_lock:
            return self.episodic_memory.audit_trail(agent_id, limit)