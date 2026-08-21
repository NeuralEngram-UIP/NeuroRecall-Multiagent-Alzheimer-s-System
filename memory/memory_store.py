"""
Unified Multi-Agent Memory Store
"""

from typing import Any, Dict, List, Optional

from memory.memory_orchestrator import (
    MemoryOrchestrator
)


class MemoryStore:
    """
    Unified memory access layer.
    """

    def __init__(
        self,
        orchestrator: MemoryOrchestrator
    ):

        self.orchestrator = (
            orchestrator
        )

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
        """
        Store memory across all tiers.
        """

        if not agent_id.strip():

            raise ValueError(
                "agent_id cannot be empty"
            )

        if not content.strip():

            raise ValueError(
                "content cannot be empty"
            )

        return self.orchestrator.store(
            agent_id=agent_id,
            content=content,
            context=context,
            embedding=embedding
        )

    # ─────────────────────────────────────────────────────

    def retrieve(
        self,
        agent_id: str,
        query: str,
        embedding: List[float],
        top_k: int = 5,
        token_budget: int = 4000
    ) -> Dict[str, Any]:
        """
        Retrieve relevant memories.
        """

        return self.orchestrator.retrieve(
            agent_id=agent_id,
            query=query,
            embedding=embedding,
            top_k=top_k,
            token_budget=token_budget
        )

    # ─────────────────────────────────────────────────────

    def get_context(
        self,
        agent_id: str,
        limit: int = 10
    ):
        """
        Retrieve scoped working memory.
        """

        return self.orchestrator.get_context(
            agent_id=agent_id,
            limit=limit
        )

    # ─────────────────────────────────────────────────────

    def reinforce(
        self,
        agent_id: str,
        memory_id: str
    ):
        """
        Reinforce memory.
        """

        self.orchestrator.reinforce(
            agent_id=agent_id,
            memory_id=memory_id
        )

    # ─────────────────────────────────────────────────────

    def cleanup(self) -> int:

        return (
            self.orchestrator
            .cleanup()
        )

    # ─────────────────────────────────────────────────────

    def apply_decay(self):

        self.orchestrator.apply_decay()

    # ─────────────────────────────────────────────────────

    def metrics(self):

        if hasattr(
            self.orchestrator,
            "metrics"
        ):
            return (
                self.orchestrator
                .metrics()
            )

        return {}

    # ─────────────────────────────────────────────────────
    # Alzheimer's-specific passthroughs (additive only)
    # ─────────────────────────────────────────────────────

    def critical_info(self, agent_id: str):
        """Always-surfaced safety info for this patient."""
        return self.orchestrator.critical_info(agent_id)

    def people_at_risk(self, agent_id: str):
        """People whose recognition is fading — caregiver alert list."""
        return self.orchestrator.people_at_risk(agent_id)

    def set_patient_profile(
        self,
        agent_id: str,
        stability_multiplier: float = 1.0,
        notes: str = "",
        actor: str = "caregiver",
    ):
        """Auditable per-patient decay tuning, set by a human, not code."""
        self.orchestrator.set_patient_profile(
            agent_id, stability_multiplier, notes, actor
        )

    def purge_memory(self, episode_id: str, actor: str, reason: str) -> bool:
        """The only sanctioned path for permanent deletion. Logged."""
        return self.orchestrator.purge_memory(episode_id, actor, reason)

    def audit_trail(self, agent_id: str, limit: int = 100):
        return self.orchestrator.audit_trail(agent_id, limit)