# working_memory.py

"""
Enterprise-Grade Working Memory

Short-term active conversational memory.

Responsibilities
────────────────
- maintain recent interaction context
- provide bounded memory window
- support prompt assembly
- remain lightweight and thread-safe
- provide snapshot-safe retrieval
- support observability metrics
- enforce memory constraints
- support graceful truncation

Architecture
────────────
Working memory is intentionally:
- lightweight
- bounded
- fast
- non-persistent

Long-term cognition belongs to:
- episodic memory
- semantic memory
"""

import copy
import logging
import threading

from collections import deque
from dataclasses import dataclass

from typing import (
    Dict,
    List,
    Optional
)


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

VALID_ROLES = {
    "user",
    "assistant",
    "system"
}

DEFAULT_CAPACITY = 20

MAX_CONTENT_LENGTH = 10000

TRUNCATION_SUFFIX = (
    "\n...[truncated]"
)


# ─────────────────────────────────────────────────────────────
# Working Memory Item
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WorkingMemoryItem:
    """
    Immutable working memory item.
    """

    role: str

    content: str


# ─────────────────────────────────────────────────────────────
# Working Memory
# ─────────────────────────────────────────────────────────────

class WorkingMemory:
    """
    Bounded conversational context.
    Scoped per agent_id.
    """

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY
    ):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")

        self.capacity = capacity
        self._buffers: Dict[str, deque] = {}
        self.lock = threading.RLock()
        self.total_adds = 0
        self.total_evictions = 0
        self.total_snapshots = 0
        self.total_clears = 0
        self.total_truncations = 0

        logger.info(
            "WorkingMemory initialized capacity=%s",
            capacity
        )

    def _get_buffer(self, agent_id: str) -> deque:
        if agent_id not in self._buffers:
            self._buffers[agent_id] = deque(
                maxlen=self.capacity
            )
        return self._buffers[agent_id]

    def add(
        self,
        role: str,
        content: str,
        agent_id: str = "default"
    ):
        role = role.strip()
        if role not in VALID_ROLES:
            raise ValueError(
                f"role must be one of {VALID_ROLES}"
            )

        content = content.strip()
        if not content:
            raise ValueError("content cannot be empty")

        truncated = False
        if len(content) > MAX_CONTENT_LENGTH:
            allowed_length = (
                MAX_CONTENT_LENGTH
                - len(TRUNCATION_SUFFIX)
            )
            content = content[:allowed_length] + TRUNCATION_SUFFIX
            truncated = True

        item = WorkingMemoryItem(role=role, content=content)

        with self.lock:
            if truncated:
                self.total_truncations += 1
            buffer = self._get_buffer(agent_id)
            if len(buffer) == self.capacity:
                self.total_evictions += 1
            buffer.append(item)
            self.total_adds += 1

    def recent(
        self,
        limit: int = 5,
        role_filter: Optional[str] = None,
        agent_id: str = "default"
    ) -> List[WorkingMemoryItem]:
        if limit <= 0:
            raise ValueError("limit must be > 0")

        with self.lock:
            buffer = self._get_buffer(agent_id)
            items = list(buffer)

        if role_filter:
            items = [
                item for item in items
                if item.role == role_filter
            ]

        return items[-limit:]

    def as_messages(
        self,
        limit: int = 10,
        role_filter: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        messages = self.recent(
            limit=limit,
            role_filter=role_filter,
            agent_id=agent_id or "default"
        )
        return [
            {"role": item.role, "content": item.content}
            for item in messages
        ]

    def snapshot(
        self,
        agent_id: str = "default"
    ) -> List[WorkingMemoryItem]:
        with self.lock:
            buffer = self._get_buffer(agent_id)
            snapshot = copy.deepcopy(list(buffer))
            self.total_snapshots += 1
        return snapshot

    def format_for_prompt(
        self,
        limit: int = 10,
        role_filter: Optional[str] = None,
        agent_id: str = "default"
    ) -> str:
        messages = self.recent(
            limit=limit,
            role_filter=role_filter,
            agent_id=agent_id
        )
        return "\n".join(
            f"{item.role}: {item.content}"
            for item in messages
        )

    def clear(self, agent_id: str = "default"):
        with self.lock:
            if agent_id in self._buffers:
                self._buffers[agent_id].clear()
            self.total_clears += 1

    def size(self, agent_id: str = "default") -> int:
        with self.lock:
            return len(self._get_buffer(agent_id))

    def is_empty(self, agent_id: str = "default") -> bool:
        return self.size(agent_id) == 0

    def metrics(self) -> Dict[str, int]:
        with self.lock:
            return {
                "capacity": self.capacity,
                "total_adds": self.total_adds,
                "total_evictions": self.total_evictions,
                "total_snapshots": self.total_snapshots,
                "total_truncations": self.total_truncations,
                "total_clears": self.total_clears
            }

    def __len__(self) -> int:
        return sum(
            len(b) for b in self._buffers.values()
        )