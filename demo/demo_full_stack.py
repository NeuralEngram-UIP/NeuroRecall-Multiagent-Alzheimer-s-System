"""
demo_full_stack.py

Exercises the full memory/ stack exactly the way orchestrator.py does,
proving:
  1. It's a drop-in replacement (same constructor/method calls).
  2. Critical info never decays.
  3. An un-reinforced "person" memory gets flagged at_risk, not deleted.
  4. A reinforced "person" memory stays healthy.
  5. Stale episodic chatter gets archived (excluded from recall), but
     the row survives in SQLite -- nothing is destroyed.
  6. No hardcoded per-patient behavior: two different patient_ids run
     through identical code paths; only the auditable, human-set
     `set_patient_profile` can differentiate them.
  7. The audit trail records everything.
"""

import os
import sqlite3
from datetime import timedelta

from memory.episodic_memory import EpisodicMemoryStore
from memory.semantic_memory import SemanticMemoryStore
from memory.working_memory import WorkingMemory
from memory.memory_orchestrator import MemoryOrchestrator
from memory.memory_store import MemoryStore


def fake_embedding(seed: int):
    import random
    r = random.Random(seed)
    return [r.uniform(-1, 1) for _ in range(384)]


def main():
    for f in ("demo_multi_agent_memory.db",):
        if os.path.exists(f):
            os.remove(f)

    working = WorkingMemory(capacity=20)
    episodic = EpisodicMemoryStore(db_path="demo_multi_agent_memory.db")
    semantic = SemanticMemoryStore(storage_path="demo_qdrant_data")

    orch = MemoryOrchestrator(
        working_memory=working, episodic_memory=episodic, semantic_memory=semantic
    )
    store = MemoryStore(orchestrator=orch)  # <- exactly what orchestrator.py builds

    patient_a = "rajan"       # deliberately using the real name from the old bug
    patient_b = "alice"       # a second, differently-configured patient

    # ── Critical info (always surfaced) ──────────────────────
    store.store(
        agent_id=patient_a, content="Allergic to penicillin",
        embedding=fake_embedding(1),
        context={"category": "critical", "role": "system"},
    )
    store.store(
        agent_id=patient_a, content="Emergency contact: daughter Priya 555-0142",
        embedding=fake_embedding(2),
        context={"category": "critical", "role": "system"},
    )

    # ── People ────────────────────────────────────────────────
    son_result = store.store(
        agent_id=patient_a, content="Rohan is your son",
        embedding=fake_embedding(3),
        context={"category": "person", "role": "system"},
    )
    neighbor_result = store.store(
        agent_id=patient_a, content="Mrs. Alvarez is your neighbor",
        embedding=fake_embedding(4),
        context={"category": "person", "role": "system"},
    )

    # ── Episodic chatter ──────────────────────────────────────
    store.store(
        agent_id=patient_a, content="Had lunch with Rohan at the garden cafe",
        embedding=fake_embedding(5),
        context={"role": "user"},  # defaults to episodic
    )

    # ── Second patient, completely independent, no special-casing ──
    store.store(
        agent_id=patient_b, content="Alice is allergic to shellfish",
        embedding=fake_embedding(6),
        context={"category": "critical", "role": "system"},
    )

    print("=== Critical info, patient A ===")
    for ep in store.critical_info(patient_a):
        print(" -", ep.content)

    print("\n=== Critical info, patient B (independent) ===")
    for ep in store.critical_info(patient_b):
        print(" -", ep.content)

    # Reinforce the son's memory (simulating successful recognition);
    # deliberately do NOT reinforce the neighbor's.
    store.reinforce(agent_id=patient_a, memory_id=son_result["episodic_id"])

    # Force the neighbor's memory to look stale (simulate time passing
    # without reinforcement) by directly aging last_reviewed_at, the
    # same technique api/main.py's /simulate_time endpoint already uses.
    neighbor_ep = episodic.get_by_id(neighbor_result["episodic_id"])
    neighbor_ep.last_reviewed_at -= timedelta(hours=400)
    neighbor_ep.stability_hours = 200  # force low retention quickly for the demo
    episodic._save_episode(neighbor_ep)

    # Force the lunch memory to look stale the same way.
    lunch_id = [ep.episode_id for ep in episodic.get_all()
                if "lunch" in str(ep.content)][0]
    lunch_ep = episodic.get_by_id(lunch_id)
    lunch_ep.last_reviewed_at -= timedelta(days=30)
    episodic._save_episode(lunch_ep)

    print("\n=== Running maintenance sweep (apply_decay) ===")
    store.apply_decay()

    print("\n=== People at risk, patient A (caregiver alert) ===")
    for ep in store.people_at_risk(patient_a):
        print(" -", ep.content)

    print("\n=== Cleanup (archives stale episodic, cleans search index only) ===")
    removed = store.cleanup()
    print("removed (from search index, NOT from durable record):", removed)

    print("\n=== Row still exists in SQLite after 'cleanup' (never truly deleted) ===")
    conn = sqlite3.connect("demo_multi_agent_memory.db")
    row = conn.execute(
        "SELECT content, archived FROM episodes WHERE episode_id = ?", (lunch_id,)
    ).fetchone()
    print(" -", row)

    print("\n=== Data-driven per-patient tuning (no hardcoded names) ===")
    store.set_patient_profile(
        patient_a, stability_multiplier=0.5, notes="early-stage, faster review cadence",
        actor="caregiver:dr_mehta",
    )
    print("Set stability_multiplier=0.5 for patient_a via an auditable API call,")
    print("not a hardcoded 'if agent_id == \"rajan\"' branch in source.")

    print("\n=== Audit trail, patient A (most recent first) ===")
    for entry in store.audit_trail(patient_a)[:6]:
        print(" -", entry["timestamp"], entry["actor"], entry["action"], entry["detail"])

    print("\n=== Restarting the semantic store to prove persistence ===")
    semantic.client.close()
    del semantic
    semantic2 = SemanticMemoryStore(storage_path="demo_qdrant_data")
    print("total_memories after restart:", semantic2.count())


if __name__ == "__main__":
    main()