# memory/models.py

"""
Category model for the Alzheimer's-aware memory tier.

Design principle
─────────────────
A generic AI-agent "forgetting curve" is appropriate for casual
conversational context. It is NOT appropriate for "the patient is
allergic to penicillin." So memories are split into categories with
different, clinically-motivated rules — instead of one decay curve
applied uniformly to everything a patient says.

    CRITICAL   Never decays. Never archived. Always in active recall.
               (medications, allergies, emergency contacts, home
                address, diagnosis details the care team always wants
                visible)

    PERSON     Decays slowly. Reinforced by successful recognition.
               Never archived/hidden — instead flagged `at_risk` so a
               caregiver is alerted well before recognition would
               actually fail.
               (family, friends, caregivers, and their relationship
                to the patient)

    ROUTINE    Not decay-driven — schedule-driven. Handled by the
               reminder system, not the forgetting curve.
               (take medication at 8am, Tuesday doctor's appointment)

    EPISODIC   Normal, gentle Ebbinghaus-style decay. "Forgotten"
               means archived / dropped from the active-recall index,
               never physically deleted from the durable record.
               (day-to-day conversations and events)

Per-patient variation (e.g. one patient needs gentler or faster
decay than another) is a legitimate clinical need — but it must be
expressed as auditable *data* set by a caregiver/clinician through
`set_patient_profile()`, never as a hardcoded name or ID in source
code. There is no per-patient branching anywhere in this module.
"""

from __future__ import annotations

from enum import Enum


class MemoryCategory(str, Enum):
    CRITICAL = "critical"
    PERSON = "person"
    ROUTINE = "routine"
    EPISODIC = "episodic"


# Baseline stability (hours) before category-specific handling kicks in.
# CRITICAL and ROUTINE are not decay-driven at all; included here only
# so every category has a well-defined value.
BASE_STABILITY_HOURS = {
    MemoryCategory.CRITICAL: float("inf"),
    MemoryCategory.PERSON: 24 * 90,     # ~90 days baseline, reinforced on recognition
    MemoryCategory.ROUTINE: float("inf"),
    MemoryCategory.EPISODIC: 24 * 3,    # ~3 days baseline
}

# Below this retention score, an EPISODIC memory is archived
# (excluded from active recall; the row itself is never deleted).
EPISODIC_ARCHIVE_THRESHOLD = 0.25

# Below this retention score, a PERSON memory is flagged at_risk
# so a caregiver is alerted — well above the point where it would
# ever be hidden, since PERSON memories are never auto-archived.
PERSON_AT_RISK_THRESHOLD = 0.55


def valid_category(value: str) -> MemoryCategory:
    """Safe coercion with a sane default (EPISODIC) for unrecognized input."""
    try:
        return MemoryCategory(value)
    except ValueError:
        return MemoryCategory.EPISODIC