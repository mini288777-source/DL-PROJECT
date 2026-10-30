"""
Canonical Taxonomy Definition for Continual Personalization (Phase 7).
Inherits and enforces the canonical 7-class intent taxonomy from config/taxonomy.yaml.
Zero tolerance for class renaming, aliasing, or unauthorized splitting/merging.
"""

from typing import Dict, List, Set, Tuple

# Canonical 7 classes in strict authoritative order
CANONICAL_CLASSES: Tuple[str, ...] = (
    "GENERAL_SPEECH",
    "CRITICAL_RESPONSE",
    "FALL",
    "BREATHING_MEDICAL",
    "PAIN",
    "SOS_HELP",
    "LOCATION",
)

EMERGENCY_CLASSES: Tuple[str, ...] = (
    "CRITICAL_RESPONSE",
    "FALL",
    "BREATHING_MEDICAL",
    "PAIN",
    "SOS_HELP",
    "LOCATION",
)

GENERAL_SPEECH_CLASS: str = "GENERAL_SPEECH"

CLASS_SET: Set[str] = set(CANONICAL_CLASSES)
EMERGENCY_SET: Set[str] = set(EMERGENCY_CLASSES)

# Hierarchical Stage-1 mapping: 0 = GENERAL_SPEECH, 1 = EMERGENCY
STAGE1_BINARY_MAPPING: Dict[str, int] = {
    "GENERAL_SPEECH": 0,
    "CRITICAL_RESPONSE": 1,
    "FALL": 1,
    "BREATHING_MEDICAL": 1,
    "PAIN": 1,
    "SOS_HELP": 1,
    "LOCATION": 1,
}


def is_valid_intent(intent: str) -> bool:
    """Return True if intent is one of the 7 canonical classes."""
    return intent in CLASS_SET


def is_emergency_intent(intent: str) -> bool:
    """Return True if intent is an emergency class (Stage 1 == 1)."""
    return intent in EMERGENCY_SET


def get_stage1_label(intent: str) -> int:
    """Return binary stage 1 label (0 for general, 1 for emergency)."""
    if intent not in STAGE1_BINARY_MAPPING:
        raise ValueError(f"Unknown intent: {intent}. Must be one of {CANONICAL_CLASSES}")
    return STAGE1_BINARY_MAPPING[intent]
