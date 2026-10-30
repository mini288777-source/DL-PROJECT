"""
TORGO Intent Taxonomy and Mapping Module.

Evaluates every TORGO utterance against the emergency intent taxonomy
using transparent, deterministic rules.
Generates:
    - data/manifests/torgo_intent_mapping.csv
"""

import glob
import os
import re
from typing import Any, Dict, List, Tuple
import pandas as pd
import yaml


def load_taxonomy(taxonomy_path: str = "configs/intent_taxonomy.yaml") -> Dict[str, Any]:
    """Load intent taxonomy configuration."""
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def classify_utterance_intent(
    transcript: str,
    taxonomy: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Classify a single transcript into an intent candidate and mapping status.

    Statuses:
        - VALID: Unambiguous emergency intent utterance (e.g., isolated emergency keyword).
        - AMBIGUOUS: Contains emergency token but context is ambiguous or mixed.
        - EXCLUDED: Keyword appears as incidental phonetic token inside literary reading sentence.
        - UNMAPPED: Standard non-emergency speech stimulus / reading vocabulary.
    """
    clean_text = transcript.strip().lower()
    tokens = [re.sub(r"[^\w]", "", w) for w in clean_text.split() if re.sub(r"[^\w]", "", w)]
    token_count = len(tokens)

    # 1. Exact match for single-word emergency commands / intents
    single_word_map = {
        # SOS / HELP
        "help": ("SOS_HELP", "VALID", "EXACT_SINGLE_WORD", 1.0, False, "Explicit isolated distress call"),
        "trouble": ("SOS_HELP", "VALID", "EXACT_SINGLE_WORD", 0.90, False, "Distress/trouble invocation"),
        "emergency": ("SOS_HELP", "VALID", "EXACT_SINGLE_WORD", 1.0, False, "Explicit emergency call"),

        # FALL
        "fall": ("FALL", "VALID", "EXACT_SINGLE_WORD", 1.0, False, "Isolated fall event indicator"),
        "fell": ("FALL", "VALID", "EXACT_SINGLE_WORD", 0.95, False, "Past tense fall event indicator"),
        "slip": ("FALL", "VALID", "EXACT_SINGLE_WORD", 0.95, False, "Isolated slip indicator"),
        "drop": ("FALL", "VALID", "EXACT_SINGLE_WORD", 0.85, False, "Drop/collapse indicator"),
        "down": ("FALL", "VALID", "EXACT_SINGLE_WORD", 0.80, False, "Down/collapse indicator"),
        "floor": ("FALL", "VALID", "EXACT_SINGLE_WORD", 0.80, False, "Grounded on floor indicator"),

        # BREATHING / MEDICAL
        "air": ("BREATHING_MEDICAL", "VALID", "EXACT_SINGLE_WORD", 0.85, False, "Air/breathing deficit indicator"),
        "pill": ("BREATHING_MEDICAL", "VALID", "EXACT_SINGLE_WORD", 0.85, False, "Urgent medication request"),
        "breath": ("BREATHING_MEDICAL", "VALID", "EXACT_SINGLE_WORD", 1.0, False, "Breathing distress indicator"),
        "cough": ("BREATHING_MEDICAL", "VALID", "EXACT_SINGLE_WORD", 0.90, False, "Respiratory symptom indicator"),

        # PAIN
        "pain": ("PAIN", "VALID", "EXACT_SINGLE_WORD", 1.0, False, "Direct acute pain report"),
        "hurt": ("PAIN", "VALID", "EXACT_SINGLE_WORD", 0.95, False, "Direct physical injury report"),
        "ache": ("PAIN", "VALID", "EXACT_SINGLE_WORD", 0.95, False, "Somatic ache report"),
        "burn": ("PAIN", "VALID", "EXACT_SINGLE_WORD", 0.85, False, "Burning pain report"),
        "bad": ("PAIN", "VALID", "EXACT_SINGLE_WORD", 0.75, False, "Acute distress/bad condition"),

        # LOCATION
        "room": ("LOCATION", "VALID", "EXACT_SINGLE_WORD", 0.85, False, "Isolated room reference"),
        "bed": ("LOCATION", "VALID", "EXACT_SINGLE_WORD", 0.85, False, "Isolated bed location reference"),
        "door": ("LOCATION", "VALID", "EXACT_SINGLE_WORD", 0.80, False, "Isolated door reference"),

        # CRITICAL RESPONSE
        "yes": ("CRITICAL_RESPONSE", "VALID", "EXACT_SINGLE_WORD", 1.0, False, "Dispatcher confirmation"),
        "no": ("CRITICAL_RESPONSE", "VALID", "EXACT_SINGLE_WORD", 1.0, False, "Dispatcher negative response"),
        "stop": ("CRITICAL_RESPONSE", "VALID", "EXACT_SINGLE_WORD", 1.0, False, "Emergency halt command"),
        "go": ("CRITICAL_RESPONSE", "VALID", "EXACT_SINGLE_WORD", 0.90, False, "Emergency proceed command"),
        "now": ("CRITICAL_RESPONSE", "VALID", "EXACT_SINGLE_WORD", 0.90, False, "Time-critical imperative"),
        "wait": ("CRITICAL_RESPONSE", "VALID", "EXACT_SINGLE_WORD", 0.90, False, "Dispatcher wait directive")
    }

    if token_count == 1 and tokens[0] in single_word_map:
        cand, status, rule, conf, rev, note = single_word_map[tokens[0]]
        return {
            "intent_candidate": cand,
            "mapping_status": status,
            "mapping_rule": rule,
            "mapping_confidence": conf,
            "manual_review_required": rev,
            "notes": note
        }

    # 2. Short multi-word phrases with clear emergency head
    if clean_text in ["skip ah slip", "skip a slip"]:
        return {
            "intent_candidate": "FALL",
            "mapping_status": "VALID",
            "mapping_rule": "PHRASE_EMERGENCY_HEAD",
            "mapping_confidence": 0.80,
            "manual_review_required": False,
            "notes": "Short phrase with slip head"
        }

    # 3. Reading sentences containing emergency keywords incidental to standard passages
    # Check if any emergency keyword appears in multi-word sentence
    sos_kws = ["help", "trouble", "save"]
    fall_kws = ["fall", "fell", "slip", "drop", "floor", "down"]
    med_kws = ["air", "pill", "medicine", "breath"]
    pain_kws = ["pain", "hurt", "ache", "burn", "bad", "hard"]

    if any(kw in tokens for kw in sos_kws):
        return {
            "intent_candidate": "SOS_HELP",
            "mapping_status": "EXCLUDED",
            "mapping_rule": "INCIDENTAL_PASSAGE_KEYWORD",
            "mapping_confidence": 0.40,
            "manual_review_required": False,
            "notes": "Keyword appears in non-emergency reading sentence; excluded from emergency intent."
        }

    if any(kw in tokens for kw in fall_kws):
        return {
            "intent_candidate": "FALL",
            "mapping_status": "EXCLUDED",
            "mapping_rule": "INCIDENTAL_PASSAGE_KEYWORD",
            "mapping_confidence": 0.40,
            "manual_review_required": False,
            "notes": "Fall token appears in narrative reading passage; excluded from emergency intent."
        }

    if any(kw in tokens for kw in med_kws):
        return {
            "intent_candidate": "BREATHING_MEDICAL",
            "mapping_status": "EXCLUDED",
            "mapping_rule": "INCIDENTAL_PASSAGE_KEYWORD",
            "mapping_confidence": 0.40,
            "manual_review_required": False,
            "notes": "Medical/air token appears in reading passage; excluded from emergency intent."
        }

    if any(kw in tokens for kw in pain_kws):
        return {
            "intent_candidate": "PAIN",
            "mapping_status": "EXCLUDED",
            "mapping_rule": "INCIDENTAL_PASSAGE_KEYWORD",
            "mapping_confidence": 0.40,
            "manual_review_required": False,
            "notes": "Pain/hard token appears in reading passage; excluded from emergency intent."
        }

    # 4. Default: Standard non-emergency speech
    return {
        "intent_candidate": "GENERAL_SPEECH",
        "mapping_status": "UNMAPPED",
        "mapping_rule": "STANDARD_SPEECH_VOCABULARY",
        "mapping_confidence": 1.0,
        "manual_review_required": False,
        "notes": "Standard reading vocabulary without emergency semantics."
    }


def map_torgo_intents(
    config_path: str = "configs/torgo_config.yaml"
) -> pd.DataFrame:
    """Map all TORGO records to the intent taxonomy and write mapping CSV."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parquet_glob = config["dataset"]["parquet_glob"]
    taxonomy_path = config["paths"]["intent_taxonomy"]
    out_csv = config["paths"]["intent_mapping"]

    taxonomy = load_taxonomy(taxonomy_path)

    parquet_files = sorted(glob.glob(parquet_glob))
    records = []

    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=["audio", "transcription"])
        for idx, row in df.iterrows():
            orig_path = row["audio"]["path"]
            filename = os.path.basename(orig_path.replace("\\", "/"))
            rec_id = filename.rsplit(".", 1)[0]
            transcript = str(row.get("transcription", "")).strip()

            clf = classify_utterance_intent(transcript, taxonomy)
            records.append({
                "recording_id": rec_id,
                "transcript": transcript,
                "intent_candidate": clf["intent_candidate"],
                "mapping_status": clf["mapping_status"],
                "mapping_rule": clf["mapping_rule"],
                "mapping_confidence": clf["mapping_confidence"],
                "manual_review_required": clf["manual_review_required"],
                "notes": clf["notes"]
            })

    mapping_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    mapping_df.to_csv(out_csv, index=False)
    print(f"Saved intent mapping: {out_csv} ({len(mapping_df)} rows)")
    print("\nMapping Status Breakdown:")
    print(mapping_df["mapping_status"].value_counts())
    print("\nValid Intent Breakdown:")
    print(mapping_df[mapping_df["mapping_status"] == "VALID"]["intent_candidate"].value_counts())
    return mapping_df


if __name__ == "__main__":
    map_torgo_intents()
