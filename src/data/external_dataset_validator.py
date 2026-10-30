"""
External Dataset Validator for Gate C / Gate D Readiness.

Provides schema validation, speaker overlap detection against TORGO,
data provenance and DUA verification, and partition separation checks
without downloading, copying, or processing restricted data without permission.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import csv
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

TORGO_SPEAKERS: Set[str] = {
    "F01", "F03", "F04", "FC01", "FC02", "FC03",
    "M01", "M02", "M03", "M04", "M05", "MC01",
    "MC02", "MC03", "MC04"
}

REQUIRED_MANIFEST_COLUMNS: List[str] = [
    "utterance_id",
    "speaker_id",
    "audio_path",
    "transcript",
    "is_emergency",
    "split"
]

RECOMMENDED_METADATA_COLUMNS: List[str] = [
    "dysarthria_severity",
    "etiology",
    "gender",
    "recording_device",
    "acoustic_environment"
]

SYNTHETIC_PATH_KEYWORDS: List[str] = [
    "synthetic",
    "sine_wave",
    "sine",
    "mock_audio",
    "mock",
    "test_fixture",
    "tone_generator",
    "generated_tone",
    "synthesized",
    "generated",
    "artificial",
    "simulated",
    "tts",
    "text_to_speech",
    "text-to-speech",
    "fastspeech",
    "vits",
    "elevenlabs",
    "vocoder",
]


@dataclass
class ValidationReport:
    """Detailed outcome of external dataset validation."""
    is_valid: bool
    cohort_name: str
    manifest_sha256: str = ""
    total_records: int = 0
    unique_speakers: int = 0
    splits_found: Dict[str, int] = field(default_factory=dict)
    emergency_count: int = 0
    non_emergency_count: int = 0
    contains_synthetic_records: bool = False
    torgo_overlap_speakers: List[str] = field(default_factory=list)
    torgo_utterance_overlap: List[str] = field(default_factory=list)
    missing_required_columns: List[str] = field(default_factory=list)
    missing_provenance_fields: List[str] = field(default_factory=list)
    ethics_approval_status: str = "NOT_PROVIDED"
    dua_status: str = "NOT_PROVIDED"
    partition_leakage_detected: bool = False
    unverifiable_checks: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "cohort_name": self.cohort_name,
            "manifest_sha256": self.manifest_sha256,
            "total_records": self.total_records,
            "unique_speakers": self.unique_speakers,
            "splits_found": self.splits_found,
            "emergency_count": self.emergency_count,
            "non_emergency_count": self.non_emergency_count,
            "contains_synthetic_records": self.contains_synthetic_records,
            "torgo_overlap_speakers": self.torgo_overlap_speakers,
            "torgo_utterance_overlap": self.torgo_utterance_overlap,
            "missing_required_columns": self.missing_required_columns,
            "missing_provenance_fields": self.missing_provenance_fields,
            "ethics_approval_status": self.ethics_approval_status,
            "dua_status": self.dua_status,
            "partition_leakage_detected": self.partition_leakage_detected,
            "unverifiable_checks": self.unverifiable_checks,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class ExternalDatasetValidator:
    """
    Validates external atypical speech datasets for confirmatory evaluation.
    Enforces strict independence from TORGO and verifies provenance metadata.
    """

    def __init__(
        self,
        torgo_speakers: Optional[Set[str]] = None,
        torgo_manifest_path: Optional[Path] = None,
        allow_synthetic: bool = False
    ):
        self.torgo_speakers = {s.upper().strip() for s in (torgo_speakers or TORGO_SPEAKERS)}
        self.allow_synthetic = allow_synthetic
        self.torgo_utterance_ids: Set[str] = self._load_torgo_utterances(torgo_manifest_path)

    def _load_torgo_utterances(self, torgo_manifest_path: Optional[Path]) -> Set[str]:
        """Loads known TORGO utterance IDs for collision detection (case-insensitive)."""
        utterance_ids: Set[str] = set()
        default_path = Path("data/manifests/torgo_manifest.csv")
        manifest_p = torgo_manifest_path or default_path
        if manifest_p.exists():
            try:
                with open(manifest_p, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        u_id = row.get("utterance_id") or row.get("session_id")
                        if u_id:
                            utterance_ids.add(u_id.strip().upper())
            except Exception as e:
                logger.warning(f"Could not read TORGO manifest for utterance collision check: {e}")
        return utterance_ids

    def validate_manifest(
        self,
        manifest_path: Path,
        cohort_metadata_path: Optional[Path] = None,
        cohort_name: str = "EXTERNAL_COHORT"
    ) -> ValidationReport:
        """
        Validate an external dataset manifest and companion metadata.
        """
        manifest_path = Path(manifest_path)
        report = ValidationReport(is_valid=True, cohort_name=cohort_name)

        # Document checks that cannot be performed without raw audio or external registries
        report.unverifiable_checks = [
            "Acoustic biometric/perceptual voice hash overlap cannot be checked without legally authorized raw audio access.",
            "Cross-institution participant identity linkage cannot be checked without prospective clinical registry linkage.",
            "Phonetic transcription accuracy cannot be checked without independent double-annotation audit."
        ]

        if not manifest_path.exists():
            report.is_valid = False
            report.errors.append(f"Manifest file not found: {manifest_path}")
            return report

        # Compute SHA-256 hash of manifest
        try:
            with open(manifest_path, "rb") as f:
                report.manifest_sha256 = hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            report.is_valid = False
            report.errors.append(f"Failed to compute manifest SHA-256: {e}")
            return report

        # 1. Provenance metadata check
        provenance_valid = self._check_provenance(cohort_metadata_path, report)
        if not provenance_valid:
            report.is_valid = False

        # 2. Manifest parsing & schema validation
        try:
            with open(manifest_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames or []

                missing_cols = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in header]
                if missing_cols:
                    report.missing_required_columns = missing_cols
                    report.errors.append(f"Missing required manifest columns: {missing_cols}")
                    report.is_valid = False

                speakers: Set[str] = set()
                split_counts: Dict[str, int] = {}
                split_speakers: Dict[str, Set[str]] = {}
                utterance_ids: Set[str] = set()
                torgo_utterance_collisions: List[str] = []
                synthetic_record_count = 0
                emergency_n = 0
                non_emergency_n = 0
                record_count = 0

                for row_idx, row in enumerate(reader):
                    record_count += 1
                    u_id = (row.get("utterance_id") or "").strip()
                    if u_id:
                        utterance_ids.add(u_id)
                        if u_id.upper() in self.torgo_utterance_ids:
                            torgo_utterance_collisions.append(u_id)

                    spk = (row.get("speaker_id") or "").strip()
                    if spk:
                        speakers.add(spk)

                    audio_p_raw = (row.get("audio_path") or "").strip()
                    if not audio_p_raw:
                        report.errors.append(f"Row {record_count}: missing or empty audio_path.")
                        report.is_valid = False

                    # Check for synthetic indicators across flags, paths, and metadata columns
                    synth_col_keys = {
                        "is_synthetic", "synthetic", "is_mock", "is_simulated", "mock",
                        "simulated", "audio_type", "recording_type", "speech_type", "data_source",
                        "data_nature", "data_type", "generation_type", "is_generated", "signal_type",
                        "audio_category", "is_artificial", "artificial", "speech_category",
                        "audio_origin", "origin", "synthesis_method"
                    }
                    synth_val_indicators = {
                        "true", "1", "yes", "synthetic", "mock", "simulated", "tone", "sine_wave",
                        "sine", "synthesized", "generated", "artificial", "tts", "text-to-speech",
                        "text_to_speech", "vits", "fastspeech", "elevenlabs", "vocoder", "in_silico",
                        "model_generated"
                    }
                    has_synth_field = False
                    for col_k, col_v in row.items():
                        k_clean = str(col_k).strip().lower()
                        v_clean = str(col_v).strip().lower()
                        if k_clean in synth_col_keys and v_clean in synth_val_indicators:
                            has_synth_field = True
                            break
                        if any(kw in v_clean for kw in SYNTHETIC_PATH_KEYWORDS) and k_clean in ("audio_path", "filename", "filepath", "source"):
                            has_synth_field = True
                            break

                    audio_p = audio_p_raw.lower()
                    has_synth_path = any(k in audio_p for k in SYNTHETIC_PATH_KEYWORDS)
                    if has_synth_field or has_synth_path:
                        synthetic_record_count += 1

                    split = (row.get("split") or "unspecified").strip().lower()
                    split_counts[split] = split_counts.get(split, 0) + 1
                    if split not in split_speakers:
                        split_speakers[split] = set()
                    if spk:
                        split_speakers[split].add(spk)

                    em_val = (row.get("is_emergency") or "").strip().lower()
                    if em_val in {"1", "true", "yes", "emergency"}:
                        emergency_n += 1
                    else:
                        non_emergency_n += 1

                report.total_records = record_count
                report.unique_speakers = len(speakers)
                report.splits_found = split_counts
                report.emergency_count = emergency_n
                report.non_emergency_count = non_emergency_n

                # 3. Check for synthetic fixtures
                if synthetic_record_count > 0:
                    report.contains_synthetic_records = True
                    if not self.allow_synthetic:
                        report.errors.append(
                            f"CONFIRMATORY_DATA_DEFECT: Manifest contains {synthetic_record_count} synthetic "
                            "or mock records. Synthetic audio cannot serve as clinical confirmatory evidence."
                        )
                        report.is_valid = False
                    else:
                        report.warnings.append(
                            f"Notice: Manifest contains {synthetic_record_count} synthetic records (permitted in testing mode)."
                        )

                # 4. Check TORGO speaker overlap (case-insensitive)
                speakers_upper = {s.upper() for s in speakers}
                overlap = sorted(list(speakers_upper.intersection(self.torgo_speakers)))
                if overlap:
                    report.torgo_overlap_speakers = overlap
                    report.errors.append(
                        f"FATAL: TORGO speaker overlap detected ({len(overlap)} speakers: {overlap}). "
                        "Confirmatory cohort MUST be completely independent."
                    )
                    report.is_valid = False

                # 5. Check TORGO utterance collisions
                if torgo_utterance_collisions:
                    report.torgo_utterance_overlap = torgo_utterance_collisions[:10]
                    report.errors.append(
                        f"FATAL: TORGO utterance ID collision detected ({len(torgo_utterance_collisions)} collisions)."
                    )
                    report.is_valid = False

                # 6. Check partition leakage between development/calibration and confirmatory/test
                dev_speakers = split_speakers.get("dev", set()) | split_speakers.get("calibration", set()) | split_speakers.get("train", set())
                test_speakers = split_speakers.get("test", set()) | split_speakers.get("confirmatory", set())
                leakage = dev_speakers.intersection(test_speakers)
                if leakage:
                    report.partition_leakage_detected = True
                    report.errors.append(
                        f"FATAL: Speaker leakage between dev/calibration and test/confirmatory: {sorted(list(leakage))}"
                    )
                    report.is_valid = False

                if record_count == 0:
                    report.is_valid = False
                    report.errors.append("Manifest is empty (0 records).")

        except Exception as e:
            report.is_valid = False
            report.errors.append(f"Failed to read/parse manifest: {str(e)}")

        return report

    def _check_provenance(self, meta_path: Optional[Path], report: ValidationReport) -> bool:
        """Verify institutional ethics, consent, and license permissions with status enforcement."""
        if not meta_path or not Path(meta_path).exists():
            report.missing_provenance_fields = [
                "ethics_approval_id", "ethics_approval_status", "data_use_agreement_id",
                "dua_status", "license", "pseudonymization_certified", "institution"
            ]
            report.ethics_approval_status = "NOT_PROVIDED"
            report.dua_status = "NOT_PROVIDED"
            report.errors.append("Provenance metadata file missing or not provided.")
            return False

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            report.metadata = data
            required_fields = [
                "ethics_approval_id",
                "ethics_approval_status",
                "data_use_agreement_id",
                "dua_status",
                "license",
                "pseudonymization_certified",
                "institution"
            ]
            missing = [f for f in required_fields if f not in data or data.get(f) is None or str(data.get(f)).strip() == ""]
            if missing:
                report.missing_provenance_fields = missing
                report.errors.append(f"Incomplete provenance metadata; missing fields: {missing}")
                return False

            ethics_st = str(data.get("ethics_approval_status", "")).strip().upper()
            report.ethics_approval_status = ethics_st
            if ethics_st not in ("APPROVED", "EXEMPT_CERTIFIED"):
                report.errors.append(
                    f"ETHICS_STATUS_UNMET: Ethics approval status is '{ethics_st}'. "
                    "Must be 'APPROVED' or 'EXEMPT_CERTIFIED' by accredited IRB/IEC."
                )
                return False

            dua_st = str(data.get("dua_status", "")).strip().upper()
            report.dua_status = dua_st
            if dua_st not in ("EXECUTED", "ACTIVE"):
                report.errors.append(
                    f"DUA_STATUS_UNMET: Data Use Agreement status is '{dua_st}'. "
                    "Must be 'EXECUTED' or 'ACTIVE' before data may be used."
                )
                return False

            if not data.get("pseudonymization_certified"):
                report.errors.append("Dataset has not certified de-identification / pseudonymization.")
                return False

            # Check for synthetic indicators in cohort metadata
            meta_synth_keys = {"is_synthetic", "synthetic", "contains_synthetic_records", "is_mock", "is_simulated"}
            meta_indicates_synthetic = False
            for sk in meta_synth_keys:
                if sk in data and data.get(sk) is True:
                    meta_indicates_synthetic = True
                    break
            if str(data.get("data_type", "")).strip().lower() in ("synthetic", "mock", "simulated"):
                meta_indicates_synthetic = True
            if str(data.get("data_nature", "")).strip().lower() in ("synthetic", "mock", "simulated", "artificial"):
                meta_indicates_synthetic = True

            if meta_indicates_synthetic:
                report.contains_synthetic_records = True
                if not self.allow_synthetic:
                    report.errors.append(
                        "CONFIRMATORY_DATA_DEFECT: Metadata explicitly identifies dataset as synthetic/simulated. "
                        "Synthetic audio cannot serve as clinical confirmatory evidence."
                    )
                    return False

            return True

        except Exception as e:
            report.errors.append(f"Error parsing provenance metadata: {str(e)}")
            return False
