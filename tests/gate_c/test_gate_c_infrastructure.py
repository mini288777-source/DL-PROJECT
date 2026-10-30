"""
Tests for Gate C Infrastructure: External Dataset Validator and Confirmatory Runner.
"""

from pathlib import Path
import csv
import json
import pytest

from src.data.external_dataset_validator import ExternalDatasetValidator
from src.evaluation.confirmatory_runner import (
    ConfirmatoryRunner,
    ConfirmatoryRunConfig,
    ConfirmatoryPrerequisiteError
)


@pytest.fixture
def temp_dataset_dir(tmp_path: Path):
    """Create a temporary directory with mock valid external dataset manifest & metadata."""
    meta_path = tmp_path / "metadata.json"
    metadata = {
        "institution": "Independent Research Hospital",
        "ethics_approval_id": "IRB-2026-TEST-999",
        "ethics_approval_status": "APPROVED",
        "data_use_agreement_id": "DUA-TEST-001",
        "dua_status": "EXECUTED",
        "license": "CC-BY-NC-4.0",
        "pseudonymization_certified": True
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    manifest_path = tmp_path / "manifest.csv"
    rows = [
        {"utterance_id": "U01", "speaker_id": "EXT_SPK_01", "audio_path": "speech1.wav", "transcript": "help", "is_emergency": "1", "split": "dev"},
        {"utterance_id": "U02", "speaker_id": "EXT_SPK_01", "audio_path": "speech2.wav", "transcript": "hello", "is_emergency": "0", "split": "dev"},
        {"utterance_id": "U03", "speaker_id": "EXT_SPK_02", "audio_path": "speech3.wav", "transcript": "fire", "is_emergency": "1", "split": "confirmatory"},
        {"utterance_id": "U04", "speaker_id": "EXT_SPK_02", "audio_path": "speech4.wav", "transcript": "water", "is_emergency": "0", "split": "confirmatory"},
    ]
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return tmp_path, manifest_path, meta_path


def test_validator_valid_dataset(temp_dataset_dir):
    _, manifest_path, meta_path = temp_dataset_dir
    validator = ExternalDatasetValidator()
    report = validator.validate_manifest(manifest_path, meta_path, cohort_name="TEST_COHORT")

    assert report.is_valid is True
    assert report.total_records == 4
    assert report.unique_speakers == 2
    assert report.emergency_count == 2
    assert report.non_emergency_count == 2
    assert len(report.torgo_overlap_speakers) == 0
    assert report.partition_leakage_detected is False
    assert len(report.manifest_sha256) == 64
    assert report.contains_synthetic_records is False


def test_validator_detects_synthetic_fixture(temp_dataset_dir, tmp_path):
    _, _, meta_path = temp_dataset_dir
    manifest_path = tmp_path / "synth_manifest.csv"
    rows = [
        {"utterance_id": "U01", "speaker_id": "EXT_SPK_01", "audio_path": "test_sine_wave_440hz.wav", "transcript": "help", "is_emergency": "1", "split": "confirmatory"},
    ]
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Disallowing synthetic (default for confirmatory runs)
    validator = ExternalDatasetValidator(allow_synthetic=False)
    report = validator.validate_manifest(manifest_path, meta_path)

    assert report.is_valid is False
    assert report.contains_synthetic_records is True
    assert any("Synthetic audio cannot serve as clinical confirmatory evidence" in err for err in report.errors)


def test_validator_detects_pending_ethics(temp_dataset_dir, tmp_path):
    _, manifest_path, _ = temp_dataset_dir
    meta_path = tmp_path / "pending_meta.json"
    metadata = {
        "institution": "Hospital",
        "ethics_approval_id": "IRB-DRAFT-001",
        "ethics_approval_status": "PENDING",  # Not APPROVED
        "data_use_agreement_id": "DUA-001",
        "dua_status": "EXECUTED",
        "license": "CC-BY",
        "pseudonymization_certified": True
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    validator = ExternalDatasetValidator()
    report = validator.validate_manifest(manifest_path, meta_path)

    assert report.is_valid is False
    assert any("ETHICS_STATUS_UNMET" in err for err in report.errors)


def test_validator_detects_pending_dua(temp_dataset_dir, tmp_path):
    _, manifest_path, _ = temp_dataset_dir
    meta_path = tmp_path / "pending_dua_meta.json"
    metadata = {
        "institution": "Hospital",
        "ethics_approval_id": "IRB-001",
        "ethics_approval_status": "APPROVED",
        "data_use_agreement_id": "DUA-DRAFT-001",
        "dua_status": "DRAFT",  # Not EXECUTED
        "license": "CC-BY",
        "pseudonymization_certified": True
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    validator = ExternalDatasetValidator()
    report = validator.validate_manifest(manifest_path, meta_path)

    assert report.is_valid is False
    assert any("DUA_STATUS_UNMET" in err for err in report.errors)


def test_validator_detects_torgo_overlap(temp_dataset_dir):
    tmp_path, _, meta_path = temp_dataset_dir
    manifest_path = tmp_path / "overlap_manifest.csv"

    # Injects TORGO speaker F01
    rows = [
        {"utterance_id": "U01", "speaker_id": "F01", "audio_path": "a1.wav", "transcript": "help", "is_emergency": "1", "split": "confirmatory"},
    ]
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    validator = ExternalDatasetValidator()
    report = validator.validate_manifest(manifest_path, meta_path)

    assert report.is_valid is False
    assert "F01" in report.torgo_overlap_speakers
    assert any("TORGO speaker overlap detected" in err for err in report.errors)


def test_validator_detects_partition_leakage(temp_dataset_dir):
    tmp_path, _, meta_path = temp_dataset_dir
    manifest_path = tmp_path / "leakage_manifest.csv"

    # Same speaker in dev and confirmatory
    rows = [
        {"utterance_id": "U01", "speaker_id": "EXT_SPK_01", "audio_path": "a1.wav", "transcript": "help", "is_emergency": "1", "split": "dev"},
        {"utterance_id": "U02", "speaker_id": "EXT_SPK_01", "audio_path": "a2.wav", "transcript": "help", "is_emergency": "1", "split": "confirmatory"},
    ]
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    validator = ExternalDatasetValidator()
    report = validator.validate_manifest(manifest_path, meta_path)

    assert report.is_valid is False
    assert report.partition_leakage_detected is True
    assert any("Speaker leakage" in err for err in report.errors)


def test_validator_detects_missing_provenance(temp_dataset_dir):
    _, manifest_path, _ = temp_dataset_dir
    validator = ExternalDatasetValidator()
    report = validator.validate_manifest(manifest_path, cohort_metadata_path=None)

    assert report.is_valid is False
    assert len(report.missing_provenance_fields) > 0


def test_confirmatory_runner_blocks_without_owner_authorization(temp_dataset_dir, tmp_path):
    _, manifest_path, meta_path = temp_dataset_dir
    proto_file = tmp_path / "protocol.yaml"
    proto_file.write_text("protocol_version: 1.0", encoding="utf-8")
    import hashlib
    expected_hash = hashlib.sha256(b"protocol_version: 1.0").hexdigest()

    ckpt_file = tmp_path / "model.pt"
    ckpt_file.write_bytes(b"mock_weights")

    config = ConfirmatoryRunConfig(
        protocol_path=proto_file,
        expected_protocol_sha256=expected_hash,
        manifest_path=manifest_path,
        cohort_metadata_path=meta_path,
        checkpoint_path=ckpt_file,
        owner_authorization_token=None  # Missing!
    )
    runner = ConfirmatoryRunner(config)
    audit = runner.audit_prerequisites()

    assert audit["status"] == "BLOCKED"
    assert any("MISSING_OWNER_AUTHORIZATION" in b for b in audit["blockers"])

    with pytest.raises(ConfirmatoryPrerequisiteError) as exc_info:
        runner.execute_confirmatory_run()
    assert "MISSING_OWNER_AUTHORIZATION" in str(exc_info.value)


def test_confirmatory_runner_blocks_dummy_or_short_token(temp_dataset_dir, tmp_path):
    _, manifest_path, meta_path = temp_dataset_dir
    proto_file = tmp_path / "protocol.yaml"
    proto_file.write_text("protocol_version: 1.0", encoding="utf-8")
    import hashlib
    expected_hash = hashlib.sha256(b"protocol_version: 1.0").hexdigest()

    ckpt_file = tmp_path / "model.pt"
    ckpt_file.write_bytes(b"mock_weights")

    # Placeholder / dummy token
    config = ConfirmatoryRunConfig(
        protocol_path=proto_file,
        expected_protocol_sha256=expected_hash,
        manifest_path=manifest_path,
        cohort_metadata_path=meta_path,
        checkpoint_path=ckpt_file,
        owner_authorization_token="mock_dummy_short"  # Invalid!
    )
    runner = ConfirmatoryRunner(config)
    audit = runner.audit_prerequisites()

    assert audit["status"] == "BLOCKED"
    assert any("MISSING_OWNER_AUTHORIZATION" in b for b in audit["blockers"])


def test_confirmatory_runner_blocks_unmaterialized_checkpoint(temp_dataset_dir, tmp_path):
    _, manifest_path, meta_path = temp_dataset_dir
    proto_file = tmp_path / "protocol.yaml"
    proto_file.write_text("protocol_version: 1.0", encoding="utf-8")
    import hashlib
    expected_hash = hashlib.sha256(b"protocol_version: 1.0").hexdigest()

    non_existent_ckpt = tmp_path / "unmaterialized_checkpoint.pt"

    config = ConfirmatoryRunConfig(
        protocol_path=proto_file,
        expected_protocol_sha256=expected_hash,
        manifest_path=manifest_path,
        cohort_metadata_path=meta_path,
        checkpoint_path=non_existent_ckpt,
        owner_authorization_token="SIG_RELEASE_AUTH_KEY_9837264819472649"
    )
    runner = ConfirmatoryRunner(config)
    audit = runner.audit_prerequisites()

    assert audit["status"] == "BLOCKED"
    assert any("CHECKPOINT_NOT_MATERIALIZED" in b for b in audit["blockers"])

    with pytest.raises(ConfirmatoryPrerequisiteError) as exc_info:
        runner.execute_confirmatory_run()
    assert "CHECKPOINT_NOT_MATERIALIZED" in str(exc_info.value)


def test_confirmatory_runner_passes_dry_run_when_all_satisfied(temp_dataset_dir, tmp_path):
    _, manifest_path, meta_path = temp_dataset_dir
    proto_file = tmp_path / "protocol.yaml"
    proto_file.write_text("protocol_version: 1.0", encoding="utf-8")
    import hashlib
    expected_hash = hashlib.sha256(b"protocol_version: 1.0").hexdigest()

    ckpt_file = tmp_path / "model.pt"
    ckpt_file.write_bytes(b"mock_weights")

    config = ConfirmatoryRunConfig(
        protocol_path=proto_file,
        expected_protocol_sha256=expected_hash,
        manifest_path=manifest_path,
        cohort_metadata_path=meta_path,
        checkpoint_path=ckpt_file,
        owner_authorization_token="SIG_RELEASE_AUTH_KEY_9837264819472649",
        dry_run_only=True
    )
    runner = ConfirmatoryRunner(config)
    res = runner.execute_confirmatory_run()

    assert res["execution_status"] == "AUDIT_PASSED_DRY_RUN_COMPLETED"
    assert res["audit"]["status"] == "READY"
    assert "REDACTED" in res["audit"]["checks"]["owner_authorization_redacted"]


def test_confirmatory_runner_enforces_single_pass_lock(temp_dataset_dir, tmp_path):
    _, manifest_path, meta_path = temp_dataset_dir
    proto_file = tmp_path / "protocol.yaml"
    proto_file.write_text("protocol_version: 1.0", encoding="utf-8")
    import hashlib
    expected_hash = hashlib.sha256(b"protocol_version: 1.0").hexdigest()

    ckpt_file = tmp_path / "model.pt"
    ckpt_file.write_bytes(b"mock_weights")

    config = ConfirmatoryRunConfig(
        protocol_path=proto_file,
        expected_protocol_sha256=expected_hash,
        manifest_path=manifest_path,
        cohort_metadata_path=meta_path,
        checkpoint_path=ckpt_file,
        owner_authorization_token="SIG_RELEASE_AUTH_KEY_9837264819472649",
        dry_run_only=False,  # Live run
        output_dir=tmp_path / "eval_out"
    )
    runner = ConfirmatoryRunner(config)
    # First execution passes and creates lock
    res1 = runner.execute_confirmatory_run()
    assert res1["execution_status"] == "COMPLETED"

    # Second execution must be blocked by single-pass lock
    runner2 = ConfirmatoryRunner(config)
    audit2 = runner2.audit_prerequisites()
    assert audit2["status"] == "BLOCKED"
    assert any("SINGLE_PASS_VIOLATION" in b for b in audit2["blockers"])
    with pytest.raises(ConfirmatoryPrerequisiteError) as exc_info:
        runner2.execute_confirmatory_run()
    assert "SINGLE_PASS_VIOLATION" in str(exc_info.value)


def test_validator_case_insensitive_torgo_overlap(temp_dataset_dir, tmp_path):
    """Verify that lowercase or untrimmed TORGO speaker IDs cannot bypass overlap checks."""
    _, _, meta_path = temp_dataset_dir
    manifest_path = tmp_path / "cased_manifest.csv"
    rows = [
        {"utterance_id": "U01", "speaker_id": "f01", "audio_path": "speech1.wav", "transcript": "help", "is_emergency": "1", "split": "confirmatory"},
    ]
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    validator = ExternalDatasetValidator()
    report = validator.validate_manifest(manifest_path, meta_path)

    assert report.is_valid is False
    assert "F01" in report.torgo_overlap_speakers or "f01" in report.torgo_overlap_speakers
    assert any("TORGO speaker overlap detected" in err for err in report.errors)


def test_validator_detects_alternative_synthetic_columns(temp_dataset_dir, tmp_path):
    """Verify that alternative schema column names (e.g. synthetic, audio_type) flag synthetic records."""
    _, _, meta_path = temp_dataset_dir
    manifest_path = tmp_path / "alt_synth_manifest.csv"
    rows = [
        {"utterance_id": "U01", "speaker_id": "EXT_01", "audio_path": "clean_human_sounding.wav", "transcript": "help", "is_emergency": "1", "split": "confirmatory", "audio_type": "synthetic"},
    ]
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    validator = ExternalDatasetValidator(allow_synthetic=False)
    report = validator.validate_manifest(manifest_path, meta_path)

    assert report.is_valid is False
    assert report.contains_synthetic_records is True
    assert any("Synthetic audio cannot serve as clinical confirmatory evidence" in err for err in report.errors)


def test_validator_detects_synthetic_in_metadata(temp_dataset_dir, tmp_path):
    """Verify that synthetic indicators in metadata.json trigger fail-closed rejection."""
    _, manifest_path, _ = temp_dataset_dir
    meta_path = tmp_path / "synthetic_meta.json"
    metadata = {
        "institution": "Hospital",
        "ethics_approval_id": "IRB-001",
        "ethics_approval_status": "APPROVED",
        "data_use_agreement_id": "DUA-001",
        "dua_status": "EXECUTED",
        "license": "CC-BY",
        "pseudonymization_certified": True,
        "is_synthetic": True  # Declares synthetic nature in metadata
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    validator = ExternalDatasetValidator(allow_synthetic=False)
    report = validator.validate_manifest(manifest_path, meta_path)

    assert report.is_valid is False
    assert report.contains_synthetic_records is True
    assert any("CONFIRMATORY_DATA_DEFECT" in err for err in report.errors)


def test_validator_detects_advanced_synthesis_keywords_and_columns(temp_dataset_dir, tmp_path):
    """Verify that TTS, ElevenLabs, VITS, or data_nature columns flag synthetic records."""
    _, _, meta_path = temp_dataset_dir
    manifest_path = tmp_path / "tts_manifest.csv"
    rows = [
        {"utterance_id": "U01", "speaker_id": "EXT_01", "audio_path": "sample_tts_vits.wav", "transcript": "help", "is_emergency": "1", "split": "confirmatory", "data_nature": "generated"},
    ]
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    validator = ExternalDatasetValidator(allow_synthetic=False)
    report = validator.validate_manifest(manifest_path, meta_path)

    assert report.is_valid is False
    assert report.contains_synthetic_records is True


def test_validator_rejects_empty_audio_path(temp_dataset_dir, tmp_path):
    """Verify that records with missing or whitespace-only audio_path are rejected."""
    _, _, meta_path = temp_dataset_dir
    manifest_path = tmp_path / "empty_path_manifest.csv"
    rows = [
        {"utterance_id": "U01", "speaker_id": "EXT_01", "audio_path": "   ", "transcript": "help", "is_emergency": "1", "split": "confirmatory"},
    ]
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    validator = ExternalDatasetValidator(allow_synthetic=False)
    report = validator.validate_manifest(manifest_path, meta_path)

    assert report.is_valid is False
    assert any("missing or empty audio_path" in err for err in report.errors)


