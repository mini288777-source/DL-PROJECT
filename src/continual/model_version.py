"""
Phase 7 Model Versioning, Provenance & Lineage Management.
Implements:
- ModelLifecycleState (CANDIDATE, VALIDATED, ACTIVE, REJECTED, ROLLED_BACK, ARCHIVED).
- ModelVersionRecord tracking lineage, parent version, dataset fingerprint, and hyperparameter hash.
- Strict immutability: models and metadata cannot be overwritten in-place.
- ModelRegistry: manages candidate registration, version lookups, and promotion constraints.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class ModelLifecycleState(str, Enum):
    """Lifecycle states matching PHASE7_MODEL_VERSION_CONTRACT.md."""
    CANDIDATE = "CANDIDATE"
    VALIDATED = "VALIDATED"
    ACTIVE = "ACTIVE"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"
    ARCHIVED = "ARCHIVED"


class ModelVersionError(Exception):
    """Raised on invalid model version operations."""
    pass


class ModelOverwriteError(ModelVersionError):
    """Raised when an attempt is made to overwrite an existing model version."""
    pass


class ModelLineageError(ModelVersionError):
    """Raised when parent model lineage is violated."""
    pass


def compute_sha256(data: Union[str, bytes]) -> str:
    """Compute standard SHA-256 hex digest."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def compute_dataset_hash(X: np.ndarray, y: List[str], sample_ids: Optional[List[str]] = None) -> str:
    """
    Deterministic cryptographic hash of training dataset.
    Hashes feature array bytes, label list, and optional sample IDs.
    """
    hasher = hashlib.sha256()
    hasher.update(X.tobytes())
    hasher.update(",".join(y).encode("utf-8"))
    if sample_ids:
        hasher.update(",".join(sample_ids).encode("utf-8"))
    return hasher.hexdigest()


def compute_configuration_hash(config: Dict[str, Any]) -> str:
    """Deterministic hash of model and training hyperparameters."""
    sorted_json = json.dumps(config, sort_keys=True)
    return compute_sha256(sorted_json)


@dataclass
class ModelVersionRecord:
    """
    Canonical representation of a versioned personalization model.
    Complies with PHASE7_MODEL_VERSION_CONTRACT.md.
    """
    model_id: str                          # e.g., "classical_adapter"
    version: str                           # e.g., "vF01_iter1_0"
    speaker_id: str                        # e.g., "F01"
    parent_version: Optional[str] = None   # Version of predecessor model
    training_sample_ids: List[str] = field(default_factory=list)
    training_dataset_hash: str = ""        # SHA-256 of training data
    configuration_hash: str = ""           # SHA-256 of hyperparameters
    representation_version: str = "whisper_tiny_384_unit_L2"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    quantum_configuration: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = ModelLifecycleState.CANDIDATE.value
    artifact_path: Optional[str] = None    # Path to serialized model checkpoint

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersionRecord":
        return cls(**data)


class ModelRegistry:
    """
    Storage and lifecycle governance registry for versioned models.
    Supports in-memory operations and persistent file-backed indexing.
    Enforces candidate-first status and strict immutability.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir
        self._records: Dict[str, ModelVersionRecord] = {}  # version -> ModelVersionRecord
        self._by_speaker: Dict[str, List[str]] = {}        # speaker_id -> [versions]
        self._active_by_speaker: Dict[str, str] = {}       # speaker_id -> active version

        if self.base_dir:
            os.makedirs(self.base_dir, exist_ok=True)
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load registered versions from disk registry index."""
        if not self.base_dir:
            return
        index_file = os.path.join(self.base_dir, "model_registry_index.json")
        if os.path.isfile(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            for v_data in raw_data:
                record = ModelVersionRecord.from_dict(v_data)
                self._index_record(record)

    def _save_to_disk(self) -> None:
        """Persist registry index atomically."""
        if not self.base_dir:
            return
        index_file = os.path.join(self.base_dir, "model_registry_index.json")
        records_list = [r.to_dict() for r in self._records.values()]
        temp_fd, temp_path = tempfile.mkstemp(dir=self.base_dir, suffix=".tmp")
        try:
            with open(temp_fd, "w", encoding="utf-8") as f:
                json.dump(records_list, f, indent=2)
            shutil.move(temp_path, index_file)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _index_record(self, record: ModelVersionRecord) -> None:
        """Internal index updater."""
        self._records[record.version] = record
        if record.speaker_id not in self._by_speaker:
            self._by_speaker[record.speaker_id] = []
        if record.version not in self._by_speaker[record.speaker_id]:
            self._by_speaker[record.speaker_id].append(record.version)
        if record.status == ModelLifecycleState.ACTIVE.value:
            self._active_by_speaker[record.speaker_id] = record.version

    def register_candidate(self, record: ModelVersionRecord) -> ModelVersionRecord:
        """
        Register a new candidate model version.
        Enforces:
        - Version uniqueness (no overwrite)
        - Parent existence (if specified)
        - Initial status MUST be CANDIDATE
        """
        if record.version in self._records:
            raise ModelOverwriteError(
                f"Model version '{record.version}' already exists. Overwrites are strictly prohibited."
            )

        if record.parent_version:
            if record.parent_version not in self._records:
                raise ModelLineageError(
                    f"Parent version '{record.parent_version}' does not exist in registry."
                )

        # Invariant: newly created model must enter as CANDIDATE
        record.status = ModelLifecycleState.CANDIDATE.value

        self._index_record(record)
        if self.base_dir:
            self._save_to_disk()
        return record

    def get_version(self, version: str) -> Optional[ModelVersionRecord]:
        """Retrieve model version record."""
        return self._records.get(version)

    def list_by_speaker(self, speaker_id: str) -> List[ModelVersionRecord]:
        """List all model versions for a given speaker."""
        versions = self._by_speaker.get(speaker_id, [])
        return [self._records[v] for v in versions]

    def get_active_model(self, speaker_id: str) -> Optional[ModelVersionRecord]:
        """Retrieve currently active model for speaker, if any."""
        active_version = self._active_by_speaker.get(speaker_id)
        if active_version:
            return self._records.get(active_version)
        return None

    def update_status(self, version: str, new_status: ModelLifecycleState) -> ModelVersionRecord:
        """
        Transition model status.
        Enforces single-active per speaker rule.
        """
        if version not in self._records:
            raise ModelVersionError(f"Cannot update status of non-existent version '{version}'")

        record = self._records[version]
        spk = record.speaker_id

        if new_status == ModelLifecycleState.ACTIVE:
            # If promoting to ACTIVE, demote current active to ARCHIVED
            curr_active = self._active_by_speaker.get(spk)
            if curr_active and curr_active != version:
                old_active = self._records[curr_active]
                old_active.status = ModelLifecycleState.ARCHIVED.value
            self._active_by_speaker[spk] = version
        else:
            if self._active_by_speaker.get(spk) == version:
                del self._active_by_speaker[spk]

        record.status = new_status.value
        if self.base_dir:
            self._save_to_disk()
        return record
