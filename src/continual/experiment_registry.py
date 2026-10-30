"""
Phase 7.4 Continual Experiment Registry.
Maintains structured, auditable tracking of all experimental runs:
- experiment_id
- phase = 7.4
- model_family = classical_adapter
- speaker_id
- experience_budget
- train_experience_ids
- evaluation_set
- seed
- configuration
- status (PENDING, RUNNING, COMPLETED, FAILED, NOT_FEASIBLE)
- metrics summary
- output artifact locations
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Union


class ExperimentStatus(str, Enum):
    """Execution status for an experiment run."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NOT_FEASIBLE = "NOT_FEASIBLE"


from dataclasses import dataclass, field, asdict, fields

@dataclass
class ExperimentRunRecord:
    """Canonical experiment metadata record matching Section 26 and Section 39."""
    experiment_id: str
    speaker_id: str
    experience_budget: int
    train_experience_ids: List[str]
    evaluation_set: str
    seed: int
    configuration: Dict[str, Any]
    phase: str = "7.4"
    model_family: str = "classical_adapter"
    model_version: Optional[str] = None
    parent_model_version: Optional[str] = None
    status: str = ExperimentStatus.PENDING.value
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    output_files: Dict[str, str] = field(default_factory=dict)
    # Phase 7.7 drift & recalibration experiment fields
    reference_window_id: Optional[str] = None
    current_window_id: Optional[str] = None
    drift_policy_version: Optional[str] = None
    calibration_policy_version: Optional[str] = None
    calibration_method: Optional[str] = None
    calibration_data_ids: List[str] = field(default_factory=list)
    evaluation_data_ids: List[str] = field(default_factory=list)
    # Phase 7.8 continual quantum fields
    matched_control_experiment_id: Optional[str] = None
    replay_ids: List[str] = field(default_factory=list)
    feature_version: Optional[str] = None
    pca_scaling_version: Optional[str] = None
    quantum_configuration_version: Optional[str] = None
    kernel_configuration: Dict[str, Any] = field(default_factory=dict)
    simulator_configuration: Dict[str, Any] = field(default_factory=dict)
    validation_set: Optional[str] = None
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)
    governance_decision: Optional[str] = None
    # Phase 7.9 trainable quantum fields
    matched_fixed_quantum_experiment_id: Optional[str] = None
    quantum_architecture_version: Optional[str] = None
    parameter_count: int = 0
    circuit_depth: int = 0
    trainable_parameter_count: int = 0
    optimizer: Optional[str] = None
    training_configuration: Dict[str, Any] = field(default_factory=dict)
    parent_parameter_fingerprint: Optional[str] = None
    parameter_fingerprint: Optional[str] = None
    initialization_mode: Optional[str] = None
    # Phase 7.10 hybrid classical + quantum fields
    matched_trainable_quantum_experiment_id: Optional[str] = None
    fusion_method: Optional[str] = None
    fusion_weight: Optional[float] = None
    classical_branch_version: Optional[str] = None
    quantum_branch_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRunRecord":
        valid_fields = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**clean)


class ExperimentRegistry:
    """
    Manages experiment run tracking, persistence, and audit validation.
    """

    def __init__(self, registry_dir: Optional[str] = None):
        self.registry_dir = registry_dir
        self._runs: Dict[str, ExperimentRunRecord] = {}

        if self.registry_dir:
            os.makedirs(self.registry_dir, exist_ok=True)
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load existing experiment records from disk."""
        if not self.registry_dir:
            return
        index_file = os.path.join(self.registry_dir, "experiment_registry.json")
        if os.path.isfile(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                raw_list = json.load(f)
            for r in raw_list:
                record = ExperimentRunRecord.from_dict(r)
                self._runs[record.experiment_id] = record

    def _save_to_disk(self) -> None:
        """Persist registry atomically."""
        if not self.registry_dir:
            return
        index_file = os.path.join(self.registry_dir, "experiment_registry.json")
        data_list = [r.to_dict() for r in self._runs.values()]
        temp_fd, temp_path = tempfile.mkstemp(dir=self.registry_dir, suffix=".tmp")
        try:
            with open(temp_fd, "w", encoding="utf-8") as f:
                json.dump(data_list, f, indent=2)
            shutil.move(temp_path, index_file)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def register_run(self, record: ExperimentRunRecord) -> ExperimentRunRecord:
        """Register a new experiment run."""
        self._runs[record.experiment_id] = record
        if self.registry_dir:
            self._save_to_disk()
        return record

    def update_run(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        metrics: Optional[Dict[str, Any]] = None,
        model_version: Optional[str] = None,
        output_files: Optional[Dict[str, str]] = None,
        error_message: Optional[str] = None,
    ) -> ExperimentRunRecord:
        """Update existing experiment run record."""
        if experiment_id not in self._runs:
            raise KeyError(f"Experiment '{experiment_id}' not found in registry")

        rec = self._runs[experiment_id]
        rec.status = status.value
        rec.completed_at = datetime.now(timezone.utc).isoformat()
        if metrics is not None:
            rec.metrics = metrics
        if model_version is not None:
            rec.model_version = model_version
        if output_files is not None:
            rec.output_files.update(output_files)
        if error_message is not None:
            rec.error_message = error_message

        if self.registry_dir:
            self._save_to_disk()
        return rec

    def get_run(self, experiment_id: str) -> Optional[ExperimentRunRecord]:
        return self._runs.get(experiment_id)

    def list_runs(
        self,
        speaker_id: Optional[str] = None,
        budget: Optional[int] = None,
    ) -> List[ExperimentRunRecord]:
        runs = list(self._runs.values())
        if speaker_id:
            runs = [r for r in runs if r.speaker_id == speaker_id]
        if budget is not None:
            runs = [r for r in runs if r.experience_budget == budget]
        return runs

    # Convenience aliases
    register_experiment = register_run

    def update_experiment(self, record_or_id: Union[ExperimentRunRecord, str], **kwargs) -> ExperimentRunRecord:
        if isinstance(record_or_id, ExperimentRunRecord):
            self._runs[record_or_id.experiment_id] = record_or_id
            if self.registry_dir:
                self._save_to_disk()
            return record_or_id
        return self.update_run(record_or_id, **kwargs)


