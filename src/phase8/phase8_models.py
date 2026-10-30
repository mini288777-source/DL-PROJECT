"""
Phase 8 Unified Model Architectures: Tracks A, B, and C.

Implements the 3-track controlled recovery ladder:
- Track A: Hierarchical Emergency-Aware Classical Baseline (Stage 1 Binary Gate + Stage 2 384-D Subtype Net)
- Track B: Quantum Residual Emergency Subtype Module (Stage 1 Binary Gate + Train-Only PCA 8-D + 8-Qubit VQC)
- Track C: Matched Classical Shadow Model (Stage 1 Binary Gate + Train-Only PCA 8-D + Matched 8-D MLP)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score

from src.continual.taxonomy import CANONICAL_CLASSES, EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS
from src.quantum.pca_pipeline import TrainOnlyPCAPipeline
from src.phase8.hierarchical_gate import BinaryEmergencyDetector
from src.phase8.classical_subtype import ClassicalSubtypeClassifier
from src.phase8.quantum_residual import QuantumResidualSubtypeClassifier
from src.phase8.classical_shadow import ClassicalShadowClassifier


class TrackAHierarchicalClassical:
    """
    Track A: Strengthened Hierarchical Classical Baseline.
    Decouples binary triage from 6-class emergency subtype classification.
    """
    def __init__(
        self,
        pos_weight: float = 3.0,
        gate_lr: float = 0.005,
        gate_epochs: int = 50,
        subtype_lr: float = 0.005,
        subtype_epochs: int = 50,
        threshold: float = 0.5,
        random_seed: int = 42,
    ):
        self.pos_weight = pos_weight
        self.gate_lr = gate_lr
        self.gate_epochs = gate_epochs
        self.subtype_lr = subtype_lr
        self.subtype_epochs = subtype_epochs
        self.threshold = threshold
        self.random_seed = random_seed

        self.gate = BinaryEmergencyDetector(
            input_dim=384,
            hidden_dim=64,
            lr=self.gate_lr,
            pos_weight=self.pos_weight,
            epochs=self.gate_epochs,
            random_seed=self.random_seed,
        )
        self.subtype = ClassicalSubtypeClassifier(
            input_dim=384,
            hidden_dim=64,
            num_classes=len(EMERGENCY_CLASSES),
            lr=self.subtype_lr,
            epochs=self.subtype_epochs,
            random_seed=self.random_seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrackAHierarchicalClassical":
        # 1. Fit binary gate on all samples
        self.gate.fit(X, y)

        # 2. Extract emergency subset for subtype training
        em_mask = np.array([label in EMERGENCY_CLASSES for label in y])
        X_em = X[em_mask]
        y_em = y[em_mask]

        if len(X_em) > 0:
            self.subtype.fit(X_em, y_em)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        gate_probs = self.gate.predict_proba(X)
        is_emergency = gate_probs[:, 1] >= self.threshold

        preds = np.array([GENERAL_SPEECH_CLASS] * len(X), dtype=object)
        em_indices = np.where(is_emergency)[0]

        if len(em_indices) > 0:
            em_preds = self.subtype.predict(X[em_indices])
            for idx, p in zip(em_indices, em_preds):
                preds[idx] = p

        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        gate_probs = self.gate.predict_proba(X)  # (N, 2)
        subtype_probs = self.subtype.predict_proba(X)  # (N, 6)

        # Build (N, 7) output probability distribution
        probs = np.zeros((len(X), len(CANONICAL_CLASSES)), dtype=np.float32)
        probs[:, 0] = gate_probs[:, 0]  # P(GENERAL_SPEECH)

        for k in range(len(EMERGENCY_CLASSES)):
            probs[:, k + 1] = gate_probs[:, 1] * subtype_probs[:, k]

        # Normalize to guarantee exact sum = 1
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums


class TrackBQuantumResidual:
    """
    Track B: Quantum Residual Emergency Subtype Module.
    Combines classical binary triage with 8-qubit variational circuit for subtype refinement.
    """
    def __init__(
        self,
        pos_weight: float = 3.0,
        gate_lr: float = 0.005,
        gate_epochs: int = 50,
        vqc_lr: float = 0.01,
        vqc_epochs: int = 40,
        threshold: float = 0.5,
        random_seed: int = 42,
    ):
        self.pos_weight = pos_weight
        self.gate_lr = gate_lr
        self.gate_epochs = gate_epochs
        self.vqc_lr = vqc_lr
        self.vqc_epochs = vqc_epochs
        self.threshold = threshold
        self.random_seed = random_seed

        self.gate = BinaryEmergencyDetector(
            input_dim=384,
            hidden_dim=64,
            lr=self.gate_lr,
            pos_weight=self.pos_weight,
            epochs=self.gate_epochs,
            random_seed=self.random_seed,
        )
        self.pca_pipeline = TrainOnlyPCAPipeline(
            n_components=8,
            whiten=False,
            random_state=self.random_seed,
            feature_range=(0.0, np.pi),
        )
        self.vqc = QuantumResidualSubtypeClassifier(
            n_qubits=8,
            n_layers=2,
            n_classes=len(EMERGENCY_CLASSES),
            lr=self.vqc_lr,
            epochs=self.vqc_epochs,
            random_seed=self.random_seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrackBQuantumResidual":
        # 1. Fit Stage 1 binary gate
        self.gate.fit(X, y)

        # 2. Fit train-only PCA pipeline strictly on adaptation training set
        X_8d = self.pca_pipeline.fit_transform(X)

        # 3. Extract emergency subset for Stage 2 VQC training
        em_mask = np.array([label in EMERGENCY_CLASSES for label in y])
        X_em_8d = X_8d[em_mask]
        y_em = y[em_mask]

        if len(X_em_8d) > 0:
            self.vqc.fit(X_em_8d, y_em)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        gate_probs = self.gate.predict_proba(X)
        is_emergency = gate_probs[:, 1] >= self.threshold

        preds = np.array([GENERAL_SPEECH_CLASS] * len(X), dtype=object)
        em_indices = np.where(is_emergency)[0]

        if len(em_indices) > 0:
            X_em_8d = self.pca_pipeline.transform(X[em_indices])
            em_preds = self.vqc.predict(X_em_8d)
            for idx, p in zip(em_indices, em_preds):
                preds[idx] = p

        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        gate_probs = self.gate.predict_proba(X)
        X_8d = self.pca_pipeline.transform(X)
        subtype_probs = self.vqc.predict_proba(X_8d)

        probs = np.zeros((len(X), len(CANONICAL_CLASSES)), dtype=np.float32)
        probs[:, 0] = gate_probs[:, 0]

        for k in range(len(EMERGENCY_CLASSES)):
            probs[:, k + 1] = gate_probs[:, 1] * subtype_probs[:, k]

        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums


class TrackCMatchedClassicalShadow:
    """
    Track C: Matched Classical Shadow Model.
    Exact capacity-matched 8-D MLP control receiving identical input and objective as Track B.
    """
    def __init__(
        self,
        pos_weight: float = 3.0,
        gate_lr: float = 0.005,
        gate_epochs: int = 50,
        shadow_lr: float = 0.01,
        shadow_epochs: int = 40,
        threshold: float = 0.5,
        random_seed: int = 42,
    ):
        self.pos_weight = pos_weight
        self.gate_lr = gate_lr
        self.gate_epochs = gate_epochs
        self.shadow_lr = shadow_lr
        self.shadow_epochs = shadow_epochs
        self.threshold = threshold
        self.random_seed = random_seed

        self.gate = BinaryEmergencyDetector(
            input_dim=384,
            hidden_dim=64,
            lr=self.gate_lr,
            pos_weight=self.pos_weight,
            epochs=self.gate_epochs,
            random_seed=self.random_seed,
        )
        self.pca_pipeline = TrainOnlyPCAPipeline(
            n_components=8,
            whiten=False,
            random_state=self.random_seed,
            feature_range=(0.0, np.pi),
        )
        self.shadow = ClassicalShadowClassifier(
            input_dim=8,
            hidden_dim=16,
            num_classes=len(EMERGENCY_CLASSES),
            lr=self.shadow_lr,
            epochs=self.shadow_epochs,
            random_seed=self.random_seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrackCMatchedClassicalShadow":
        # 1. Fit Stage 1 binary gate
        self.gate.fit(X, y)

        # 2. Fit train-only PCA pipeline strictly on adaptation training set
        X_8d = self.pca_pipeline.fit_transform(X)

        # 3. Extract emergency subset for Stage 2 Shadow training
        em_mask = np.array([label in EMERGENCY_CLASSES for label in y])
        X_em_8d = X_8d[em_mask]
        y_em = y[em_mask]

        if len(X_em_8d) > 0:
            self.shadow.fit(X_em_8d, y_em)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        gate_probs = self.gate.predict_proba(X)
        is_emergency = gate_probs[:, 1] >= self.threshold

        preds = np.array([GENERAL_SPEECH_CLASS] * len(X), dtype=object)
        em_indices = np.where(is_emergency)[0]

        if len(em_indices) > 0:
            X_em_8d = self.pca_pipeline.transform(X[em_indices])
            em_preds = self.shadow.predict(X_em_8d)
            for idx, p in zip(em_indices, em_preds):
                preds[idx] = p

        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        gate_probs = self.gate.predict_proba(X)
        X_8d = self.pca_pipeline.transform(X)
        subtype_probs = self.shadow.predict_proba(X_8d)

        probs = np.zeros((len(X), len(CANONICAL_CLASSES)), dtype=np.float32)
        probs[:, 0] = gate_probs[:, 0]

        for k in range(len(EMERGENCY_CLASSES)):
            probs[:, k + 1] = gate_probs[:, 1] * subtype_probs[:, k]

        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums


def compute_phase8_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes canonical Phase 8 metrics under the locked research protocol:
    - Emergency Recall
    - Emergency Precision
    - False Emergency Rate (FER)
    - Macro-F1 across canonical 7 classes
    """
    # Binary masks
    is_true_emergency = np.array([label in EMERGENCY_CLASSES for label in y_true])
    is_pred_emergency = np.array([label in EMERGENCY_CLASSES for label in y_pred])
    is_true_general = (y_true == GENERAL_SPEECH_CLASS)

    # 1. Emergency Recall
    n_true_em = int(np.sum(is_true_emergency))
    if n_true_em > 0:
        emergency_recall = float(np.sum(is_true_emergency & is_pred_emergency) / n_true_em)
    else:
        emergency_recall = 0.0

    # 2. Emergency Precision
    n_pred_em = int(np.sum(is_pred_emergency))
    if n_pred_em > 0:
        emergency_precision = float(np.sum(is_true_emergency & is_pred_emergency) / n_pred_em)
    else:
        emergency_precision = 0.0

    # 3. False Emergency Rate (FER)
    n_true_gen = int(np.sum(is_true_general))
    if n_true_gen > 0:
        false_emergency_rate = float(np.sum(is_true_general & is_pred_emergency) / n_true_gen)
    else:
        false_emergency_rate = 0.0

    # 4. Macro-F1 across 7 canonical classes
    macro_f1 = float(f1_score(y_true, y_pred, labels=list(CANONICAL_CLASSES), average="macro", zero_division=0))

    return {
        "emergency_recall": emergency_recall,
        "emergency_precision": emergency_precision,
        "false_emergency_rate": false_emergency_rate,
        "macro_f1": macro_f1,
        "total_samples": len(y_true),
        "true_emergencies": n_true_em,
        "pred_emergencies": n_pred_em,
    }
