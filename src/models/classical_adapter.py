"""
Classical Adapter Head Model for Frozen Whisper Embeddings.

Phase 3 Baseline A: Lightweight trainable MLP projection and classification head
operating on top of frozen 384-dimensional Whisper acoustic representations.
Strictly deterministic initialization, configurable capacity, and parameter logging.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils.seeding import set_seed


class ClassicalAdapterNet(nn.Module):
    """
    Lightweight feedforward adapter network for acoustic embeddings.

    Architecture:
        Input (d_in = 384)
            -> Linear(d_in, d_hidden)
            -> LayerNorm(d_hidden)
            -> GELU()
            -> Dropout(dropout_p)
            -> Linear(d_hidden, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 64,
        num_classes: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_p = dropout

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Deterministic Xavier uniform weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits."""
        return self.net(x)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ClassicalAdapterClassifier:
    """
    Scikit-learn style wrapper for training and evaluating ClassicalAdapterNet.
    Manages deterministic training, label encoding/decoding, class weighting,
    and inference latency tracking.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 64,
        num_classes: int = 7,
        dropout: float = 0.1,
        learning_rate: float = 0.005,
        weight_decay: float = 1e-4,
        epochs: int = 60,
        batch_size: int = 8,
        class_weight: Optional[str] = "balanced",
        seed: int = 42,
        device: str = "cpu",
        all_classes: Optional[List[str]] = None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.seed = seed
        self.device = torch.device(device)

        # Global intent classes from taxonomy
        if all_classes is None:
            self.classes_ = [
                "SOS_HELP",
                "FALL",
                "BREATHING_MEDICAL",
                "PAIN",
                "LOCATION",
                "CRITICAL_RESPONSE",
                "GENERAL_SPEECH",
            ]
        else:
            self.classes_ = list(all_classes)

        self.num_classes = len(self.classes_)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes_)}

        self.model: Optional[ClassicalAdapterNet] = None
        self.adaptation_time_sec: float = 0.0
        self.inference_time_sec: float = 0.0
        self.is_single_class_fallback: bool = False
        self.single_observed_class: Optional[str] = None

    def fit(self, X: np.ndarray, y: List[str]) -> "ClassicalAdapterClassifier":
        """
        Fit adapter head on speaker adaptation embeddings and labels.

        Args:
            X: 2D numpy array of shape (N, input_dim).
            y: List of intent strings of length N.

        Returns:
            self
        """
        start_time = time.perf_counter()
        set_seed(self.seed)

        unique_labels = sorted(list(set(y)))

        # Handle edge-case: single observed class in few-shot adaptation
        if len(unique_labels) == 1:
            self.is_single_class_fallback = True
            self.single_observed_class = unique_labels[0]
            self.adaptation_time_sec = time.perf_counter() - start_time
            # Instantiate model anyway for parameter accounting
            self.model = ClassicalAdapterNet(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_classes=self.num_classes,
                dropout=self.dropout,
            ).to(self.device)
            return self

        self.is_single_class_fallback = False
        self.single_observed_class = None

        # Build network
        self.model = ClassicalAdapterNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            dropout=self.dropout,
        ).to(self.device)
        self.model.train()

        # Map targets to integer indices
        y_indices = np.array([self.class_to_idx[label] for label in y], dtype=np.int64)

        # Compute balanced class weights if specified
        weight_tensor = None
        if self.class_weight == "balanced":
            counts = np.bincount(y_indices, minlength=self.num_classes).astype(np.float32)
            # Avoid division by zero for unobserved classes in adaptation
            weights = np.zeros(self.num_classes, dtype=np.float32)
            observed_mask = counts > 0
            n_samples = len(y)
            n_observed_classes = np.sum(observed_mask)
            weights[observed_mask] = n_samples / (n_observed_classes * counts[observed_mask])
            weight_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Prepare dataset & loader
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_indices, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = min(self.batch_size, len(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        # Training loop
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        self.adaptation_time_sec = time.perf_counter() - start_time
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute predicted class probabilities."""
        start_time = time.perf_counter()

        if self.is_single_class_fallback:
            probs = np.zeros((len(X), self.num_classes), dtype=np.float32)
            c_idx = self.class_to_idx[self.single_observed_class]
            probs[:, c_idx] = 1.0
            self.inference_time_sec = time.perf_counter() - start_time
            return probs

        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        self.inference_time_sec = time.perf_counter() - start_time
        return probs

    def predict(self, X: np.ndarray) -> List[str]:
        """Predict intent class labels."""
        if self.is_single_class_fallback:
            return [self.single_observed_class] * len(X)

        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return [self.idx_to_class[idx] for idx in pred_indices]

    def get_parameter_count(self) -> int:
        """Return total trainable parameter count."""
        if self.model is None:
            # Calculate theoretical count
            net = ClassicalAdapterNet(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_classes=self.num_classes,
                dropout=self.dropout,
            )
            return net.count_parameters()
        return self.model.count_parameters()

    def get_model_info(self) -> Dict[str, Any]:
        """Return architecture and training metadata."""
        return {
            "method": "adapter",
            "architecture": f"Linear({self.input_dim}, {self.hidden_dim}) -> LayerNorm({self.hidden_dim}) -> GELU -> Dropout({self.dropout}) -> Linear({self.hidden_dim}, {self.num_classes})",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "parameter_count": self.get_parameter_count(),
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "class_weight": self.class_weight,
            "seed": self.seed,
            "is_single_class_fallback": self.is_single_class_fallback,
        }
