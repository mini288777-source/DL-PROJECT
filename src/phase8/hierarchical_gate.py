"""
Phase 8 Hierarchical Emergency Gate: Stage 1 Binary Emergency Detector.

Decouples binary emergency detection (General Speech vs Emergency) from
fine-grained 6-class subtype classification to overcome Failure Mode 1
(Extreme majority class dominance, ~95% GENERAL_SPEECH).
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.continual.taxonomy import EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS


class BinaryEmergencyGateNet(nn.Module):
    """
    Stage 1 Binary Emergency Classifier operating on 384-D frozen Whisper embeddings.
    Trained with class-weighted Cross-Entropy to counteract the 95:5 class imbalance.
    """
    def __init__(self, input_dim: int = 384, hidden_dim: int = 64, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 2)  # Class 0: General Speech, Class 1: Emergency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.ln(h)
        h = self.act(h)
        h = self.dropout(h)
        logits = self.fc2(h)
        return logits


class BinaryEmergencyDetector:
    """
    Manages training, calibration, and inference of the Stage 1 Binary Gate.
    """
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 64,
        lr: float = 0.005,
        pos_weight: float = 3.0,
        epochs: int = 50,
        random_seed: int = 42
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.pos_weight = pos_weight
        self.epochs = epochs
        self.random_seed = random_seed
        self.net: Optional[BinaryEmergencyGateNet] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinaryEmergencyDetector":
        """
        Fits binary gate.
        y is an array of string labels from canonical 7 classes.
        Emergency classes -> 1, GENERAL_SPEECH -> 0.
        """
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.net = BinaryEmergencyGateNet(self.input_dim, self.hidden_dim)
        self.net.train()

        # Convert labels to binary: 1 if emergency, 0 if general
        binary_y = np.array([1 if label in EMERGENCY_CLASSES else 0 for label in y], dtype=np.int64)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(binary_y, dtype=torch.long)

        # Loss with pos_weight to prioritize Emergency Recall
        class_weights = torch.tensor([1.0, float(self.pos_weight)], dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        batch_size = min(16, max(4, len(X)))
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.net(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        self.net.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns shape (N, 2) probabilities [P(General), P(Emergency)]."""
        if self.net is None:
            raise RuntimeError("Detector not fitted.")
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = self.net(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
