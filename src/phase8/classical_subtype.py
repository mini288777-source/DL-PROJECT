"""
Phase 8 Classical Subtype Classifier: Stage 2 for Track A.

Discriminates between the 6 canonical emergency classes given an emergency candidate:
['CRITICAL_RESPONSE', 'FALL', 'BREATHING_MEDICAL', 'PAIN', 'SOS_HELP', 'LOCATION']
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.continual.taxonomy import EMERGENCY_CLASSES


class ClassicalSubtypeNet(nn.Module):
    """
    Stage 2 Subtype Classifier operating on 384-D Whisper embeddings.
    Maps to the 6 canonical emergency classes.
    """
    def __init__(self, input_dim: int = 384, hidden_dim: int = 64, num_classes: int = 6, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.ln(h)
        h = self.act(h)
        h = self.dropout(h)
        return self.fc2(h)


class ClassicalSubtypeClassifier:
    """
    Manages training and inference of the Stage 2 Classical Subtype Net.
    """
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 64,
        num_classes: int = 6,
        lr: float = 0.005,
        epochs: int = 50,
        random_seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.random_seed = random_seed
        self.net: Optional[ClassicalSubtypeNet] = None
        self.class_to_idx: Dict[str, int] = {cls: idx for idx, cls in enumerate(EMERGENCY_CLASSES)}
        self.idx_to_class: Dict[int, str] = {idx: cls for idx, cls in enumerate(EMERGENCY_CLASSES)}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalSubtypeClassifier":
        """
        Fits Stage 2 network on emergency samples only.
        y is an array of emergency class names.
        """
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.net = ClassicalSubtypeNet(self.input_dim, self.hidden_dim, self.num_classes)
        self.net.train()

        if len(X) == 0:
            return self

        y_indices = np.array([self.class_to_idx[label] for label in y], dtype=np.int64)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_indices, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        batch_size = min(8, max(2, len(X)))
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
        """Returns shape (N, 6) probabilities over the 6 emergency classes."""
        if self.net is None:
            raise RuntimeError("Subtype classifier not fitted.")
        if len(X) == 0:
            return np.zeros((0, self.num_classes), dtype=np.float32)

        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = self.net(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> List[str]:
        """Returns predicted emergency class strings."""
        probs = self.predict_proba(X)
        if len(probs) == 0:
            return []
        preds = np.argmax(probs, axis=1)
        return [self.idx_to_class[p] for p in preds]
