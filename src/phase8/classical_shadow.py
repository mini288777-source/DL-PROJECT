"""
Phase 8 Matched Classical Shadow Model: Stage 2 for Track C.

Capacity-matched MLP operating on the identical 8-D bottleneck projection as Track B:
- Architecture: Linear(8, 16) -> GELU() -> Linear(16, 6)
- Trainable parameters: (8*16 + 16) + (16*6 + 6) = 144 + 102 = 246 parameters
- Strict matched input: Receives identical 8-D features in [0, pi]
- Matched objective: CrossEntropyLoss over the 6 emergency classes
- Matched optimization: Adam(lr=0.01), 40 epochs
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.continual.taxonomy import EMERGENCY_CLASSES


class ClassicalShadowNet(nn.Module):
    """
    Capacity-matched 8-D MLP for Stage 2 emergency subtype classification.
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, num_classes: int = 6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        return self.fc2(h)


class ClassicalShadowClassifier:
    """
    Manages training and inference of the Stage 2 Classical Shadow Model.
    """
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        num_classes: int = 6,
        lr: float = 0.01,
        epochs: int = 40,
        random_seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.random_seed = random_seed
        self.net: Optional[ClassicalShadowNet] = None
        self.class_to_idx: Dict[str, int] = {cls: idx for idx, cls in enumerate(EMERGENCY_CLASSES)}
        self.idx_to_class: Dict[int, str] = {idx: cls for idx, cls in enumerate(EMERGENCY_CLASSES)}

    def fit(self, X_8d: np.ndarray, y: np.ndarray) -> "ClassicalShadowClassifier":
        """
        Fits matched MLP on projected 8-D emergency training samples.
        X_8d is in range [0, pi].
        y is array of emergency class names.
        """
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.net = ClassicalShadowNet(self.input_dim, self.hidden_dim, self.num_classes)
        self.net.train()

        if len(X_8d) == 0:
            return self

        y_indices = np.array([self.class_to_idx[label] for label in y], dtype=np.int64)
        X_t = torch.tensor(X_8d, dtype=torch.float32)
        y_t = torch.tensor(y_indices, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        batch_size = min(8, max(2, len(X_8d)))
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

    def predict_proba(self, X_8d: np.ndarray) -> np.ndarray:
        """Returns shape (N, 6) probabilities over the 6 emergency classes."""
        if self.net is None:
            raise RuntimeError("Classical shadow model not fitted.")
        if len(X_8d) == 0:
            return np.zeros((0, self.num_classes), dtype=np.float32)

        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_8d, dtype=torch.float32)
            logits = self.net(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X_8d: np.ndarray) -> List[str]:
        """Returns predicted emergency class strings."""
        probs = self.predict_proba(X_8d)
        if len(probs) == 0:
            return []
        preds = np.argmax(probs, axis=1)
        return [self.idx_to_class[p] for p in preds]
