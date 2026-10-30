"""
Phase 8 Repaired: Stage 1 Classical Emergency Gate.

Produces a continuous scalar logit z_gate representing the prior classical
emergency confidence, trained with class-balanced BCEWithLogitsLoss.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.continual.taxonomy import EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS


class ScalarEmergencyGateNet(nn.Module):
    """
    Stage 1 Binary Emergency Net producing a single scalar logit.
    Input: 384-D Whisper embedding.
    Output: (batch_size, 1) raw logit z_gate.
    """
    def __init__(self, input_dim: int = 384, hidden_dim: int = 64, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.ln(h)
        h = self.act(h)
        h = self.dropout(h)
        logits = self.fc2(h)
        return logits.squeeze(-1)


class ScalarEmergencyGate:
    """
    Manages training and logit inference of Stage 1 Scalar Emergency Gate.
    """
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 64,
        lr: float = 0.005,
        epochs: int = 40,
        random_seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.random_seed = random_seed
        self.net: Optional[ScalarEmergencyGateNet] = None
        self.pos_weight_value: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScalarEmergencyGate":
        """
        Fits gate on adaptation training data.
        y is array of strings (EMERGENCY_CLASSES -> 1.0, GENERAL_SPEECH -> 0.0).
        """
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.net = ScalarEmergencyGateNet(self.input_dim, self.hidden_dim)
        self.net.train()

        binary_y = np.array([1.0 if label in EMERGENCY_CLASSES else 0.0 for label in y], dtype=np.float32)
        n_pos = float(np.sum(binary_y == 1.0))
        n_neg = float(np.sum(binary_y == 0.0))
        # Compute pos_weight strictly from training data
        self.pos_weight_value = n_neg / max(1.0, n_pos)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(binary_y, dtype=torch.float32)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight_value], dtype=torch.float32))
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

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Returns shape (N,) raw scalar logits z_gate."""
        if self.net is None:
            raise RuntimeError("Gate not fitted.")
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = self.net(X_t).cpu().numpy()
        return logits
