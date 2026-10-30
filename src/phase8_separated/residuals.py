"""
Phase 8 Repaired: Track-Specific Residual Modules.

Implements the three matched residual modules:
- Track A: ClassicalResidualModule (384-D MLP -> scalar logit)
- Track B: QuantumResidualModule (8-Qubit VQC -> Pauli-Z expval -> scalar logit)
- Track C: MatchedShadowResidualModule (8-D MLP -> scalar logit)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

from src.continual.taxonomy import EMERGENCY_CLASSES


class ClassicalResidualNet(nn.Module):
    """384-D Classical Residual Network producing scalar logit z_residual."""
    def __init__(self, input_dim: int = 384, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        return self.fc2(h).squeeze(-1)


class ClassicalResidualModule:
    """Manages Track A Classical Residual training and inference."""
    def __init__(self, input_dim: int = 384, hidden_dim: int = 64, lr: float = 0.005, epochs: int = 30, random_seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.random_seed = random_seed
        self.net: Optional[ClassicalResidualNet] = None

    def fit(self, X: np.ndarray, y: np.ndarray, pos_weight: float = 1.0) -> "ClassicalResidualModule":
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.net = ClassicalResidualNet(self.input_dim, self.hidden_dim)
        self.net.train()

        binary_y = np.array([1.0 if label in EMERGENCY_CLASSES else 0.0 for label in y], dtype=np.float32)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(binary_y, dtype=torch.float32)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
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

    def predict_residual_logits(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Module not fitted.")
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = self.net(X_t).cpu().numpy()
        return logits


class QuantumResidualCircuit(nn.Module):
    """
    8-Qubit Variational Quantum Circuit on PennyLane default.qubit.
    Encodes 8-D input via AngleEmbedding.
    Ansatz: BasicEntanglerLayers (2 layers = 16 trainable angles).
    Measures expectation of PauliZ on wire 0.
    Scales expectation value by a trainable weight w and bias b to form raw logit.
    """
    def __init__(self, n_qubits: int = 8, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Trainable rotation angles: (2, 8) = 16 parameters
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, dtype=torch.float32) * 0.1)
        self.scale = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        self.qnode = _circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        try:
            expvals = self.qnode(x, self.weights)
        except Exception:
            outputs = []
            for i in range(x.shape[0]):
                expval = self.qnode(x[i], self.weights)
                outputs.append(expval)
            expvals = torch.stack(outputs)
        return self.scale * expvals + self.bias


class QuantumResidualModule:
    """Manages Track B Quantum Residual training and inference."""
    def __init__(self, n_qubits: int = 8, n_layers: int = 2, lr: float = 0.01, epochs: int = 25, random_seed: int = 42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.random_seed = random_seed
        self.net: Optional[QuantumResidualCircuit] = None
        self.initial_weights_fingerprint: Optional[str] = None
        self.final_weights_fingerprint: Optional[str] = None
        self.weight_delta_norm: float = 0.0

    def fit(self, X_8d: np.ndarray, y: np.ndarray, pos_weight: float = 1.0) -> "QuantumResidualModule":
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.net = QuantumResidualCircuit(self.n_qubits, self.n_layers)
        # Capture initial parameter fingerprint
        init_w = self.net.weights.detach().cpu().numpy().copy()
        self.initial_weights_fingerprint = str(np.round(init_w, 4).tolist())

        self.net.train()

        binary_y = np.array([1.0 if label in EMERGENCY_CLASSES else 0.0 for label in y], dtype=np.float32)
        X_t = torch.tensor(X_8d, dtype=torch.float32)
        y_t = torch.tensor(binary_y, dtype=torch.float32)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
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
        # Capture final parameter fingerprint
        final_w = self.net.weights.detach().cpu().numpy().copy()
        self.final_weights_fingerprint = str(np.round(final_w, 4).tolist())
        self.weight_delta_norm = float(np.linalg.norm(final_w - init_w))
        return self

    def predict_residual_logits(self, X_8d: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Module not fitted.")
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_8d, dtype=torch.float32)
            logits = self.net(X_t).cpu().numpy()
        return logits


class MatchedShadowResidualNet(nn.Module):
    """Capacity-matched 8-D MLP producing scalar logit z_residual."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        return self.fc2(h).squeeze(-1)


class MatchedShadowResidualModule:
    """Manages Track C Matched Classical Shadow training and inference."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, lr: float = 0.01, epochs: int = 25, random_seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.random_seed = random_seed
        self.net: Optional[MatchedShadowResidualNet] = None

    def fit(self, X_8d: np.ndarray, y: np.ndarray, pos_weight: float = 1.0) -> "MatchedShadowResidualModule":
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.net = MatchedShadowResidualNet(self.input_dim, self.hidden_dim)
        self.net.train()

        binary_y = np.array([1.0 if label in EMERGENCY_CLASSES else 0.0 for label in y], dtype=np.float32)
        X_t = torch.tensor(X_8d, dtype=torch.float32)
        y_t = torch.tensor(binary_y, dtype=torch.float32)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
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

    def predict_residual_logits(self, X_8d: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Module not fitted.")
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_8d, dtype=torch.float32)
            logits = self.net(X_t).cpu().numpy()
        return logits
