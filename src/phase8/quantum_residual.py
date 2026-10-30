"""
Phase 8 Quantum Residual Emergency Subtype Module: Stage 2 for Track B.

8-Qubit Variational Quantum Circuit (VQC) with:
- 8 qubits on PennyLane default.qubit
- AngleEmbedding(8) for feature encoding
- BasicEntanglerLayers (2 layers, 16 trainable rotation parameters)
- Expectation values of Pauli-Z on wires 0..5 mapped to 6 emergency classes
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

from src.continual.taxonomy import EMERGENCY_CLASSES


class QuantumSubtypeCircuit(nn.Module):
    """
    PennyLane 8-qubit variational circuit with 16 trainable rotation parameters.
    """
    def __init__(self, n_qubits: int = 8, n_layers: int = 2, n_classes: int = 6):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes

        # Trainable rotation angles: (n_layers, n_qubits) = (2, 8) = 16 parameters
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, dtype=torch.float32) * 0.1)

        # PennyLane device and QNode
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_classes)]

        self.qnode = _circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, 8) input tensor in range [0, pi]
        Returns: (batch_size, 6) logits
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_outputs = []
        for i in range(x.shape[0]):
            expvals = self.qnode(x[i], self.weights)
            # Stack expectation values into tensor of shape (6,)
            out_i = torch.stack(expvals)
            batch_outputs.append(out_i)
        
        return torch.stack(batch_outputs)


class QuantumResidualSubtypeClassifier:
    """
    Manages training and inference of the Stage 2 Quantum VQC Subtype Module.
    """
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        n_classes: int = 6,
        lr: float = 0.01,
        epochs: int = 40,
        random_seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs
        self.random_seed = random_seed
        self.net: Optional[QuantumSubtypeCircuit] = None
        self.class_to_idx: Dict[str, int] = {cls: idx for idx, cls in enumerate(EMERGENCY_CLASSES)}
        self.idx_to_class: Dict[int, str] = {idx: cls for idx, cls in enumerate(EMERGENCY_CLASSES)}

    def fit(self, X_8d: np.ndarray, y: np.ndarray) -> "QuantumResidualSubtypeClassifier":
        """
        Fits 8-qubit variational circuit on projected 8-D emergency training samples.
        X_8d is in range [0, pi].
        y is array of emergency class names.
        """
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.net = QuantumSubtypeCircuit(self.n_qubits, self.n_layers, self.n_classes)
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
            raise RuntimeError("Quantum subtype module not fitted.")
        if len(X_8d) == 0:
            return np.zeros((0, self.n_classes), dtype=np.float32)

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
