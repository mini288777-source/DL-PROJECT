# FINAL MODEL CONFIGURATION REGISTRY
## Authoritative Architecture & Parameter Specifications

```yaml
# ==============================================================================
# CANONICAL MODEL REGISTRY
# AUTHORITATIVE CONFIGURATION FOR ALL EVALUATED MODELS
# ==============================================================================

models:
  classical_adapter:
    model_id: "classical_adapter"
    family: "classical_deep_learning"
    display_name: "Classical Adapter Head (MLP)"
    source_class: "src.models.classical_adapter.ClassicalAdapterNet"
    wrapper_class: "src.models.classical_adapter.ClassicalAdapterClassifier"
    architecture:
      layers:
        - "Linear(input_dim, 64)"
        - "LayerNorm(64)"
        - "GELU()"
        - "Dropout(p=0.1)"
        - "Linear(64, num_classes)"
      parameter_count:
        input_384_classes_7: 25223
        input_8_classes_2: 642
        input_8_classes_6: 898
        input_4_classes_2: 386
        input_4_classes_6: 642
    training:
      optimizer: "AdamW"
      learning_rate: 0.005
      weight_decay: 0.0001
      epochs: 60
      batch_size: 8
      seed: 42
    reconciliation_note: >
      A preliminary Phase 5 report text historically referenced 'ReLU + Dropout(0.2)'.
      Forensic code audit (Phase 5 LOG-P5-01) confirmed that both Phase 3 and Phase 5 code
      and checkpoint state dicts executed GELU + Dropout(0.1) with exactly 25,223 parameters.
      The preliminary prose was an isolated documentation typo and was formally superseded.

  classical_rbf_svm:
    model_id: "classical_rbf_svm"
    family: "classical_kernel_machine"
    display_name: "Classical Support Vector Classifier (RBF Kernel)"
    source_class: "sklearn.svm.SVC"
    wrapper_class: "src.models.rbf_svm.ClassicalRBFSVMClassifier"
    parameters:
      kernel: "rbf"
      C: 1.0
      gamma: "scale"
      class_weight: "balanced"
      probability: true
      random_state: 42
    input_representation:
      flat_original: "PCA d=4, StandardScaler"
      recovery_track_primary: "PCA d=8, MinMaxScaler [0, pi] on train only"
      recovery_track_compact: "PCA d=4, MinMaxScaler [0, pi] on train only"
    role: "Primary matched classical kernel baseline for QSVC comparison"

  qsvc_q4_r1:
    model_id: "qsvc_q4_r1"
    family: "quantum_kernel_machine"
    display_name: "Quantum SVC (4 Qubits, Reps=1)"
    source_class: "src.models.quantum_kernel.QSVCClassifier"
    parameters:
      qubit_count: 4
      representation_dim: 4
      feature_map: "ZZFeatureMap"
      entanglement: "linear"
      reps: 1
      kernel_type: "FidelityQuantumKernel"
      classifier: "QSVC"
      C: 1.0
      backend: "exact_statevector"
      simulation_framework: "qiskit_algorithms"
    scaling: "MinMaxScaler fitted on adaptation embeddings to [0, pi]"
    role: "Compact quantum model evaluated in Phase 4 and Phase 6.5.5 qubit ablation"

  qsvc_q8_r1:
    model_id: "qsvc_q8_r1"
    family: "quantum_kernel_machine"
    display_name: "Quantum SVC (8 Qubits, Reps=1)"
    source_class: "src.models.quantum_kernel.QSVCClassifier"
    parameters:
      qubit_count: 8
      representation_dim: 8
      feature_map: "ZZFeatureMap"
      entanglement: "linear"
      reps: 1
      kernel_type: "FidelityQuantumKernel"
      classifier: "QSVC"
      C: 1.0
      backend: "exact_statevector"
      simulation_framework: "qiskit_algorithms"
    scaling: "MinMaxScaler fitted on adaptation embeddings to [0, pi]"
    role: "PRIMARY CANDIDATE QUANTUM MODEL for definitive matched comparison against RBF-SVM (d=8)"

  qsvc_q8_r2:
    model_id: "qsvc_q8_r2"
    family: "quantum_kernel_machine"
    display_name: "Quantum SVC (8 Qubits, Reps=2)"
    source_class: "src.models.quantum_kernel.QSVCClassifier"
    parameters:
      qubit_count: 8
      representation_dim: 8
      feature_map: "ZZFeatureMap"
      entanglement: "linear"
      reps: 2
      kernel_type: "FidelityQuantumKernel"
      classifier: "QSVC"
      C: 1.0
      backend: "exact_statevector"
      simulation_framework: "qiskit_algorithms"
    scaling: "MinMaxScaler fitted on adaptation embeddings to [0, pi]"
    role: "Deep variational circuit depth ablation evaluated in Phase 6.5.5"

historical_exploratory_models:
  linear_svm:
    model_id: "linear_svm"
    family: "classical_linear"
    description: "Evaluated as an exploratory linear baseline in preliminary Phase 3 screening"
    status: "HISTORICAL_EXPLORATORY_ONLY"
  knn:
    model_id: "knn"
    family: "classical_instance_based"
    description: "Evaluated as an exploratory nearest-neighbor baseline in preliminary Phase 3 screening"
    status: "HISTORICAL_EXPLORATORY_ONLY"

```
