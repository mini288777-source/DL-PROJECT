# Project Context

## Title
Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication

## Team
kernal

## Core Research Question
Can a quantum kernel provide better few-shot personalization of atypical speech than classical adaptation methods when only a very small number of labeled samples from an individual speaker are available?

## Scientific Principle
The quantum method is a **hypothesis to test**, not an assumed advantage. The project must remain scientifically useful even if the quantum method performs similarly to or worse than classical baselines.

## Application Context
Reliable recognition of fixed emergency-oriented speech/intent for people with atypical speech (dysarthria, cerebral palsy, ALS). Uses an emergency-focused fixed vocabulary/intent framing rather than unrestricted daily speech.

## Research Pipeline

```
TORGO / UASpeech / SAP
  → Preprocessing (resample, normalize, segment)
  → Whisper encoder embeddings
  → Three adaptation strategies (same conditions):
      1. Classical fine-tuning / adapter baseline
      2. Classical RBF kernel baseline
      3. Proposed quantum kernel (ZZFeatureMap → FidelityQuantumKernel)
  → Transcription + Intent classification
  → Evaluation: WER / Intent-F1 / IPER
  → Sample-efficiency comparison (N = 20, 30, 40, 50)
  → Noise robustness sweep (clean, 15 dB, 10 dB, 5 dB)
  → Research evidence
```

## Datasets

| Dataset   | Status       | Notes                                  |
|-----------|-------------|----------------------------------------|
| TORGO     | Available   | Current executable dataset (parquet)   |
| UASpeech  | Planned     | Future integration                     |
| SAP       | Planned     | Future integration                     |

## Technical Stack

| Component              | Implementation                        |
|------------------------|---------------------------------------|
| Speech encoder         | Whisper (pretrained, encoder only)    |
| Classical baseline 1   | Fine-tuning / adapter                 |
| Classical baseline 2   | RBF kernel + SVM                      |
| Quantum kernel         | PCA → ZZFeatureMap → FidelityQuantumKernel |
| Quantum backend        | Qiskit statevector simulator          |
| Few-shot regime        | N = 20, 30, 40, 50 per speaker       |
| Noise conditions       | Clean, 15 dB, 10 dB, 5 dB SNR       |
| Metrics                | WER, Intent-F1, IPER                  |

## Phases

| Phase | Description                          | Status        |
|-------|--------------------------------------|---------------|
| 0     | Environment & smoke tests            | **Complete**  |
| 1     | TORGO data foundation & splits       | **Complete**  |
| 2     | Acoustic representation & embeddings | **Active**    |
| 3     | Classical baselines (Fine-tuning/RBF)| Planned       |
| 4     | Quantum kernel personalization       | Planned       |
| 5     | Noise robustness evaluation          | Planned       |
| 6     | Sample-efficiency & comparative study| Planned       |
| 7     | Analysis & final research report     | Planned       |
