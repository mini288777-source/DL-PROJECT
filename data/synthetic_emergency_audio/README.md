# Synthetic Emergency Atypical Audio Benchmark Suite

This directory contains **10 synthetic emergency and atypical acoustic audio files** specifically generated for validating the **Quantum Speech Intent Application** (`qsvc_q8_r1`).

## Audio Specifications
- **Format:** PCM WAV (RIFF)
- **Sample Rate:** 16,000 Hz (Canonical 16 kHz Mono)
- **Channels:** 1 (Mono)
- **Bit Depth:** 16-bit Signed Integer
- **Acoustic Normalization:** -1.5 dBFS Peak, Active Dynamic SNR ~28-34 dB

## Sample Roster

| # | Filename | Target Intent | Emergency? | Duration | Prompt |
|---|---|---|---|---|---|
| 1 | `01_emergency_sos_help_atypical.wav` | `SOS_HELP` | 🚨 YES | 6.51s | *"Help, please help me, I need emergency help right now."* |
| 2 | `02_emergency_fall_down_atypical.wav` | `FALL` | 🚨 YES | 7.07s | *"I fell down on the floor, I cannot get up, please help."* |
| 3 | `03_emergency_breathing_medical_atypical.wav` | `BREATHING_MEDICAL` | 🚨 YES | 7.45s | *"I cannot breathe, having severe chest pain, need doctor immediately."* |
| 4 | `04_emergency_severe_pain_atypical.wav` | `PAIN` | 🚨 YES | 6.88s | *"Extreme pain, my leg hurts so bad, please send someone."* |
| 5 | `05_emergency_critical_response_atypical.wav` | `CRITICAL_RESPONSE` | 🚨 YES | 6.0s | *"Yes, this is an urgent emergency, send an ambulance now."* |
| 6 | `06_emergency_location_stairs_atypical.wav` | `LOCATION` | 🚨 YES | 5.87s | *"I am trapped near the stairs in the hallway, send help here."* |
| 7 | `07_emergency_choking_medical_atypical.wav` | `BREATHING_MEDICAL` | 🚨 YES | 5.87s | *"Choking, cannot swallow food, need urgent medical help."* |
| 8 | `08_emergency_collapse_floor_atypical.wav` | `FALL` | 🚨 YES | 7.69s | *"Help me, I collapsed on the ground, call emergency service."* |
| 9 | `09_general_speech_flowers_nominal.wav` | `GENERAL_SPEECH` | ℹ️ NO | 3.55s | *"The family requests that flowers be omitted."* |
| 10 | `10_general_speech_greeting_atypical.wav` | `GENERAL_SPEECH` | ℹ️ NO | 4.62s | *"Good morning, thank you for checking in on me today."* |

## How to Test in the Web Application

1. Open the application: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
2. Switch to the **Quantum Audio Lab** tab.
3. Drag & drop any `.wav` file from this folder into the upload dropzone (or click Browse).
4. Click **Quantum Classify**.
5. Observe:
   - Whisper speech transcription matches the prompt.
   - Signal quality displays **Nominal (16 kHz mono resampled)** with high quality score.
   - Quantum prediction executes via `qsvc_q8_r1` in ~100 ms.
   - Probability distribution highlights the predicted class and emergency status.
