"""
Patch IPER columns in phase5_predictions.csv to ensure no GENERAL_SPEECH
utterance is recorded as a preserved emergency intent.

Adds explicit columns:
- is_emergency: bool (True for emergency intents, False for GENERAL_SPEECH)
- intent_preserved: bool (True ONLY if is_emergency AND true_intent == predicted_intent)
- triage_preserved: bool (True ONLY if is_emergency AND predicted_intent != 'GENERAL_SPEECH')
"""

import os
import pandas as pd

files_to_update = [
    "results/phase5/phase5_predictions.csv",
    "phase5/results/phase5_predictions.csv"
]

for filepath in files_to_update:
    if not os.path.isfile(filepath):
        print(f"Skipping {filepath} (not found)")
        continue

    print(f"Patching {filepath}...")
    df = pd.read_csv(filepath)

    is_emg = df["true_intent"] != "GENERAL_SPEECH"
    exact_preserved = is_emg & (df["true_intent"] == df["predicted_intent"])
    triage_preserved = is_emg & (df["predicted_intent"] != "GENERAL_SPEECH")

    df["is_emergency"] = is_emg
    df["intent_preserved"] = exact_preserved
    df["triage_preserved"] = triage_preserved

    # Verify that no GENERAL_SPEECH row has intent_preserved == True
    non_emg_preserved = df[~df["is_emergency"]]["intent_preserved"].sum()
    assert non_emg_preserved == 0, f"Error: {non_emg_preserved} non-emergency rows have intent_preserved=True"

    df.to_csv(filepath, index=False)
    print(f"Successfully patched {filepath}:")
    print(f"  Total rows: {len(df)}")
    print(f"  Emergency rows: {is_emg.sum()}")
    print(f"  Exact emergency preserved rows: {exact_preserved.sum()}")
    print(f"  Triage emergency preserved rows: {triage_preserved.sum()}")
    print(f"  Non-emergency preserved rows: {non_emg_preserved}")
