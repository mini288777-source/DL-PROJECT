"""
Embedding Cache Management and Manifest Tracking.

Phase 2: Atomic saving, strict validation, zero-dependency loading of .npy embeddings,
and comprehensive embedding manifest management.
Ensures 1-to-1 traceability from recording_id to cached representation.
"""

import os
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def get_embedding_path(
    recording_id: str,
    feature_type: str = "whisper",
    base_dir: str = "embeddings/torgo"
) -> str:
    """Get the standard relative or absolute path for a cached .npy file."""
    return os.path.join(base_dir, feature_type, f"{recording_id}.npy")


def has_cached_embedding(
    recording_id: str,
    feature_type: str = "whisper",
    base_dir: str = "embeddings/torgo"
) -> bool:
    """Check if a valid cached .npy file exists on disk."""
    path = get_embedding_path(recording_id, feature_type=feature_type, base_dir=base_dir)
    return os.path.isfile(path) and os.path.getsize(path) > 0


def save_embedding_file(
    recording_id: str,
    array: np.ndarray,
    feature_type: str = "whisper",
    base_dir: str = "embeddings/torgo"
) -> str:
    """
    Safely save an embedding array to disk as a .npy file.
    Uses atomic write via temporary file to prevent partial write corruptions.

    Args:
        recording_id: Unique recording identifier.
        array: 1D numpy array of float32 features.
        feature_type: Subdirectory name ('whisper' or 'mfcc').
        base_dir: Base embeddings directory.

    Returns:
        Final file path of the saved .npy file.
    """
    target_dir = os.path.join(base_dir, feature_type)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f"{recording_id}.npy")

    # Validate array before saving
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(array)}")
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"Cannot save non-finite embedding for {recording_id}")

    # Atomic write pattern
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"tmp_{recording_id}_{feature_type}.npy")
    try:
        np.save(temp_file, array)
        shutil.move(temp_file, target_path)
    finally:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass

    return target_path


def load_embedding_file(
    recording_id: str,
    feature_type: str = "whisper",
    base_dir: str = "embeddings/torgo",
    expected_dim: Optional[int] = None
) -> np.ndarray:
    """
    Reload a cached .npy embedding file with strict integrity validation.
    No model inference or PyTorch model loading is invoked.

    Args:
        recording_id: Unique recording identifier.
        feature_type: Feature type ('whisper' or 'mfcc').
        base_dir: Base embeddings directory.
        expected_dim: Optional expected feature dimension to verify.

    Returns:
        1D numpy array with dtype float32.
    """
    path = get_embedding_path(recording_id, feature_type=feature_type, base_dir=base_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cached {feature_type} embedding not found for: {recording_id} at {path}")

    array = np.load(path)

    # Integrity validations
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Loaded object for {recording_id} is not a numpy array")
    if array.dtype != np.float32:
        raise TypeError(f"Loaded embedding for {recording_id} has unexpected dtype {array.dtype}, expected float32")
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"Loaded embedding for {recording_id} contains NaN or Inf values")
    if expected_dim is not None and array.shape[-1] != expected_dim:
        raise ValueError(f"Loaded embedding for {recording_id} has shape {array.shape}, expected last dim {expected_dim}")

    return array


class EmbeddingManifestManager:
    """
    Manages reading, updating, and saving the master embedding manifest CSV.
    Tracks lineage, dimensions, paths, and extraction status.
    """

    MANIFEST_COLUMNS = [
        "recording_id",
        "speaker_id",
        "session_id",
        "split",
        "intent",
        "audio_path",
        "whisper_path",
        "whisper_dim",
        "whisper_dtype",
        "mfcc_path",
        "mfcc_dim",
        "duration_sec",
        "sample_rate",
        "model_name",
        "pooling_method",
        "extraction_status",
        "extraction_timestamp"
    ]

    def __init__(self, manifest_path: str = "embeddings/metadata/embedding_manifest.csv"):
        self.manifest_path = manifest_path
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        self.df = self._load_or_create()

    def _load_or_create(self) -> pd.DataFrame:
        string_cols = [
            "recording_id", "speaker_id", "session_id", "split", "intent",
            "audio_path", "whisper_path", "whisper_dtype", "mfcc_path",
            "model_name", "pooling_method", "extraction_status", "extraction_timestamp"
        ]
        if os.path.isfile(self.manifest_path):
            df = pd.read_csv(self.manifest_path)
            for col in self.MANIFEST_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            for col in string_cols:
                if col in df.columns:
                    df[col] = df[col].astype("object")
            return df
        else:
            df = pd.DataFrame(columns=self.MANIFEST_COLUMNS)
            for col in string_cols:
                df[col] = df[col].astype("object")
            return df

    def upsert_record(self, record_dict: Dict[str, Any]) -> None:
        """Insert or update a row in the manifest for recording_id."""
        rec_id = record_dict["recording_id"]
        # Ensure timestamp
        if "extraction_timestamp" not in record_dict or not record_dict["extraction_timestamp"]:
            record_dict["extraction_timestamp"] = datetime.now(timezone.utc).isoformat()

        # Check if row exists
        existing_indices = self.df.index[self.df["recording_id"] == rec_id].tolist()
        if existing_indices:
            idx = existing_indices[0]
            for k, v in record_dict.items():
                if k in self.MANIFEST_COLUMNS:
                    self.df.at[idx, k] = v
        else:
            new_row = {col: record_dict.get(col, None) for col in self.MANIFEST_COLUMNS}
            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

    def save(self) -> None:
        """Save the manifest DataFrame atomically to disk."""
        temp_path = self.manifest_path + ".tmp"
        self.df.to_csv(temp_path, index=False)
        shutil.move(temp_path, self.manifest_path)
