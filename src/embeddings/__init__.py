"""
Embeddings package for speech representation extraction, MFCC baseline features, and persistent caching.

Exports:
    - load_audio, preprocess_audio
    - WhisperEncoder, extract_whisper_embedding
    - MFCCExtractor, extract_mfcc_features
    - save_embedding_file, load_embedding_file, has_cached_embedding, EmbeddingManifestManager
"""

from src.embeddings.whisper_preprocessing import load_audio, preprocess_audio
from src.embeddings.whisper_encoder import WhisperEncoder, extract_whisper_embedding
from src.embeddings.mfcc import MFCCExtractor, extract_mfcc_features
from src.embeddings.cache import (
    save_embedding_file,
    load_embedding_file,
    has_cached_embedding,
    get_embedding_path,
    EmbeddingManifestManager,
)

__all__ = [
    "load_audio",
    "preprocess_audio",
    "WhisperEncoder",
    "extract_whisper_embedding",
    "MFCCExtractor",
    "extract_mfcc_features",
    "save_embedding_file",
    "load_embedding_file",
    "has_cached_embedding",
    "get_embedding_path",
    "EmbeddingManifestManager",
]
