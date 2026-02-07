"""Cognitive Deduplication — cross-system compression via C3 memory store.

Features:
- LSH-accelerated similarity search (O(1) per band)
- Tiered chunks (hot/warm/cold) mirroring C3's memory model
- Predictive pre-compression using co-occurrence data
- Streaming encoder for real-time agent trace compression
- Compression-aware retrieval via structural similarity
- Self-compressing context windows (token savings for LLM calls)
- Compression-based anomaly detection (drift/novelty signals)
- Cross-agent knowledge transfer via cogstore federation
- Adversarial robustness (integrity verification, ref_count limits)
- Recursive self-compression (USC compressing C3's own state)
- Temporal compression (event sequence motif detection)
"""
from usc.cogdedup.codec import cogdedup_encode, cogdedup_decode
from usc.cogdedup.store import CogStore, MemoryCogStore, ChunkEntry
from usc.cogdedup.lsh import LSHIndex
from usc.cogdedup.predictor import PredictiveCompressor
from usc.cogdedup.streaming import CogdedupStream
from usc.cogdedup.context_compactor import ContextCompactor, CompactionResult
from usc.cogdedup.anomaly import AnomalyDetector, AnomalyAlert, DriftReport
from usc.cogdedup.federation import CogstoreFederation, FederatedStore
from usc.cogdedup.integrity import IntegrityVerifier, SecurityPolicy
from usc.cogdedup.recursive import RecursiveCompressor, CompressionResult
from usc.cogdedup.temporal import TemporalMotifTracker, TemporalEncoder

__all__ = [
    "cogdedup_encode", "cogdedup_decode",
    "CogStore", "MemoryCogStore", "ChunkEntry",
    "LSHIndex", "PredictiveCompressor", "CogdedupStream",
    "ContextCompactor", "CompactionResult",
    "AnomalyDetector", "AnomalyAlert", "DriftReport",
    "CogstoreFederation", "FederatedStore",
    "IntegrityVerifier", "SecurityPolicy",
    "RecursiveCompressor", "CompressionResult",
    "TemporalMotifTracker", "TemporalEncoder",
]
