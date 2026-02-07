from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from usc.mem.zstd_trained_dict import (
    train_dict,
    compress_plain,
    compress_with_dict,
    ZstdDictBundle,
)


@dataclass
class WarmDictResult:
    warmup_packets: int
    warmup_bytes: int
    rest_raw_bytes: int
    rest_compressed_bytes: int
    total_bytes: int
    trained_dict_bytes: int
    used_mode: str  # "dict" or "plain"


def _chunks(blob: bytes, chunk_size: int = 512) -> List[bytes]:
    if not blob:
        return []
    return [blob[i:i + chunk_size] for i in range(0, len(blob), chunk_size)]


def warmdict_compress_packets(
    packets: List[bytes],
    warmup_packets: int = 1,
    dict_target_size: int = 8192,
    level: int = 10,
) -> WarmDictResult:
    """
    WarmDict protocol (sizing):
      - First N packets sent raw (warmup)
      - Both sides train dict from warmup bytes (no dict transmitted)
      - Remaining bytes compressed using dict

    This version is robust:
      - If dict training fails, it falls back to plain zstd on the rest.
    """
    if warmup_packets < 1:
        raise ValueError("warmup_packets must be >= 1")
    if warmup_packets > len(packets):
        warmup_packets = len(packets)

    warmup = packets[:warmup_packets]
    rest = packets[warmup_packets:]

    warmup_blob = b"".join(warmup)
    rest_blob = b"".join(rest)

    bundle: Optional[ZstdDictBundle] = None
    used_mode = "plain"
    dict_bytes = 0

    # Train dict from warmup bytes (best effort)
    try:
        samples = _chunks(warmup_blob, chunk_size=512)
        bundle = train_dict(samples, dict_size=dict_target_size)
        dict_bytes = len(bundle.dict_bytes)
        used_mode = "dict"
    except Exception:
        bundle = None
        dict_bytes = 0
        used_mode = "plain"

    # Compress rest
    if bundle is None:
        rest_comp = compress_plain(rest_blob, level=level)
    else:
        rest_comp = compress_with_dict(rest_blob, bundle, level=level)

    total = len(warmup_blob) + len(rest_comp)

    return WarmDictResult(
        warmup_packets=warmup_packets,
        warmup_bytes=len(warmup_blob),
        rest_raw_bytes=len(rest_blob),
        rest_compressed_bytes=len(rest_comp),
        total_bytes=total,
        trained_dict_bytes=dict_bytes,
        used_mode=used_mode,
    )
