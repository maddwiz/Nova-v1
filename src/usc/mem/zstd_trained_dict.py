from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import zstandard as zstd


@dataclass
class ZstdDictBundle:
    dict_bytes: bytes
    cdict: zstd.ZstdCompressionDict
    ddict: zstd.ZstdCompressionDict


def _clean_samples(samples: List[bytes]) -> List[bytes]:
    out: List[bytes] = []
    for s in samples:
        if isinstance(s, (bytes, bytearray)) and len(s) > 0:
            out.append(bytes(s))
    return out


def _candidate_sizes(target: int, total_src: int) -> List[int]:
    """
    More conservative than before.
    Many zstd builds are picky when total_src is small-ish.
    Rule: dict <= 1/32 of total source, then halve down to 256.
    """
    if total_src <= 0:
        return []

    cap = max(256, min(target, total_src // 32))
    if cap < 256:
        return []

    sizes = []
    s = cap
    while s >= 256:
        sizes.append(s)
        s //= 2

    # unique, descending
    seen = set()
    uniq = []
    for x in sizes:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def train_dict(samples: List[bytes], dict_size: int = 8192) -> ZstdDictBundle:
    samples = _clean_samples(samples)
    if not samples:
        raise ValueError("train_dict: samples is empty")

    total_src = sum(len(s) for s in samples)
    sizes = _candidate_sizes(dict_size, total_src)
    if not sizes:
        raise RuntimeError("train_dict: not enough source bytes for dict training")

    last_err: Optional[Exception] = None
    for size in sizes:
        try:
            d = zstd.train_dictionary(size, samples)
            dict_bytes = d.as_bytes()
            cdict = zstd.ZstdCompressionDict(dict_bytes)
            ddict = zstd.ZstdCompressionDict(dict_bytes)
            return ZstdDictBundle(dict_bytes=dict_bytes, cdict=cdict, ddict=ddict)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"train_dict failed: {last_err}")


def compress_plain(data: bytes, level: int = 10) -> bytes:
    cctx = zstd.ZstdCompressor(level=level)
    return cctx.compress(data)


def decompress_plain(data: bytes) -> bytes:
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(data)


def compress_with_dict(data: bytes, bundle: ZstdDictBundle, level: int = 10) -> bytes:
    cctx = zstd.ZstdCompressor(level=level, dict_data=bundle.cdict)
    return cctx.compress(data)


def decompress_with_dict(data: bytes, bundle: ZstdDictBundle) -> bytes:
    dctx = zstd.ZstdDecompressor(dict_data=bundle.ddict)
    return dctx.decompress(data)
