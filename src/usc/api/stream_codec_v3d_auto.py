"""
USC Stream Codec Wrapper — v3d9 (Drain3 + persistent dictionary)

Correct contract discovered from traces:

- build_dict_state_from_chunks(state, chunks, ...)
- encode_dict_packet(state)
- encode_data_packet(state, chunks, ...)   # mines ALL chunks internally

So we must call encode_data_packet ONCE (NOT per chunk).
"""

from __future__ import annotations

from typing import List
import importlib
import inspect
import struct

DEFAULT_MODULE = "usc.mem.stream_proto_canz_v3d9_slots_bitpack"
MAGIC_STREAM = b"USST"


def _load_mod(module_name: str = DEFAULT_MODULE):
    return importlib.import_module(module_name)


def _chunk_texts(lines: List[str], chunk_lines: int) -> List[str]:
    """
    v3d9 wants chunks as newline-terminated strings.
    """
    if chunk_lines <= 0:
        chunk_lines = 250

    chunks: List[str] = []
    buf: List[str] = []

    for ln in lines:
        buf.append(ln)
        if len(buf) >= chunk_lines:
            chunks.append("\n".join(buf) + "\n")
            buf = []

    if buf:
        chunks.append("\n".join(buf) + "\n")

    return chunks


def _call_only_kwargs(fn, **kwargs):
    """
    Call fn with only kwargs it accepts.
    """
    params = inspect.signature(fn).parameters
    filt = {k: v for k, v in kwargs.items() if k in params}
    return fn(**filt)


def encode_stream_auto(
    lines: List[str],
    *,
    module_name: str = DEFAULT_MODULE,
    chunk_lines: int = 250,
    zstd_level: int = 10,
) -> bytes:
    mod = _load_mod(module_name)

    # Create state
    state = mod.StreamStateV3D9()

    # Chunk into strings
    chunks = _chunk_texts(lines, chunk_lines)

    # Build dict state once
    _call_only_kwargs(
        mod.build_dict_state_from_chunks,
        state=state,
        chunks=chunks,
        chunk_lines=chunk_lines,
        zstd_level=zstd_level,
    )

    # Dict packet
    dict_pkt = _call_only_kwargs(mod.encode_dict_packet, state=state)
    if not isinstance(dict_pkt, (bytes, bytearray)):
        raise RuntimeError("encode_dict_packet returned non-bytes")

    # Data packet ONCE (mines all chunks internally)
    data_pkt = _call_only_kwargs(
        mod.encode_data_packet,
        state=state,
        chunks=chunks,
        zstd_level=zstd_level,
    )
    if not isinstance(data_pkt, (bytes, bytearray)):
        raise RuntimeError("encode_data_packet returned non-bytes")

    # USST container: magic + dict_pkt_len + dict_pkt + data_pkt
    dict_bytes = bytes(dict_pkt)
    data_bytes = bytes(data_pkt)
    return MAGIC_STREAM + struct.pack("<I", len(dict_bytes)) + dict_bytes + data_bytes


def decode_stream_auto(
    blob: bytes,
    *,
    module_name: str = DEFAULT_MODULE,
) -> List[str]:
    """Decode a USST stream blob back to lines."""
    if blob[:4] != MAGIC_STREAM:
        raise ValueError(f"not a USST stream blob (got {blob[:4]!r})")

    dict_len = struct.unpack("<I", blob[4:8])[0]
    dict_pkt = blob[8:8 + dict_len]
    data_pkt = blob[8 + dict_len:]

    mod = _load_mod(module_name)
    state = mod.StreamStateV3D9()
    mod.apply_dict_packet(dict_pkt, state)
    chunks = mod.decode_data_packet(data_pkt, state)

    lines: List[str] = []
    for chunk in chunks:
        for ln in chunk.splitlines():
            lines.append(ln)
    return lines
