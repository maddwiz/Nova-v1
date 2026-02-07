from __future__ import annotations

from typing import List, Optional, Tuple
import math


# -----------------------------
# uvarint helpers (self-contained)
# -----------------------------
def _uvarint_encode(x: int) -> bytes:
    out = bytearray()
    n = int(x)
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _uvarint_decode(data: bytes, off: int = 0) -> Tuple[int, int]:
    shift = 0
    x = 0
    while True:
        if off >= len(data):
            raise ValueError("uvarint decode past end")
        b = data[off]
        off += 1
        x |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
        if shift > 63:
            raise ValueError("uvarint too large")
    return x, off


def _bytes_encode(b: bytes) -> bytes:
    return _uvarint_encode(len(b)) + b


def _bytes_decode(data: bytes, off: int) -> Tuple[bytes, int]:
    n, off = _uvarint_decode(data, off)
    end = off + n
    if end > len(data):
        raise ValueError("bytes decode past end")
    return data[off:end], end


# -----------------------------
# H1M2 Rowmask Blob Spec (v1)
# -----------------------------
# Encodes a fixed-length chunk of rows preserving order:
# rows: List[Optional[(event_id:int, params:List[str])]]
#
# Format:
#   n_rows(uvarint)
#   mask_bytes_len(uvarint)
#   mask_bytes[mask_bytes_len]          # bit=1 => templated row, bit=0 => unknown row
#   templated_rows:
#       for each bit==1 in row order:
#           event_id(uvarint)
#           n_params(uvarint)
#           param_i: bytes(len+utf8)
#   unknown_rows:
#       for each bit==0 in row order:
#           line: bytes(len+utf8)
#
def encode_h1m2_rowmask_blob(
    rows: List[Optional[Tuple[int, List[str]]]],
    unknown_lines: List[str],
) -> bytes:
    n = len(rows)
    mask_len = (n + 7) // 8
    mask = bytearray(mask_len)

    # build mask + count unknown needed
    need_unknown = 0
    for i, r in enumerate(rows):
        if r is not None:
            mask[i // 8] |= (1 << (i % 8))
        else:
            need_unknown += 1

    if need_unknown != len(unknown_lines):
        # safety: allow encode even if caller got counts wrong
        # but we will only encode min available
        unknown_lines = unknown_lines[:need_unknown]

    out = bytearray()
    out += _uvarint_encode(n)
    out += _uvarint_encode(mask_len)
    out += bytes(mask)

    # templated rows payload
    for r in rows:
        if r is None:
            continue
        eid, params = r
        out += _uvarint_encode(int(eid))
        out += _uvarint_encode(len(params))
        for p in params:
            out += _bytes_encode((p or "").encode("utf-8", errors="ignore"))

    # unknown rows payload (in order)
    uidx = 0
    for r in rows:
        if r is not None:
            continue
        line = unknown_lines[uidx] if uidx < len(unknown_lines) else ""
        uidx += 1
        out += _bytes_encode((line or "").encode("utf-8", errors="ignore"))

    return bytes(out)


def decode_h1m2_rowmask_blob(
    blob: bytes,
) -> Tuple[List[Optional[Tuple[int, List[str]]]], List[str]]:
    off = 0
    n_rows, off = _uvarint_decode(blob, off)
    mask_len, off = _uvarint_decode(blob, off)
    if off + mask_len > len(blob):
        raise ValueError("rowmask: mask past end")
    mask = blob[off : off + mask_len]
    off += mask_len

    rows: List[Optional[Tuple[int, List[str]]]] = [None] * n_rows

    # decode templated rows first
    templ_positions = []
    unknown_positions = []
    for i in range(n_rows):
        bit = (mask[i // 8] >> (i % 8)) & 1
        if bit == 1:
            templ_positions.append(i)
        else:
            unknown_positions.append(i)

    for pos in templ_positions:
        eid, off = _uvarint_decode(blob, off)
        n_params, off = _uvarint_decode(blob, off)
        params: List[str] = []
        for _ in range(n_params):
            b, off = _bytes_decode(blob, off)
            params.append(b.decode("utf-8", errors="ignore"))
        rows[pos] = (int(eid), params)

    unknown_lines: List[str] = []
    for pos in unknown_positions:
        b, off = _bytes_decode(blob, off)
        line = b.decode("utf-8", errors="ignore")
        unknown_lines.append(line)
        rows[pos] = None

    return rows, unknown_lines
