from __future__ import annotations

import struct
from typing import List, Tuple

import zstandard as zstd

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.token.tokenizer import Tokenizer, WhitespaceTokenizer

MAGIC = b"UTOK"  # 4 bytes

# Tokenizer ID registry
_TOKENIZER_IDS = {
    "whitespace": 0,
    "tiktoken-cl100k_base": 1,
    "tiktoken-o200k_base": 2,
}

_ID_TO_NAME = {v: k for k, v in _TOKENIZER_IDS.items()}


def _delta_encode(values: List[int]) -> List[int]:
    """Delta-code a list of ints: [a, b, c] -> [a, b-a, c-b]."""
    if not values:
        return []
    deltas = [values[0]]
    for i in range(1, len(values)):
        deltas.append(values[i] - values[i - 1])
    return deltas


def _delta_decode(deltas: List[int]) -> List[int]:
    """Undo delta coding."""
    if not deltas:
        return []
    values = [deltas[0]]
    for i in range(1, len(deltas)):
        values.append(values[-1] + deltas[i])
    return values


def _encode_signed_varints(deltas: List[int]) -> bytes:
    """Encode signed deltas using zigzag + uvarint."""
    out = bytearray()
    for d in deltas:
        # Zigzag encoding: map signed to unsigned
        z = (d << 1) ^ (d >> 63) if d >= 0 else ((d << 1) ^ (d >> 63)) & 0xFFFFFFFFFFFFFFFF
        # Simpler zigzag
        if d >= 0:
            z = d * 2
        else:
            z = (-d) * 2 - 1
        out += encode_uvarint(z)
    return bytes(out)


def _decode_signed_varints(data: bytes, count: int, offset: int = 0) -> Tuple[List[int], int]:
    """Decode zigzag-encoded signed varints."""
    values = []
    off = offset
    for _ in range(count):
        z, off = decode_uvarint(data, off)
        # Zigzag decode
        if z & 1:
            values.append(-((z + 1) >> 1))
        else:
            values.append(z >> 1)
    return values, off


def token_encode(
    text: str,
    tokenizer: Tokenizer | None = None,
    level: int = 10,
) -> bytes:
    """
    Encode text at token granularity.

    Wire format:
        UTOK (4B) + tokenizer_id (1B) + num_tokens (uvarint)
        + compressed_len (u32) + zstd(zigzag_delta_coded_token_ids)
    """
    if tokenizer is None:
        tokenizer = WhitespaceTokenizer()

    tok_id = _TOKENIZER_IDS.get(tokenizer.name, 0)
    tokens = tokenizer.encode(text)
    deltas = _delta_encode(tokens)
    raw_deltas = _encode_signed_varints(deltas)

    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(raw_deltas)

    out = bytearray(MAGIC)
    out += struct.pack("B", tok_id)
    out += encode_uvarint(len(tokens))
    out += struct.pack("<I", len(compressed))
    out += compressed
    return bytes(out)


def token_decode(
    blob: bytes,
    tokenizer: Tokenizer | None = None,
) -> str:
    """
    Decode a UTOK blob back to text.

    If tokenizer is None, uses WhitespaceTokenizer (only works if that was
    the encoder's tokenizer too).
    """
    if len(blob) < 4:
        raise ValueError("token: blob too small")
    if blob[:4] != MAGIC:
        raise ValueError("token: bad magic")

    off = 4
    tok_id = blob[off]
    off += 1
    num_tokens, off = decode_uvarint(blob, off)
    comp_len = struct.unpack_from("<I", blob, off)[0]
    off += 4
    compressed = blob[off:off + comp_len]

    dctx = zstd.ZstdDecompressor()
    raw_deltas = dctx.decompress(compressed)

    deltas, _ = _decode_signed_varints(raw_deltas, num_tokens)
    tokens = _delta_decode(deltas)

    if tokenizer is None:
        tokenizer = WhitespaceTokenizer()

    return tokenizer.decode(tokens)
