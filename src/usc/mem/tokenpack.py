import gzip
import re
from typing import Dict, List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint


MAGIC = b"USCTP1"  # Unified State Codec TokenPack v0.1 (binary)


_TOKEN_RE = re.compile(r"\w+|[^\w\s]|\s+")


def tokenize(text: str) -> List[str]:
    """
    Tokenize into:
    - words (\\w+)
    - punctuation ([^\\w\\s])
    - whitespace (\\s+)

    This preserves exact formatting for lossless reconstruction.
    """
    return _TOKEN_RE.findall(text)


def build_token_table(chunks: List[str]) -> Tuple[List[str], Dict[str, int]]:
    """
    Shared token dictionary across all chunks.
    """
    table: List[str] = []
    index: Dict[str, int] = {}

    for ch in chunks:
        for tok in tokenize(ch):
            if tok not in index:
                index[tok] = len(table)
                table.append(tok)

    return table, index


def _pack_table(table: List[str]) -> bytes:
    out = bytearray()
    out += encode_uvarint(len(table))
    for tok in table:
        b = tok.encode("utf-8")
        out += encode_uvarint(len(b))
        out += b
    return bytes(out)


def _unpack_table(data: bytes, offset: int) -> Tuple[List[str], int]:
    n, off = decode_uvarint(data, offset)
    table: List[str] = []
    for _ in range(n):
        ln_len, off = decode_uvarint(data, off)
        b = data[off : off + ln_len]
        off += ln_len
        table.append(b.decode("utf-8"))
    return table, off


def _pack_chunks(chunks: List[str], idx: Dict[str, int]) -> bytes:
    """
    Store chunks as token-index sequences.
    """
    out = bytearray()
    out += encode_uvarint(len(chunks))

    for ch in chunks:
        toks = tokenize(ch)
        out += encode_uvarint(len(toks))
        for t in toks:
            out += encode_uvarint(idx[t])

    return bytes(out)


def _unpack_chunks(data: bytes, offset: int, table: List[str]) -> Tuple[List[str], int]:
    num_chunks, off = decode_uvarint(data, offset)
    out: List[str] = []

    for _ in range(num_chunks):
        n_toks, off = decode_uvarint(data, off)
        buf = []
        for _ in range(n_toks):
            ix, off = decode_uvarint(data, off)
            buf.append(table[ix])
        out.append("".join(buf))

    return out, off


def encode_chunks_with_tokentable(chunks: List[str]) -> bytes:
    """
    Lossless word/token-level shared dictionary compression.
    """
    table, idx = build_token_table(chunks)

    raw = bytearray()
    raw += MAGIC
    raw += _pack_table(table)
    raw += _pack_chunks(chunks, idx)

    return gzip.compress(bytes(raw), compresslevel=9)


def decode_chunks_with_tokentable(packet_bytes: bytes) -> List[str]:
    raw = gzip.decompress(packet_bytes)

    if not raw.startswith(MAGIC):
        raise ValueError("Not a TOKENPACK v0.1 packet")

    off = len(MAGIC)
    table, off = _unpack_table(raw, off)
    chunks, off = _unpack_chunks(raw, off, table)
    return chunks
