from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import heapq

from usc.mem.varint import encode_uvarint, decode_uvarint


@dataclass(order=True)
class _Node:
    freq: int
    sym: Optional[int] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


def build_codebook(freqs: Dict[int, int]) -> Dict[int, str]:
    """
    Build Huffman codebook: symbol -> bitstring like '0101'
    """
    heap: List[_Node] = []
    for sym, f in freqs.items():
        heapq.heappush(heap, _Node(f, sym=sym))

    if not heap:
        return {}

    # Special case: only one symbol
    if len(heap) == 1:
        only = heap[0].sym
        return {only: "0"}

    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, _Node(a.freq + b.freq, left=a, right=b))

    root = heap[0]

    codes: Dict[int, str] = {}

    def dfs(node: _Node, path: str):
        if node.sym is not None:
            codes[node.sym] = path
            return
        dfs(node.left, path + "0")
        dfs(node.right, path + "1")

    dfs(root, "")
    return codes


def pack_bits(bitstring: str) -> bytes:
    """
    Pack '010101' into bytes.
    """
    out = bytearray()
    cur = 0
    nbits = 0
    for ch in bitstring:
        cur = (cur << 1) | (1 if ch == "1" else 0)
        nbits += 1
        if nbits == 8:
            out.append(cur)
            cur = 0
            nbits = 0

    if nbits > 0:
        cur = cur << (8 - nbits)
        out.append(cur)

    return bytes(out)


def unpack_bits(data: bytes, bit_len: int) -> str:
    """
    Unpack bytes into exactly bit_len bits.
    """
    bits = []
    count = 0
    for b in data:
        for i in range(7, -1, -1):
            if count >= bit_len:
                break
            bits.append("1" if (b >> i) & 1 else "0")
            count += 1
        if count >= bit_len:
            break
    return "".join(bits)


def encode_huffman(symbols: List[int]) -> bytes:
    """
    Encode list of ints with Huffman coding.
    Output format:
      [nsym] [sym,freq]... [bit_len] [packed_bits]
    """
    if not symbols:
        return encode_uvarint(0) + encode_uvarint(0) + b""

    freqs: Dict[int, int] = {}
    for s in symbols:
        freqs[s] = freqs.get(s, 0) + 1

    codes = build_codebook(freqs)

    bitstring = "".join(codes[s] for s in symbols)
    packed = pack_bits(bitstring)

    out = bytearray()
    out += encode_uvarint(len(freqs))
    for sym, f in freqs.items():
        out += encode_uvarint(sym)
        out += encode_uvarint(f)

    out += encode_uvarint(len(bitstring))
    out += encode_uvarint(len(packed))
    out += packed
    return bytes(out)


def decode_huffman(data: bytes, offset: int) -> Tuple[List[int], int]:
    """
    Decode Huffman payload starting at offset.
    Returns: (symbols, new_offset)
    """
    nsym, off = decode_uvarint(data, offset)
    if nsym == 0:
        bit_len, off = decode_uvarint(data, off)  # stored as 0
        return [], off

    freqs: Dict[int, int] = {}
    total = 0
    for _ in range(nsym):
        sym, off = decode_uvarint(data, off)
        f, off = decode_uvarint(data, off)
        freqs[sym] = f
        total += f

    bit_len, off = decode_uvarint(data, off)
    packed_len, off = decode_uvarint(data, off)
    packed = data[off : off + packed_len]
    off += packed_len

    codes = build_codebook(freqs)
    # invert
    inv: Dict[str, int] = {v: k for k, v in codes.items()}

    bitstring = unpack_bits(packed, bit_len)

    out_syms: List[int] = []
    cur = ""
    for ch in bitstring:
        cur += ch
        if cur in inv:
            out_syms.append(inv[cur])
            cur = ""
            if len(out_syms) == total:
                break

    return out_syms, off
