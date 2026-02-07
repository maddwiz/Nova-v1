import re
import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple
from bisect import bisect_left

try:
    import zstandard as zstd
except Exception:
    zstd = None

from usc.api.hdfs_template_codec_v1_channels_mask import encode_template_channels_v1_mask
from usc.mem.tpl_pf1_recall_v1 import (
    _uvarint_encode,
    _uvarint_decode,
    _bytes_encode,
    _bytes_decode,
    load_template_map_from_csv_text,
    render_template,
)


MAGIC = b"TPQ1"
VERSION = 1


# ----------------------------
# Bloom filter (simple + fast)
# ----------------------------

def _hash32(s: str, seed: int) -> int:
    # FNV-1a-ish
    h = 2166136261 ^ seed
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def bloom_make(bits: int) -> bytearray:
    return bytearray((bits + 7) // 8)


def bloom_add(bloom: bytearray, bits: int, k: int, token: str):
    t = token.lower()
    for i in range(k):
        h = _hash32(t, 0x9E3779B9 + i * 0x85EBCA6B)
        pos = h % bits
        bloom[pos // 8] |= (1 << (pos % 8))


def bloom_has_all(bloom: bytes, bits: int, k: int, tokens: List[str]) -> bool:
    for token in tokens:
        t = token.lower()
        for i in range(k):
            h = _hash32(t, 0x9E3779B9 + i * 0x85EBCA6B)
            pos = h % bits
            if not (bloom[pos // 8] & (1 << (pos % 8))):
                return False
    return True


# ----------------------------
# Tokenization
# ----------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]{2,}")

def tokenize_line(line: str) -> List[str]:
    return _WORD_RE.findall(line.lower())


# ----------------------------
# Minimal FULL decode of H1M1 raw_struct
# (for scanning candidate packets after bloom hit)
# ----------------------------

def _zigzag_decode(u: int) -> int:
    return (u >> 1) ^ (-(u & 1))


def _decode_int_stream(payload: bytes) -> List[str]:
    off = 0
    n, off = _uvarint_decode(payload, off)
    out: List[str] = []
    if n == 0:
        return out
    first_u, off = _uvarint_decode(payload, off)
    prev = _zigzag_decode(first_u)
    out.append(str(prev))
    for _ in range(n - 1):
        du, off = _uvarint_decode(payload, off)
        prev = prev + _zigzag_decode(du)
        out.append(str(prev))
    return out


def _decode_raw_stream(payload: bytes) -> List[str]:
    off = 0
    n, off = _uvarint_decode(payload, off)
    out: List[str] = []
    for _ in range(n):
        b, off = _bytes_decode(payload, off)
        out.append(b.decode("utf-8", errors="replace"))
    return out


def _decode_hex_stream(payload: bytes) -> List[str]:
    off = 0
    n, off = _uvarint_decode(payload, off)
    out: List[str] = []
    for _ in range(n):
        b, off = _bytes_decode(payload, off)
        out.append(b.hex())
    return out


def _decode_ip_stream(payload: bytes) -> List[str]:
    off = 0
    n, off = _uvarint_decode(payload, off)
    out: List[str] = []
    for _ in range(n):
        if off + 4 > len(payload):
            raise ValueError("ip decode overflow")
        b = payload[off:off+4]
        off += 4
        out.append(".".join(str(x) for x in b))
    return out


def _decode_dict_stream(payload: bytes) -> List[str]:
    off = 0
    nvals, off = _uvarint_decode(payload, off)
    vocab_n, off = _uvarint_decode(payload, off)
    vocab: List[str] = []
    for _ in range(vocab_n):
        b, off = _bytes_decode(payload, off)
        vocab.append(b.decode("utf-8", errors="replace"))
    out: List[str] = []
    for _ in range(nvals):
        idx, off = _uvarint_decode(payload, off)
        if vocab:
            out.append(vocab[min(idx, len(vocab)-1)])
        else:
            out.append("")
    return out


def decode_h1m1_all_events(raw: bytes) -> Tuple[List[Tuple[int, List[str]]], List[str]]:
    if raw[:4] != b"H1M1":
        raise ValueError("bad H1M1 magic")

    off = 4
    _ver = struct.unpack("<I", raw[off:off+4])[0]
    off += 4

    n_events, off = _uvarint_decode(raw, off)
    n_unknown, off = _uvarint_decode(raw, off)
    max_params, off = _uvarint_decode(raw, off)

    eids: List[int] = []
    for _ in range(n_events):
        eid, off = _uvarint_decode(raw, off)
        eids.append(int(eid))

    params_mat: List[List[str]] = [[""] * max_params for _ in range(n_events)]

    for chan in range(max_params):
        mask_len, off = _uvarint_decode(raw, off)
        mask = raw[off:off+mask_len]
        off += mask_len

        ctype, off = _uvarint_decode(raw, off)
        payload, off = _bytes_decode(raw, off)

        if ctype == 1:
            vals = _decode_int_stream(payload)
        elif ctype == 2:
            vals = _decode_hex_stream(payload)
        elif ctype == 3:
            vals = _decode_ip_stream(payload)
        elif ctype == 4:
            vals = _decode_dict_stream(payload)
        else:
            vals = _decode_raw_stream(payload)

        vi = 0
        for row in range(n_events):
            if mask[row // 8] & (1 << (row % 8)):
                if vi < len(vals):
                    params_mat[row][chan] = vals[vi]
                vi += 1

    unknown: List[str] = []
    for _ in range(n_unknown):
        b, off = _bytes_decode(raw, off)
        unknown.append(b.decode("utf-8", errors="replace"))

    events = [(eids[i], params_mat[i]) for i in range(n_events)]
    return events, unknown


# ----------------------------
# Packet index structs
# ----------------------------

@dataclass
class PFQ1Packet:
    offset: int
    length: int
    eids_sorted: List[int]
    bloom: bytes


@dataclass
class PFQ1Index:
    templates_map: Dict[int, str]
    bloom_bits: int
    bloom_k: int
    packets: List[PFQ1Packet]


@dataclass
class PFQ1Meta:
    blob_bytes: int
    packet_count: int
    packet_events: int
    template_bytes: int
    bloom_bits: int
    bloom_k: int


def _encode_eidset(eids: List[int]) -> bytes:
    s = sorted(set(int(x) for x in eids))
    out = bytearray()
    out += _uvarint_encode(len(s))
    prev = 0
    for x in s:
        out += _uvarint_encode(x - prev)
        prev = x
    return bytes(out)


def _decode_eidset(buf: bytes) -> List[int]:
    off = 0
    n, off = _uvarint_decode(buf, off)
    out: List[int] = []
    prev = 0
    for _ in range(n):
        d, off = _uvarint_decode(buf, off)
        prev += int(d)
        out.append(prev)
    return out


def _sorted_contains(arr: List[int], x: int) -> bool:
    i = bisect_left(arr, x)
    return i < len(arr) and arr[i] == x


def _zstd_compress(b: bytes, level: int) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard missing")
    return zstd.ZstdCompressor(level=level).compress(b)


def _zstd_decompress(b: bytes) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard missing")
    return zstd.ZstdDecompressor().decompress(b)


# ----------------------------
# Build PFQ1 blob
# ----------------------------

def build_pfq1_blob(
    events: List[Tuple[int, List[str]]],
    unknown_lines: List[str],
    template_csv_text: str,
    packet_events: int = 32768,
    zstd_level: int = 10,
    bloom_bits: int = 8192,
    bloom_k: int = 4,
) -> Tuple[bytes, PFQ1Meta]:
    """
    PFQ1 = packetized template codec + per-packet keyword bloom index.

    Layout:
      MAGIC 'TPQ1'
      u32 VERSION
      u32 zstd_level
      u32 packet_events
      u32 bloom_bits
      u32 bloom_k
      u32 template_csv_len
      template_csv_bytes
      uvarint packet_count
      For each packet:
        u32 offset
        u32 length
        eidset_bytes (len+payload)
        bloom_bytes (len+payload)
      packet_bytes...
    """
    tpl_bytes = template_csv_text.encode("utf-8", errors="replace")
    tmap = load_template_map_from_csv_text(template_csv_text)

    packets_comp: List[bytes] = []
    eidsets: List[bytes] = []
    blooms: List[bytes] = []

    i = 0
    n = len(events)
    pkt_idx = 0


    # UNKNOWN_ONLY_PACKET_MODE:
    # If events are empty but unknown_lines exist, we still build ONE packet
    # so HOT remains queryable on real logs without template coverage.
    if n == 0 and unknown_lines:
        raw_struct = encode_template_channels_v1_mask([], unknown_lines)
        comp = _zstd_compress(raw_struct, level=zstd_level)

        packets_comp.append(comp)
        eidsets.append(_encode_eidset([]))

        b = bloom_make(bloom_bits)
        for ln in unknown_lines:
            for tok in tokenize_line(ln):
                bloom_add(b, bloom_bits, bloom_k, tok)
        blooms.append(bytes(b))

        n = 1  # force table build + offsets patching

    while i < n:
        chunk = events[i:i+packet_events]
        i += packet_events

        # unknown lines only in first packet for compactness
        ul = unknown_lines if pkt_idx == 0 else []
        raw_struct = encode_template_channels_v1_mask(chunk, ul)
        comp = _zstd_compress(raw_struct, level=zstd_level)
        packets_comp.append(comp)

        eids = [eid for eid, _p in chunk]
        eidsets.append(_encode_eidset(eids))

        # keyword bloom built from rendered lines (fast + good enough)
        # ✅ unknown_lines ALSO indexed into bloom (critical for real logs)
        # Many real datasets have the important text in unknown_lines, not templates.
        b = bloom_make(bloom_bits)
        for eid, params in chunk:
            tpl = tmap.get(int(eid), "")
            if tpl:
                line = render_template(tpl, params)
            else:
                line = f"E{eid} " + " ".join(params)
            toks = tokenize_line(line)
            for tok in toks:
                bloom_add(b, bloom_bits, bloom_k, tok)
        # add unknown_lines tokens into bloom (only present in pkt_idx==0)
        for ln in ul:
            for tok in tokenize_line(ln):
                bloom_add(b, bloom_bits, bloom_k, tok)

        blooms.append(bytes(b))

        pkt_idx += 1

    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", VERSION)
    out += struct.pack("<I", int(zstd_level))
    out += struct.pack("<I", int(packet_events))
    out += struct.pack("<I", int(bloom_bits))
    out += struct.pack("<I", int(bloom_k))
    out += struct.pack("<I", len(tpl_bytes))
    out += tpl_bytes
    out += _uvarint_encode(len(packets_comp))

    table_start = len(out)
    for pkt, eidset, bl in zip(packets_comp, eidsets, blooms):
        out += struct.pack("<I", 0)         # offset placeholder
        out += struct.pack("<I", len(pkt))  # length
        out += _bytes_encode(eidset)
        out += _bytes_encode(bl)

    offsets: List[int] = []
    for pkt in packets_comp:
        offsets.append(len(out))
        out += pkt

    # patch offsets
    off = table_start
    for po in offsets:
        out[off:off+4] = struct.pack("<I", po)
        off += 4
        _plen = struct.unpack("<I", out[off:off+4])[0]
        off += 4
        _eidset, off = _bytes_decode(out, off)
        _bloom, off = _bytes_decode(out, off)

    meta = PFQ1Meta(
        blob_bytes=len(out),
        packet_count=len(packets_comp),
        packet_events=packet_events,
        template_bytes=len(tpl_bytes),
        bloom_bits=bloom_bits,
        bloom_k=bloom_k,
    )
    return bytes(out), meta


def build_pfq1_index(blob: bytes) -> PFQ1Index:
    if blob[:4] != MAGIC:
        raise ValueError("bad PFQ1 magic")

    off = 4
    ver = struct.unpack("<I", blob[off:off+4])[0]
    off += 4
    if ver != 1:
        raise ValueError("wrong PFQ1 version")

    _zlvl = struct.unpack("<I", blob[off:off+4])[0]
    off += 4
    _pkt_events = struct.unpack("<I", blob[off:off+4])[0]
    off += 4

    bloom_bits = struct.unpack("<I", blob[off:off+4])[0]
    off += 4
    bloom_k = struct.unpack("<I", blob[off:off+4])[0]
    off += 4

    tpl_len = struct.unpack("<I", blob[off:off+4])[0]
    off += 4
    tpl_bytes = blob[off:off+tpl_len]
    off += tpl_len
    templates_map = load_template_map_from_csv_text(tpl_bytes.decode("utf-8", errors="replace"))

    packet_count, off = _uvarint_decode(blob, off)

    packets: List[PFQ1Packet] = []
    for _ in range(packet_count):
        pkt_off = struct.unpack("<I", blob[off:off+4])[0]
        off += 4
        pkt_len = struct.unpack("<I", blob[off:off+4])[0]
        off += 4
        eidset_bytes, off = _bytes_decode(blob, off)
        bloom_bytes, off = _bytes_decode(blob, off)

        eids_sorted = _decode_eidset(eidset_bytes)
        packets.append(PFQ1Packet(offset=int(pkt_off), length=int(pkt_len), eids_sorted=eids_sorted, bloom=bloom_bytes))

    return PFQ1Index(templates_map=templates_map, bloom_bits=int(bloom_bits), bloom_k=int(bloom_k), packets=packets)


# ----------------------------
# Query
# ----------------------------

def query_keywords(
    index: PFQ1Index,
    blob: bytes,
    query: str,
    limit: int = 50,
    require_all_terms: bool = True,
) -> List[str]:
    """
    Keyword search:
      - bloom filters packets
      - decompress candidate packets
      - render + substring filter

    query: "receiveBlock IOException"
    """
    terms = tokenize_line(query)
    if not terms:
        return []

    hits: List[str] = []
    terms_lc = [t.lower() for t in terms]

    for pkt in index.packets:
        # fast filter using bloom
        if not bloom_has_all(pkt.bloom, index.bloom_bits, index.bloom_k, terms_lc):
            continue

        comp = blob[pkt.offset:pkt.offset+pkt.length]
        raw = _zstd_decompress(comp)

        events, unknown_lines = decode_h1m1_all_events(raw)

        for eid, params in events:
            tpl = index.templates_map.get(int(eid), "")
            if tpl:
                line = render_template(tpl, params)
            else:
                line = f"E{eid} " + " ".join(params)

            s = line.lower()
            if require_all_terms:
                ok = all(t in s for t in terms_lc)
            else:
                ok = any(t in s for t in terms_lc)

            if ok:
                hits.append(line)
                if len(hits) >= limit:
                    return hits


        # ✅ scan unknown_lines for matches too (critical for real logs)
        for ln in unknown_lines:
            s2 = ln.lower()
            if require_all_terms:
                ok2 = all(t in s2 for t in terms_lc)
            else:
                ok2 = any(t in s2 for t in terms_lc)

            if ok2:
                hits.append(ln)
                if len(hits) >= limit:
                    return hits

    return hits
