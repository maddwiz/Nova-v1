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
    decode_h1m1_select_params,
)


MAGIC = b"TPF2"
VERSION = 1


@dataclass
class PF2Packet:
    offset: int
    length: int
    eids_sorted: List[int]


@dataclass
class PF2Index:
    templates_map: Dict[int, str]
    dict_bytes: bytes
    packets: List[PF2Packet]


@dataclass
class PF2Meta:
    blob_bytes: int
    packet_count: int
    packet_events: int
    template_bytes: int
    dict_bytes: int


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


def _safe_train_dict(samples: List[bytes], target_dict_size: int) -> bytes:
    """
    Robust zstd dictionary training:
      - auto-shrinks dict size to fit sample bytes
      - retries smaller sizes if libzstd rejects training buffer
      - returns b"" if training fails (fallback to no-dict)
    """
    if zstd is None:
        raise RuntimeError("zstandard missing (pip install zstandard)")

    clean = [s for s in samples if isinstance(s, (bytes, bytearray)) and len(s) > 0]
    if not clean:
        return b""

    total = sum(len(s) for s in clean)
    if total < 8192:
        # too small to train anything meaningful
        return b""

    # dict must be meaningfully smaller than training set
    # (this also avoids ZstdError: Src size is incorrect)
    max_allowed = max(1024, total // 8)

    ds = int(target_dict_size)
    ds = min(ds, max_allowed)
    ds = max(ds, 1024)

    # Try a few shrinking passes
    for _ in range(6):
        try:
            trainer = zstd.train_dictionary(ds, clean)
            return trainer.as_bytes()
        except Exception:
            ds = ds // 2
            if ds < 1024:
                break

    return b""


def build_tpl_pf2_blob(
    events: List[Tuple[int, List[str]]],
    unknown_lines: List[str],
    template_csv_text: str,
    packet_events: int = 32768,
    zstd_level: int = 10,
    dict_train_packets: int = 3,
    dict_size: int = 65536,
) -> Tuple[bytes, PF2Meta]:
    """
    PF2 = PF1 packetized recall + OPTIONAL shared Zstd dictionary.

    Layout:
      MAGIC 'TPF2'
      u32 VERSION
      u32 zstd_level
      u32 packet_events
      u32 dict_len
      dict_bytes
      u32 template_csv_len
      template_csv_bytes
      uvarint packet_count
      For each packet:
        u32 offset
        u32 length
        eidset_bytes (len+payload)
      packet_bytes...
    """
    tpl_bytes = template_csv_text.encode("utf-8", errors="replace")

    raw_samples: List[bytes] = []
    raw_structs: List[bytes] = []
    eidsets: List[bytes] = []

    i = 0
    n = len(events)
    pkt_index = 0

    while i < n:
        chunk = events[i:i + packet_events]
        i += packet_events

        ul = unknown_lines if pkt_index == 0 else []
        raw_struct = encode_template_channels_v1_mask(chunk, ul)
        raw_structs.append(raw_struct)

        if pkt_index < dict_train_packets:
            raw_samples.append(raw_struct)

        eids = [eid for eid, _p in chunk]
        eidsets.append(_encode_eidset(eids))

        pkt_index += 1

    # ---- train dict safely (may return empty bytes)
    dict_bytes = _safe_train_dict(raw_samples, dict_size)

    # ---- compressor setup
    if zstd is None:
        raise RuntimeError("zstandard missing (pip install zstandard)")

    if dict_bytes:
        zd = zstd.ZstdCompressionDict(dict_bytes)
        cctx = zstd.ZstdCompressor(level=zstd_level, dict_data=zd)
    else:
        cctx = zstd.ZstdCompressor(level=zstd_level)

    packets_comp: List[bytes] = []
    for raw_struct in raw_structs:
        packets_comp.append(cctx.compress(raw_struct))

    # ---- build blob
    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", VERSION)
    out += struct.pack("<I", int(zstd_level))
    out += struct.pack("<I", int(packet_events))

    out += struct.pack("<I", len(dict_bytes))
    out += dict_bytes

    out += struct.pack("<I", len(tpl_bytes))
    out += tpl_bytes

    out += _uvarint_encode(len(packets_comp))

    table_start = len(out)
    for pkt, eidset in zip(packets_comp, eidsets):
        out += struct.pack("<I", 0)        # offset placeholder
        out += struct.pack("<I", len(pkt)) # length
        out += _bytes_encode(eidset)

    offsets: List[int] = []
    for pkt in packets_comp:
        offsets.append(len(out))
        out += pkt

    # patch offsets
    off = table_start
    for po in offsets:
        out[off:off + 4] = struct.pack("<I", po)
        off += 4
        _plen = struct.unpack("<I", out[off:off + 4])[0]
        off += 4
        _eidset, off = _bytes_decode(out, off)

    meta = PF2Meta(
        blob_bytes=len(out),
        packet_count=len(packets_comp),
        packet_events=packet_events,
        template_bytes=len(tpl_bytes),
        dict_bytes=len(dict_bytes),
    )
    return bytes(out), meta


def build_pf2_index(blob: bytes) -> PF2Index:
    if blob[:4] != MAGIC:
        raise ValueError("bad PF2 magic")

    off = 4
    _ver = struct.unpack("<I", blob[off:off + 4])[0]
    off += 4

    _zlvl = struct.unpack("<I", blob[off:off + 4])[0]
    off += 4

    _pkt_events = struct.unpack("<I", blob[off:off + 4])[0]
    off += 4

    dict_len = struct.unpack("<I", blob[off:off + 4])[0]
    off += 4
    dict_bytes = blob[off:off + dict_len]
    off += dict_len

    tpl_len = struct.unpack("<I", blob[off:off + 4])[0]
    off += 4
    tpl_bytes = blob[off:off + tpl_len]
    off += tpl_len

    templates_map = load_template_map_from_csv_text(
        tpl_bytes.decode("utf-8", errors="replace")
    )

    pkt_count, off = _uvarint_decode(blob, off)

    packets: List[PF2Packet] = []
    for _ in range(pkt_count):
        pkt_off = struct.unpack("<I", blob[off:off + 4])[0]
        off += 4
        pkt_len = struct.unpack("<I", blob[off:off + 4])[0]
        off += 4
        eidset_bytes, off = _bytes_decode(blob, off)
        eids_sorted = _decode_eidset(eidset_bytes)
        packets.append(PF2Packet(offset=int(pkt_off), length=int(pkt_len), eids_sorted=eids_sorted))

    return PF2Index(templates_map=templates_map, dict_bytes=dict_bytes, packets=packets)


def recall_event_id_pf2(index: PF2Index, blob: bytes, event_id: int, limit: int = 50) -> List[str]:
    if zstd is None:
        raise RuntimeError("zstandard missing (pip install zstandard)")

    if index.dict_bytes:
        zd = zstd.ZstdCompressionDict(index.dict_bytes)
        dctx = zstd.ZstdDecompressor(dict_data=zd)
    else:
        dctx = zstd.ZstdDecompressor()

    eid = int(event_id)
    hits: List[str] = []
    tpl = index.templates_map.get(eid, "")

    for pkt in index.packets:
        if not _sorted_contains(pkt.eids_sorted, eid):
            continue

        comp = blob[pkt.offset:pkt.offset + pkt.length]
        raw = dctx.decompress(comp)

        hit_params = decode_h1m1_select_params(raw, eid)
        if not hit_params:
            continue

        for params in hit_params:
            if tpl:
                hits.append(render_template(tpl, params))
            else:
                hits.append(f"E{eid} " + " | ".join(params))
            if len(hits) >= limit:
                return hits

    return hits
