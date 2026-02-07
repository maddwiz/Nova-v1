from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Any, Tuple

import hashlib
import re
import json

from usc.mem.sas_dict_token_v1 import (
    MAGIC_SASD,
    MAGIC_SASA,
    _decode_dict_packet,
)

from usc.mem.sas_dict_token_v1 import decode_sas_packets_to_lines

RE_WORD = re.compile(r"[A-Za-z0-9_]{3,}")


def _hash64(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)


def _k_hashes(h64: int, k: int, m_bits: int) -> List[int]:
    out: List[int] = []
    x = h64 & ((1 << 64) - 1)
    for i in range(k):
        x = (x * 0x9E3779B97F4A7C15 + (i + 1) * 0xD1B54A32D192ED03) & ((1 << 64) - 1)
        out.append(int(x % m_bits))
    return out


def _set_bit(bits: bytearray, pos: int) -> None:
    bits[pos >> 3] |= (1 << (pos & 7))


def _get_bit(bits: bytes, pos: int) -> bool:
    return (bits[pos >> 3] >> (pos & 7)) & 1 == 1


def _tokenize_text(s: str) -> List[str]:
    return [w.lower() for w in RE_WORD.findall(s)]


def _stem_lite(word: str) -> str:
    w = word.lower()
    if len(w) <= 3:
        return w
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w


def _variants_for_keyword(kw: str, enable_stem: bool, prefix_len: int) -> Set[str]:
    out = set()
    k = kw.strip().lower()
    if not k:
        return out

    out.add(k)

    if enable_stem:
        out.add(_stem_lite(k))

    if prefix_len and len(k) >= prefix_len:
        out.add(f"pref:{k[:prefix_len]}")
        if enable_stem:
            ks = _stem_lite(k)
            if len(ks) >= prefix_len:
                out.add(f"pref:{ks[:prefix_len]}")
    return out


def _flatten_payload(payload: object, out: List[Tuple[str, object]], prefix: str = "") -> None:
    if isinstance(payload, dict):
        for k, v in payload.items():
            kp = f"{prefix}.{k}" if prefix else str(k)
            _flatten_payload(v, out, kp)
    elif isinstance(payload, list):
        for i, v in enumerate(payload):
            kp = f"{prefix}[{i}]"
            _flatten_payload(v, out, kp)
    else:
        out.append((prefix, payload))


def _try_parse_payload_from_line(line: str) -> Optional[dict]:
    """
    Expected tool line shape:
      tool_call::X rid=... payload={...}
    """
    if "payload=" not in line:
        return None
    try:
        p = line.split("payload=", 1)[1].strip()
        return json.loads(p)
    except Exception:
        return None


def _try_parse_tool_from_line(line: str) -> Optional[str]:
    if "tool_call::" not in line:
        return None
    try:
        after = line.split("tool_call::", 1)[1]
        tool = after.split(" ", 1)[0].strip()
        return tool if tool else None
    except Exception:
        return None


@dataclass
class SASKeywordIndex:
    m_bits: int
    k_hashes: int
    packet_blooms: List[bytes]
    total_packets: int
    keyword_df: Dict[str, int]


def build_keyword_index(
    packets: List[bytes],
    m_bits: int = 2048,
    k_hashes: int = 4,
    include_tool_names: bool = True,
    include_raw_lines: bool = True,
    include_payload_fields: bool = True,
    enable_stem: bool = True,
    prefix_len: int = 5,
) -> SASKeywordIndex:
    if not packets:
        return SASKeywordIndex(m_bits=m_bits, k_hashes=k_hashes, packet_blooms=[], total_packets=0, keyword_df={})

    if not packets[0].startswith(MAGIC_SASD):
        raise ValueError("Not a SAS dict-token v1 stream")

    # Validate dict packet
    _ = _decode_dict_packet(packets[0])

    if m_bits % 8 != 0:
        raise ValueError("m_bits must be multiple of 8")

    bloom_bytes = m_bits // 8
    packet_blooms: List[bytes] = []
    keyword_df: Dict[str, int] = {}

    # Iterate each data packet; decode lines for bulletproof parsing
    for pi in range(1, len(packets)):
        pkt = packets[pi]
        if not pkt.startswith(MAGIC_SASA):
            packet_blooms.append(bytes(bloom_bytes))
            continue

        bits = bytearray(bloom_bytes)
        seen_in_packet: Set[str] = set()

        # Decode ONLY this packet + dict
        lines = decode_sas_packets_to_lines([packets[0], pkt])

        for ln in lines:
            lnl = ln.lower()

            # index raw words for plain queries
            if include_raw_lines:
                for w in _tokenize_text(ln):
                    for v in _variants_for_keyword(w, enable_stem=enable_stem, prefix_len=prefix_len):
                        seen_in_packet.add(v)

            # tool indexing
            if include_tool_names:
                tool = _try_parse_tool_from_line(ln)
                if tool:
                    seen_in_packet.add(f"tool:{tool}".lower())

            # payload indexing
            if include_payload_fields:
                payload = _try_parse_payload_from_line(ln)
                if payload is not None:
                    flat: List[Tuple[str, object]] = []
                    _flatten_payload(payload, flat)

                    for keypath, val in flat:
                        if not keypath:
                            continue

                        kp = keypath.lower()
                        seen_in_packet.add(f"k:{kp}")

                        if isinstance(val, str):
                            vv = val.strip().lower()
                            if vv:
                                seen_in_packet.add(f"v:{vv}")
                                seen_in_packet.add(f"kv:{kp}={vv}")
                        elif isinstance(val, (int, float, bool)):
                            seen_in_packet.add(f"kv:{kp}={val}")
                            seen_in_packet.add(f"n:{kp}={int(val) if isinstance(val, bool) else val}")

        # Set bloom bits and update DF
        for tok in seen_in_packet:
            h64 = _hash64(tok)
            for pos in _k_hashes(h64, k_hashes, m_bits):
                _set_bit(bits, pos)
            keyword_df[tok] = keyword_df.get(tok, 0) + 1

        packet_blooms.append(bytes(bits))

    return SASKeywordIndex(
        m_bits=int(m_bits),
        k_hashes=int(k_hashes),
        packet_blooms=packet_blooms,
        total_packets=len(packets),
        keyword_df=keyword_df,
    )


def query_packets_for_keywords(
    kwi: SASKeywordIndex,
    packets: List[bytes],
    keywords: Set[str],
    enable_stem: bool = True,
    prefix_len: int = 5,
    require_all: bool = True,
) -> Set[int]:
    if not keywords or not packets:
        return set()

    kw_groups: List[Set[str]] = []
    for k in keywords:
        k = k.strip().lower()
        if not k:
            continue

        if any(k.startswith(pfx) for pfx in ("k:", "v:", "kv:", "n:", "tool:")):
            kw_groups.append({k})
        else:
            kw_groups.append(_variants_for_keyword(k, enable_stem=enable_stem, prefix_len=prefix_len))

    if not kw_groups:
        return set()

    variant_positions: Dict[str, List[int]] = {}
    for group in kw_groups:
        for v in group:
            if v not in variant_positions:
                variant_positions[v] = _k_hashes(_hash64(v), kwi.k_hashes, kwi.m_bits)

    out: Set[int] = set()

    for j, bits in enumerate(kwi.packet_blooms):
        pi = j + 1
        if pi >= len(packets):
            break

        if require_all:
            ok = True
            for group in kw_groups:
                g_ok = False
                for v in group:
                    pos_list = variant_positions[v]
                    if all(_get_bit(bits, pos) for pos in pos_list):
                        g_ok = True
                        break
                if not g_ok:
                    ok = False
                    break
            if ok:
                out.add(pi)
        else:
            hit = False
            for group in kw_groups:
                for v in group:
                    pos_list = variant_positions[v]
                    if all(_get_bit(bits, pos) for pos in pos_list):
                        hit = True
                        break
                if hit:
                    break
            if hit:
                out.add(pi)

    return out


def keywords_to_tool_keyword(tool_name: str) -> str:
    return f"tool:{tool_name}".lower()
