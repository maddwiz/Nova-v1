from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set, Optional, Tuple

import json

from usc.mem.sas_dict_token_v1 import (
    MAGIC_SASD,
    decode_sas_packets_to_lines,
)

from usc.mem.sas_keyword_index_v0 import (
    SASKeywordIndex,
    build_keyword_index,
    query_packets_for_keywords,
    _stem_lite,
)

from usc.api.odc2_sharded_v0 import (
    odc2s_decode_selected_blocks,
    packet_indices_to_block_ids,
)


@dataclass
class RecallResult:
    matched_lines: List[str]
    selected_packets: int
    selected_blocks: int
    total_blocks: int


def _normalize_keywords(keywords: Set[str]) -> Set[str]:
    return {k.strip().lower() for k in keywords if k.strip()}


def _keyword_match_terms(k: str, prefix_len: int = 5) -> List[str]:
    out: List[str] = []
    k = k.strip().lower()
    if not k:
        return out

    out.append(k)

    ks = _stem_lite(k)
    if ks and ks not in out:
        out.append(ks)

    if prefix_len and len(k) >= prefix_len:
        p = k[:prefix_len]
        if p not in out:
            out.append(p)

    if prefix_len and len(ks) >= prefix_len:
        ps = ks[:prefix_len]
        if ps not in out:
            out.append(ps)

    return out


def _try_parse_tool_from_line(line: str) -> Optional[str]:
    if "tool_call::" not in line:
        return None
    try:
        after = line.split("tool_call::", 1)[1]
        tool = after.split(" ", 1)[0].strip()
        return tool if tool else None
    except Exception:
        return None


def _try_parse_payload_from_line(line: str) -> Optional[dict]:
    if "payload=" not in line:
        return None
    try:
        p = line.split("payload=", 1)[1].strip()
        return json.loads(p)
    except Exception:
        return None


def _flatten_payload(payload: Any, out: List[Tuple[str, Any]], prefix: str = "") -> None:
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


def _match_structured_tokens(kws: Set[str], tool: Optional[str], payload: Optional[dict]) -> bool:
    # Tool match
    for k in kws:
        if k.startswith("tool:"):
            want = k.split("tool:", 1)[1].strip()
            if not want:
                return False
            if tool is None or tool.lower() != want.lower():
                return False

    needs_payload = any(k.startswith(("k:", "v:", "kv:", "n:")) for k in kws)
    if not needs_payload:
        return True

    if payload is None:
        return False

    flat: List[Tuple[str, Any]] = []
    _flatten_payload(payload, flat)

    keys = {kp.lower() for kp, _ in flat if kp}
    str_values = {str(v).lower() for _, v in flat if isinstance(v, str)}

    kv_map: Dict[str, List[Any]] = {}
    for kp, v in flat:
        if kp:
            kv_map.setdefault(kp.lower(), []).append(v)

    for k in kws:
        if k.startswith("k:"):
            want_k = k.split("k:", 1)[1].strip().lower()
            if want_k not in keys:
                return False

        elif k.startswith("v:"):
            want_v = k.split("v:", 1)[1].strip().lower()
            if want_v not in str_values:
                return False

        elif k.startswith("kv:"):
            rest = k.split("kv:", 1)[1].strip()
            if "=" not in rest:
                return False
            kp, vv = rest.split("=", 1)
            kp = kp.strip().lower()
            vv = vv.strip().lower()

            if kp not in kv_map:
                return False

            ok = False
            for cur in kv_map[kp]:
                if str(cur).strip().lower() == vv:
                    ok = True
                    break
            if not ok:
                return False

        elif k.startswith("n:"):
            rest = k.split("n:", 1)[1].strip()
            if "=" not in rest:
                return False
            kp, vv = rest.split("=", 1)
            kp = kp.strip().lower()
            vv = vv.strip()

            if kp not in kv_map:
                return False

            try:
                want_num = float(vv) if "." in vv else int(vv)
            except Exception:
                return False

            ok = False
            for cur in kv_map[kp]:
                if isinstance(cur, (int, float, bool)):
                    if float(cur) == float(want_num):
                        ok = True
                        break
            if not ok:
                return False

    return True


def _match_plain_keywords(kws_plain: Set[str], line: str, prefix_len: int) -> bool:
    s = line.lower()
    for kw in kws_plain:
        terms = _keyword_match_terms(kw, prefix_len=prefix_len)
        if not any(t in s for t in terms):
            return False
    return True


def recall_from_packets_decoded(
    packets_part: List[bytes],
    keywords: Set[str],
    require_all: bool = True,
    prefix_len: int = 5,
    selected_packets: int = 0,
    selected_blocks: int = 0,
    total_blocks: int = 0,
) -> RecallResult:
    """
    Recall directly from *already decoded* packet bytes list.
    packets_part must include dict packet at index 0.
    """
    kws = _normalize_keywords(keywords)
    if not kws or not packets_part:
        return RecallResult(matched_lines=[], selected_packets=0, selected_blocks=0, total_blocks=0)

    if not packets_part[0].startswith(MAGIC_SASD):
        raise ValueError("recall_from_packets_decoded: missing dict packet")

    lines = decode_sas_packets_to_lines(packets_part)

    kws_struct = {k for k in kws if k.startswith(("tool:", "k:", "v:", "kv:", "n:"))}
    kws_plain = {k for k in kws if k not in kws_struct}

    out: List[str] = []
    for ln in lines:
        tool = _try_parse_tool_from_line(ln)
        payload = _try_parse_payload_from_line(ln)

        if require_all:
            if kws_struct and not _match_structured_tokens(kws_struct, tool, payload):
                continue
            if kws_plain and not _match_plain_keywords(kws_plain, ln, prefix_len=prefix_len):
                continue
            out.append(ln)
        else:
            ok = False
            if kws_struct and _match_structured_tokens(kws_struct, tool, payload):
                ok = True
            if not ok and kws_plain:
                for kw in kws_plain:
                    terms = _keyword_match_terms(kw, prefix_len=prefix_len)
                    if any(t in ln.lower() for t in terms):
                        ok = True
                        break
            if ok:
                out.append(ln)

    return RecallResult(
        matched_lines=out,
        selected_packets=int(selected_packets),
        selected_blocks=int(selected_blocks),
        total_blocks=int(total_blocks),
    )


def recall_from_odc2s(
    blob: bytes,
    packets_all: List[bytes],
    keywords: Set[str],
    kwi: Optional[SASKeywordIndex] = None,
    group_size: int = 2,
    require_all: bool = True,
    prefix_len: int = 5,
) -> RecallResult:
    """
    Normal recall path:
      - bloom packet prefilter
      - decode selected blocks
      - filter via decoded lines
    """
    kws = _normalize_keywords(keywords)
    if not kws or not packets_all:
        return RecallResult(matched_lines=[], selected_packets=0, selected_blocks=0, total_blocks=0)

    if kwi is None:
        kwi = build_keyword_index(
            packets_all,
            m_bits=2048,
            k_hashes=4,
            include_tool_names=True,
            include_raw_lines=True,
            include_payload_fields=True,
            enable_stem=True,
            prefix_len=prefix_len,
        )

    want_packet_indices = query_packets_for_keywords(
        kwi,
        packets_all,
        kws,
        enable_stem=True,
        prefix_len=prefix_len,
        require_all=require_all,
    )

    block_ids = packet_indices_to_block_ids(want_packet_indices, group_size)

    # Always include dict block
    dict_block = next(iter(packet_indices_to_block_ids({0}, group_size)))
    block_ids.add(dict_block)

    packets_part, meta = odc2s_decode_selected_blocks(blob, block_ids=block_ids)

    # Ensure dict packet first
    if not packets_part:
        packets_part = [packets_all[0]]
    else:
        if not packets_part[0].startswith(MAGIC_SASD):
            packets_part = [packets_all[0]] + packets_part

    return recall_from_packets_decoded(
        packets_part=packets_part,
        keywords=keywords,
        require_all=require_all,
        prefix_len=prefix_len,
        selected_packets=len(want_packet_indices),
        selected_blocks=len(block_ids),
        total_blocks=meta.block_count,
    )
