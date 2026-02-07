from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional


_WILDCARD_RE = re.compile(r"(\<\*\>|\[\*\])")

_TS_FULL = re.compile(r"\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\b")
_TS_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _normalize_template(tpl: str) -> str:
    s = (tpl or "").strip()
    s = _TS_FULL.sub("<*>", s)
    s = _TS_DATE.sub("<*>", s)
    return s


def _parse_event_id(eid_raw: str) -> Optional[int]:
    """
    LogHub templates often use EventId like:
      - "E1", "E2", ...
      - or numeric strings
      - sometimes hex-ish
    We convert to a stable int ID for packing.
    """
    if not eid_raw:
        return None
    eid_raw = eid_raw.strip()

    # Pure integer
    try:
        return int(eid_raw)
    except Exception:
        pass

    # "E123" format
    if (len(eid_raw) >= 2) and (eid_raw[0] in ("E", "e")) and eid_raw[1:].isdigit():
        return int(eid_raw[1:])

    # Hex-ish (0x...)
    if eid_raw.lower().startswith("0x"):
        try:
            return int(eid_raw, 16)
        except Exception:
            pass

    return None


def _compile_template_regex(template: str) -> Tuple[re.Pattern, int, str]:
    """
    Compile template to regex capturing wildcards.
    Returns: (regex, wildcard_count, anchor_literal)
    """
    t = _normalize_template(template)

    parts = _WILDCARD_RE.split(t)
    out = ["^"]
    wildcards = 0

    anchor = ""
    for p in parts:
        if not p:
            continue
        if p in ("<*>", "[*]"):
            out.append("(.*?)")
            wildcards += 1
        else:
            out.append(re.escape(p))
            lit = p.strip()
            if len(lit) > len(anchor):
                anchor = lit

    out.append("$")
    rx = re.compile("".join(out))
    return rx, wildcards, anchor


@dataclass
class CompiledTemplate:
    event_id: int
    rx: re.Pattern
    wildcard_count: int
    anchor: str


class HDFSTemplateBank:
    def __init__(self, compiled: List[CompiledTemplate]):
        self.compiled = compiled

    @classmethod
    def from_csv(cls, path: Path | str) -> "HDFSTemplateBank":
        p = Path(path)
        compiled: List[CompiledTemplate] = []

        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                eid_raw = (row.get("EventId") or "").strip()
                tpl = (row.get("EventTemplate") or "").strip()
                if not eid_raw or not tpl:
                    continue

                eid = _parse_event_id(eid_raw)
                if eid is None:
                    continue

                rx, wc, anchor = _compile_template_regex(tpl)
                compiled.append(CompiledTemplate(event_id=eid, rx=rx, wildcard_count=wc, anchor=anchor))

        # Prefer more specific templates first
        compiled.sort(key=lambda x: (x.wildcard_count, -len(x.anchor)))

        return cls(compiled)


def parse_hdfs_lines(lines: List[str], bank: HDFSTemplateBank) -> Tuple[List[Tuple[int, List[str]]], List[str]]:
    events: List[Tuple[int, List[str]]] = []
    unknown: List[str] = []

    for ln in lines:
        s = (ln or "").rstrip("\n")
        hit = False

        for ct in bank.compiled:
            if ct.anchor and ct.anchor not in s:
                continue
            m = ct.rx.match(s)
            if not m:
                continue
            params = list(m.groups()) if m.groups() else []
            events.append((ct.event_id, params))
            hit = True
            break

        if not hit:
            unknown.append(s)

    return events, unknown


def parse_hdfs_lines_rows(
    lines: List[str],
    bank: HDFSTemplateBank,
) -> Tuple[List[Optional[Tuple[int, List[str]]]], List[str]]:
    rows: List[Optional[Tuple[int, List[str]]]] = []
    unknown_lines: List[str] = []

    for ln in lines:
        s = (ln or "").rstrip("\n")
        matched: Optional[Tuple[int, List[str]]] = None

        for ct in bank.compiled:
            if ct.anchor and ct.anchor not in s:
                continue
            m = ct.rx.match(s)
            if not m:
                continue
            params = list(m.groups()) if m.groups() else []
            matched = (ct.event_id, params)
            break

        if matched is None:
            rows.append(None)
            unknown_lines.append(s)
        else:
            rows.append(matched)

    return rows, unknown_lines
