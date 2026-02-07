from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from usc.slots.typed_slots import (
    infer_slot_types,
    TypedSlotsState,
    pack_params_typed,
    unpack_params_typed,
    uvarint_encode,
    uvarint_decode,
)

MAGIC = b"PFT1"  # PF1 Typed Params v1


@dataclass
class PF1TypedParamsBundle:
    """
    Stores typed params for many templates inside a chunk.

    by_tid:
      tid -> (event_count, typed_blob)
    """
    by_tid: Dict[int, Tuple[int, bytes]]


def build_pf1_typed_params(
    tids: List[int],
    params_by_event: List[List[str]],
    int_min_hits: int = 50,
) -> PF1TypedParamsBundle:
    """
    Inputs:
      tids: template_id per event (len N)
      params_by_event: list of params list[str] per event (len N)

    For each template_id:
      - collect its events
      - infer slot types from samples
      - pack typed params
    """
    if len(tids) != len(params_by_event):
        raise ValueError("tids and params_by_event length mismatch")

    events_by_tid: Dict[int, List[List[str]]] = {}
    for tid, params in zip(tids, params_by_event):
        events_by_tid.setdefault(int(tid), []).append(list(params))

    by_tid: Dict[int, Tuple[int, bytes]] = {}

    for tid, events in events_by_tid.items():
        if not events:
            continue

        # Normalize slot counts (pad shorter rows)
        nslots = max((len(ev) for ev in events), default=0)
        if nslots == 0:
            # no params at all for this tid
            st = TypedSlotsState.from_types([])
            blob = pack_params_typed(st, [[] for _ in events])
            by_tid[tid] = (len(events), blob)
            continue

        norm_events: List[List[str]] = []
        samples_by_slot: List[List[str]] = [[] for _ in range(nslots)]

        for ev in events:
            if len(ev) < nslots:
                ev = ev + [""] * (nslots - len(ev))
            else:
                ev = ev[:nslots]

            norm_events.append(ev)

            for si, v in enumerate(ev):
                samples_by_slot[si].append(v)

        slot_types = infer_slot_types(samples_by_slot, int_min_hits=int_min_hits)
        st = TypedSlotsState.from_types(slot_types)

        blob = pack_params_typed(st, norm_events)
        by_tid[tid] = (len(norm_events), blob)

    return PF1TypedParamsBundle(by_tid=by_tid)


def encode_pf1_typed_params(bundle: PF1TypedParamsBundle) -> bytes:
    """
    Binary format:
      MAGIC
      ntids (uvarint)
      for each tid:
        tid (uvarint)
        event_count (uvarint)
        blob_len (uvarint)
        blob bytes
    """
    out = bytearray()
    out += MAGIC
    out += uvarint_encode(len(bundle.by_tid))

    for tid, (ev_count, blob) in bundle.by_tid.items():
        out += uvarint_encode(int(tid))
        out += uvarint_encode(int(ev_count))
        out += uvarint_encode(len(blob))
        out += blob

    return bytes(out)


def decode_pf1_typed_params(data: bytes) -> PF1TypedParamsBundle:
    if not data.startswith(MAGIC):
        raise ValueError("bad PF1 typed params magic")

    i = len(MAGIC)

    ntids, i = uvarint_decode(data, i)
    by_tid: Dict[int, Tuple[int, bytes]] = {}

    for _ in range(ntids):
        tid, i = uvarint_decode(data, i)
        ev_count, i = uvarint_decode(data, i)
        blen, i = uvarint_decode(data, i)
        blob = data[i : i + blen]
        i += blen
        by_tid[int(tid)] = (int(ev_count), blob)

    return PF1TypedParamsBundle(by_tid=by_tid)


def decode_typed_params_for_tid(blob: bytes, event_count: int) -> List[List[str]]:
    """
    Decode typed params for a single tid.
    Returns params rows (strings) in the same order they were packed for that tid.
    """
    _slot_types, events = unpack_params_typed(blob, event_count=int(event_count))
    return events
