import re
from typing import List, Tuple

from usc.mem.tpl_pf1_recall_v1 import build_pf1_index, recall_event_id_index, PF1Index


_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]{2,}")


def _tokenize(q: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(q.lower())]


def find_candidate_eids_by_template_any(index: PF1Index, query: str, max_candidates: int = 32) -> List[int]:
    """
    OR-mode candidate selection (robust):
    - pick templates that contain ANY query token
    - score by count of matched tokens
    - return top-N candidates

    This fixes cases where miners wildcard important words (e.g. 'IOException').
    """
    terms = _tokenize(query)
    if not terms:
        return []

    scored: List[Tuple[int, int]] = []  # (score, eid)
    for eid, tpl in index.templates_map.items():
        s = tpl.lower()
        score = 0
        for t in terms:
            if t in s:
                score += 1
        if score > 0:
            scored.append((score, int(eid)))

    # Highest score first (more terms matched)
    scored.sort(reverse=True)
    return [eid for _score, eid in scored[:max_candidates]]


def query_fast_pf1(
    pf1_blob: bytes,
    query: str,
    limit: int = 50,
    max_candidates: int = 32,
) -> Tuple[List[str], List[int]]:
    """
    FAST keyword search using:
      templates → candidate EventIDs (OR + scoring) → selective recall → filter full query on rendered lines

    Returns (hits, candidate_eids).
    """
    idx = build_pf1_index(pf1_blob)
    cands = find_candidate_eids_by_template_any(idx, query, max_candidates=max_candidates)
    if not cands:
        return [], []

    terms = _tokenize(query)
    if not terms:
        return [], cands

    hits: List[str] = []
    for eid in cands:
        lines = recall_event_id_index(idx, pf1_blob, eid, limit=limit)
        for ln in lines:
            s = ln.lower()
            if all(t in s for t in terms):
                hits.append(ln)
                if len(hits) >= limit:
                    return hits, cands

    return hits, cands
