from typing import List, Tuple

from usc.mem.tpl_fast_query_v1 import query_fast_pf1
from usc.mem.tpl_pfq1_query_v1 import build_pfq1_index, query_keywords


def query_router_v1(
    pf1_blob: bytes,
    pfq1_blob: bytes,
    query: str,
    limit: int = 50,
) -> Tuple[List[str], str]:
    """
    Router:
      1) FAST PF1 (template-routed selective recall)
      2) fallback PFQ1 bloom scan if FAST returns no hits

    Returns: (hits, mode)
      mode in {"FAST", "PFQ1"}
    """
    hits, cands = query_fast_pf1(pf1_blob, query, limit=limit)
    if hits:
        return hits, "FAST"

    # fallback
    idx = build_pfq1_index(pfq1_blob)
    hits2 = query_keywords(idx, pfq1_blob, query, limit=limit, require_all_terms=True)
    return hits2, "PFQ1"
