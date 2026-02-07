from typing import List


def probe_truth_spine(header: str, goal: str, witnesses: List[str]) -> bool:
    """
    Chunk-friendly probe.

    Old version expected every decode to start with:
      Project: ...
      Goal: ...
    That breaks for chunked logs (middle chunks won't have that).

    New rule:
    - allow ANY non-empty header/goal
    - if witnesses exist, they must still be valid witness lines
    """
    # Must have some content
    if not header.strip():
        return False
    if not goal.strip():
        return False

    # Witness lines must still look like witnesses
    for w in witnesses:
        if not (w.startswith("Decision:") or w.startswith("Note:")):
            return False

    return True


def confidence_score(tier: int, probe_ok: bool) -> float:
    """
    v1 confidence model:
    - probe must pass
    - tier 3 (lossless) = high confidence
    - tier 0 (tiny) = medium confidence
    """
    if not probe_ok:
        return 0.0

    if tier == 3:
        return 0.95
    if tier == 0:
        return 0.70

    return 0.50
