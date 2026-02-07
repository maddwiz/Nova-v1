import hashlib
from typing import List


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]


def make_fingerprint(header: str, goal: str, witnesses: List[str]) -> str:
    """
    v0.5 fingerprint:
    A tiny behavior-id derived from the meaning spine.
    Later this can include probe outputs, tool-call args, etc.
    """
    blob = header + "\n" + goal + "\n" + "\n".join(witnesses)
    return _h(blob)
