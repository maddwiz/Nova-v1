from dataclasses import dataclass
from typing import List


@dataclass
class Witnesses:
    lines: List[str]


def extract_witnesses(text: str) -> Witnesses:
    # For v0: keep any line that starts with "Decision:" or "Note:"
    keep = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith("Decision:") or ln.startswith("Note:"):
            keep.append(ln)
    return Witnesses(lines=keep)
