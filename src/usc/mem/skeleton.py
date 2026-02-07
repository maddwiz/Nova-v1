from dataclasses import dataclass
from typing import List


@dataclass
class Skeleton:
    """
    Minimal truth spine for USC memory.
    Must always produce non-empty header+goal so probes don't fail on chunks.
    """
    header: str
    goal: str


def _nonempty_lines(text: str) -> List[str]:
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if s:
            lines.append(ln.rstrip("\n"))
    return lines


def extract_skeleton(text: str) -> Skeleton:
    """
    v0.9 skeleton extractor

    Rules:
    1) If the first two non-empty lines are 'Project:' and 'Goal:', use them.
    2) Otherwise: use the first 2 non-empty lines as-is.

    IMPORTANT: The skeleton MUST come from the first lines of the text so that
    render_skeleton() produces a prefix that can be cleanly stripped for residual
    coding. Using markers from arbitrary positions would break Tier 3 lossless
    roundtrip — the residual strip assumes skeleton == text prefix.
    """
    lines = _nonempty_lines(text)

    header = lines[0] if len(lines) >= 1 else "Chunk:"
    goal = lines[1] if len(lines) >= 2 else header

    return Skeleton(header=header, goal=goal)


def render_skeleton(sk: Skeleton) -> str:
    """
    Render skeleton into text that can be removed as a prefix for residual coding.
    """
    return f"{sk.header}\n{sk.goal}\n"
