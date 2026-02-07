from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    """
    A chunk is a small piece of memory.
    """
    idx: int
    text: str


def chunk_by_lines(text: str, max_lines: int = 8) -> List[Chunk]:
    """
    Old chunker: fixed number of lines.
    Kept for reference/testing.
    """
    lines = text.splitlines(keepends=True)
    chunks: List[Chunk] = []

    cur: List[str] = []
    idx = 0

    for ln in lines:
        cur.append(ln)
        if len(cur) >= max_lines:
            chunks.append(Chunk(idx=idx, text="".join(cur)))
            idx += 1
            cur = []

    if cur:
        chunks.append(Chunk(idx=idx, text="".join(cur)))

    return chunks


def chunk_by_paragraph(text: str) -> List[Chunk]:
    """
    New chunker: split by blank lines.
    This is MUCH better for agent memory logs.

    Example:
    - truth spine chunk: Project/Goal/Decision/Next/Note
    - recap chunk
    - reminder chunk
    """
    lines = text.splitlines(keepends=True)
    chunks: List[Chunk] = []
    cur: List[str] = []
    idx = 0

    for ln in lines:
        # blank line ends a paragraph
        if ln.strip() == "":
            if cur:
                chunks.append(Chunk(idx=idx, text="".join(cur)))
                idx += 1
                cur = []
            continue

        cur.append(ln)

    if cur:
        chunks.append(Chunk(idx=idx, text="".join(cur)))

    return chunks
