import json
import time
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class CommitRecord:
    """
    A committed memory is a "known-good" decoded result.
    This is how we prevent drift over time.
    """
    ts: float
    packet_version: str
    used_tier: int
    confidence: float
    text_fingerprint: str
    text: str


def _fingerprint_text(text: str) -> str:
    """
    Simple stable fingerprint of decoded text (not cryptographic security,
    just a stable identity).
    """
    # cheap + deterministic
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def commit_memory(
    store_path: str,
    packet_version: str,
    used_tier: int,
    confidence: float,
    decoded_text: str,
) -> CommitRecord:
    rec = CommitRecord(
        ts=time.time(),
        packet_version=packet_version,
        used_tier=used_tier,
        confidence=confidence,
        text_fingerprint=_fingerprint_text(decoded_text),
        text=decoded_text,
    )

    with open(store_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    return rec


def load_last_commit(store_path: str) -> Optional[CommitRecord]:
    try:
        with open(store_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return None
        obj = json.loads(lines[-1])
        return CommitRecord(**obj)
    except FileNotFoundError:
        return None
