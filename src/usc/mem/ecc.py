import hashlib
from typing import List


def _checksum(parts: List[str]) -> str:
    """
    Light ECC: one checksum over the truth spine.
    We use sha256 but truncate heavily to reduce overhead.
    """
    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:10]  # tiny


def make_ecc(header: str, goal: str, witnesses: List[str]) -> str:
    """
    Returns a tiny checksum string.
    """
    return _checksum([header, goal] + witnesses)


def verify_ecc(ecc_checksum: str, header: str, goal: str, witnesses: List[str]) -> bool:
    """
    Recompute and compare.
    """
    return ecc_checksum == make_ecc(header, goal, witnesses)
