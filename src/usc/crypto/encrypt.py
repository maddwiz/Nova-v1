"""
N3: Field-level encryption for PII slots using AES-256-GCM.

Falls back gracefully if cryptography package is not installed.
"""
from __future__ import annotations

import os
import struct
from typing import List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint

MAGIC = b"UPII"  # 4 bytes


def _require_cryptography():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        return AESGCM
    except ImportError:
        raise ImportError(
            "PII encryption requires the 'cryptography' package. "
            "Install with: pip install usc[crypto]"
        )


def generate_key() -> bytes:
    """Generate a random 256-bit AES key."""
    return os.urandom(32)


def encrypt_fields(
    fields: List[str],
    key: bytes,
    pii_indices: List[int],
) -> bytes:
    """
    Encrypt specific fields by index using AES-256-GCM.

    Wire format:
        UPII (4B)
        nonce (12B)
        n_fields (uvarint)
        For each field:
            is_encrypted (1B): 0x01 if encrypted, 0x00 if not
            [if encrypted]: ciphertext_len (uvarint) + ciphertext (includes 16B tag)
            [if not encrypted]: plaintext_len (uvarint) + plaintext
    """
    AESGCM = _require_cryptography()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    pii_set = set(pii_indices)

    out = bytearray(MAGIC)
    out += nonce
    out += encode_uvarint(len(fields))

    for i, value in enumerate(fields):
        value_bytes = value.encode("utf-8")
        if i in pii_set:
            out += b"\x01"
            ct = aesgcm.encrypt(nonce, value_bytes, None)
            out += encode_uvarint(len(ct))
            out += ct
        else:
            out += b"\x00"
            out += encode_uvarint(len(value_bytes))
            out += value_bytes

    return bytes(out)


def decrypt_fields(blob: bytes, key: bytes) -> List[str]:
    """Decrypt a UPII blob back to field values."""
    if len(blob) < 4:
        raise ValueError("pii: blob too small")
    if blob[:4] != MAGIC:
        raise ValueError("pii: bad magic")

    AESGCM = _require_cryptography()
    aesgcm = AESGCM(key)

    off = 4
    nonce = blob[off:off + 12]
    off += 12

    n_fields, off = decode_uvarint(blob, off)
    fields: List[str] = []

    for _ in range(n_fields):
        is_encrypted = blob[off]
        off += 1
        data_len, off = decode_uvarint(blob, off)
        data = blob[off:off + data_len]
        off += data_len

        if is_encrypted:
            plaintext = aesgcm.decrypt(nonce, data, None)
            fields.append(plaintext.decode("utf-8"))
        else:
            fields.append(data.decode("utf-8"))

    return fields
