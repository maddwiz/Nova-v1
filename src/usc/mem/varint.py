from typing import Tuple


def encode_uvarint(n: int) -> bytes:
    """
    Unsigned varint (LEB128 style).
    Small numbers use 1 byte. Bigger numbers use more.
    """
    if n < 0:
        raise ValueError("uvarint cannot be negative")

    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def decode_uvarint(data: bytes, offset: int = 0) -> Tuple[int, int]:
    """
    Returns (value, new_offset)
    """
    shift = 0
    result = 0
    i = offset

    while True:
        if i >= len(data):
            raise ValueError("uvarint truncated")

        b = data[i]
        i += 1

        result |= (b & 0x7F) << shift

        if (b & 0x80) == 0:
            break

        shift += 7
        if shift > 63:
            raise ValueError("uvarint too large")

    return result, i
