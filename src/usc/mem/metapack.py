from typing import List, Tuple

from usc.mem.dictpack import encode_chunks_with_table, decode_chunks_with_table
from usc.mem.templatepack import encode_chunks_with_templates, decode_chunks_with_templates
from usc.mem.hybridpack import encode_chunks_hybrid, decode_chunks_hybrid
from usc.mem.templatemtf import encode_chunks_with_template_mtf, decode_chunks_with_template_mtf
from usc.mem.templatemtf_bits import (
    encode_chunks_with_template_mtf_bits,
    decode_chunks_with_template_mtf_bits,
)
from usc.mem.templatemtf_bits_deltaonly import (
    encode_chunks_with_template_mtf_bits_deltaonly,
    decode_chunks_with_template_mtf_bits_deltaonly,
)
from usc.mem.templatemtf_bits_deltaonly_canon import (
    encode_chunks_with_template_mtf_bits_deltaonly_canon,
    decode_chunks_with_template_mtf_bits_deltaonly_canon,
)


MAGIC = b"M"  # MetaPack v0.7 (includes TMTFDO_CAN)


# method ids
METHOD_DICTPACK = 1
METHOD_TEMPLATEPACK = 2
METHOD_HYBRIDPACK = 3
METHOD_TMTF = 4
METHOD_TMTFB = 6
METHOD_TMTFDO = 7
METHOD_TMTFDO_CAN = 8


def encode_chunks_metapack(chunks: List[str]) -> bytes:
    """
    MetaPack v0.7:
    - Try multiple packers
    - Pick the smallest output
    - Store: 1 byte MAGIC + 1 byte method_id + payload_bytes
    """
    candidates: List[Tuple[int, bytes]] = []

    dp = encode_chunks_with_table(chunks)
    candidates.append((METHOD_DICTPACK, dp))

    tp = encode_chunks_with_templates(chunks)
    candidates.append((METHOD_TEMPLATEPACK, tp))

    hp = encode_chunks_hybrid(chunks)
    candidates.append((METHOD_HYBRIDPACK, hp))

    mtf = encode_chunks_with_template_mtf(chunks)
    candidates.append((METHOD_TMTF, mtf))

    mtfb = encode_chunks_with_template_mtf_bits(chunks)
    candidates.append((METHOD_TMTFB, mtfb))

    mtfdo = encode_chunks_with_template_mtf_bits_deltaonly(chunks)
    candidates.append((METHOD_TMTFDO, mtfdo))

    mtfdo_can = encode_chunks_with_template_mtf_bits_deltaonly_canon(chunks)
    candidates.append((METHOD_TMTFDO_CAN, mtfdo_can))

    best_method, best_payload = min(candidates, key=lambda x: len(x[1]))

    return MAGIC + bytes([best_method]) + best_payload


def decode_chunks_metapack(packet_bytes: bytes) -> List[str]:
    if len(packet_bytes) < 2 or packet_bytes[:1] != MAGIC:
        raise ValueError("Not a METAPACK packet")

    method_id = packet_bytes[1]
    payload = packet_bytes[2:]

    if method_id == METHOD_DICTPACK:
        return decode_chunks_with_table(payload)

    if method_id == METHOD_TEMPLATEPACK:
        return decode_chunks_with_templates(payload)

    if method_id == METHOD_HYBRIDPACK:
        return decode_chunks_hybrid(payload)

    if method_id == METHOD_TMTF:
        return decode_chunks_with_template_mtf(payload)

    if method_id == METHOD_TMTFB:
        return decode_chunks_with_template_mtf_bits(payload)

    if method_id == METHOD_TMTFDO:
        return decode_chunks_with_template_mtf_bits_deltaonly(payload)

    if method_id == METHOD_TMTFDO_CAN:
        return decode_chunks_with_template_mtf_bits_deltaonly_canon(payload)

    raise ValueError("Unknown METAPACK method id")
