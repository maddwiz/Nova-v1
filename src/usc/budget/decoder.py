"""
N11: Budgeted Decode — decode only what fits within resource budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from usc.zerocopy.lazy_packet import LazyBlob


@dataclass
class PartialResult:
    """Result of a budgeted decode."""
    packets: List[bytes] = field(default_factory=list)
    total_packets: int = 0
    decoded_packets: int = 0
    total_bytes_decoded: int = 0
    budget_exhausted: bool = False


def budgeted_decode(
    blob: bytes,
    max_bytes: int = 1024,
    max_packets: int = 10,
) -> PartialResult:
    """
    Decode packets from a blob within resource constraints.

    Stops when either max_bytes or max_packets is reached.
    Uses LazyBlob (N4) for packet-level granularity.
    """
    lazy = LazyBlob(blob)
    total = lazy.num_packets()

    result = PartialResult(total_packets=total)
    bytes_so_far = 0

    for i in range(total):
        if result.decoded_packets >= max_packets:
            result.budget_exhausted = True
            break

        pkt = lazy.packet(i)

        if bytes_so_far + len(pkt) > max_bytes and result.decoded_packets > 0:
            result.budget_exhausted = True
            break

        result.packets.append(pkt)
        result.decoded_packets += 1
        bytes_so_far += len(pkt)
        result.total_bytes_decoded = bytes_so_far

    return result
