import gzip
from dataclasses import dataclass


@dataclass
class SizeReport:
    raw_bytes: int
    gzip_bytes: int
    usc_bytes: int

    @property
    def usc_ratio(self) -> float:
        return self.raw_bytes / max(1, self.usc_bytes)

    @property
    def gzip_ratio(self) -> float:
        return self.raw_bytes / max(1, self.gzip_bytes)


def bytes_len(s: bytes) -> int:
    return len(s)


def gzip_compress(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=9)
