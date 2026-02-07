import zstandard as zstd


def zstd_compress(data: bytes, level: int = 10) -> bytes:
    c = zstd.ZstdCompressor(level=level)
    return c.compress(data)


def zstd_decompress(data: bytes) -> bytes:
    d = zstd.ZstdDecompressor()
    return d.decompress(data)
