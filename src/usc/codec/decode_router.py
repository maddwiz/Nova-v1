from __future__ import annotations
from pathlib import Path

def sniff_magic(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read(4)

def decode_auto(in_path: str, out_path: str) -> str:
    """
    Auto-detect container format and route to correct decoder.
    Returns the detected mode label.
    """
    p = Path(in_path)
    magic = sniff_magic(p)
    blob = p.read_bytes()

    # STREAM (USST container) -> magic = USST
    if magic == b"USST":
        from usc.api.stream_codec_v3d_auto import decode_stream_auto
        lines = decode_stream_auto(blob)
        Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return "stream"

    # PF3 (HOT-LITE-FULL)  -> magic = TPF3
    if magic == b"TPF3":
        from usc.mem.tpl_pf3_decode_v1_h1m2 import decode_pf3_h1m2_to_lines
        lines = decode_pf3_h1m2_to_lines(blob)
        Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return "hot-lite-full"

    # HOT (USCH + PFQ1) -> magic = USCH
    if magic == b"USCH":
        raise SystemExit("HOT decode not wired yet (USCH). Next step will add it.")

    # COLD (USCC bundle) -> magic = USCC
    if magic == b"USCC":
        raise SystemExit("COLD decode (USCC) not yet wired. Use USZR fallback mode for decodable cold.")

    # USZR: raw zstd fallback
    if magic == b"USZR":
        try:
            import zstandard as _zstd
        except ImportError:
            raise SystemExit("zstandard required for USZR decode")
        text = _zstd.ZstdDecompressor().decompress(blob[4:]).decode("utf-8", errors="replace")
        Path(out_path).write_text(text, encoding="utf-8")
        return "cold-zstd"

    # USBR: raw brotli fallback
    if magic == b"USBR":
        try:
            import brotli
        except ImportError:
            raise SystemExit("brotli required for USBR decode (pip install brotli)")
        text = brotli.decompress(blob[4:]).decode("utf-8", errors="replace")
        Path(out_path).write_text(text, encoding="utf-8")
        return "cold-brotli"

    # USBZ: raw bzip2 fallback
    if magic == b"USBZ":
        import bz2
        text = bz2.decompress(blob[4:]).decode("utf-8", errors="replace")
        Path(out_path).write_text(text, encoding="utf-8")
        return "cold-bzip2"

    # USZD: trained-dictionary zstd
    if magic == b"USZD":
        import struct as struct_mod
        try:
            import zstandard as _zstd
        except ImportError:
            raise SystemExit("zstandard required for USZD decode")
        dict_len = struct_mod.unpack("<I", blob[4:8])[0]
        dict_bytes = blob[8:8 + dict_len]
        compressed = blob[8 + dict_len:]
        dict_data = _zstd.ZstdCompressionDict(dict_bytes)
        dctx = _zstd.ZstdDecompressor(dict_data=dict_data)
        text = dctx.decompress(compressed).decode("utf-8", errors="replace")
        Path(out_path).write_text(text, encoding="utf-8")
        return "cold-zstd-dict"

    raise SystemExit(f"Unknown file magic: {magic!r} (expected USST / TPF3 / USCH / USCC / USZR / USBR / USBZ / USZD)")
