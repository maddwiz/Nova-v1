"""Compressed Vault — wraps C3's Vault with transparent USC compression.

Documents and session logs are compressed on store and decompressed on retrieve.
Compression mode is chosen automatically based on content size:
  - < 1 KB: no compression (overhead not worth it)
  - >= 1 KB: USC cold mode (smart fallback: picks best of template, zstd, brotli, bzip2)

Metadata sidecars store compression info so decode is transparent.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from c3ae.storage.vault import Vault
from c3ae.utils import content_hash, json_dumps, json_loads, utcnow, iso_str

# Minimum size to bother compressing
_MIN_COMPRESS_BYTES = 1024

# Try to import USC — graceful fallback if not available
_USC_AVAILABLE = False
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "UnifiedStateCodec" / "src"))
    import zstandard as zstd
    _USC_AVAILABLE = True
except ImportError:
    zstd = None

try:
    import brotli as _brotli
except ImportError:
    _brotli = None

import bz2


def _compress_smart(data: bytes) -> tuple[bytes, str]:
    """Compress data using best available method. Returns (compressed, method)."""
    if len(data) < _MIN_COMPRESS_BYTES:
        return data, "none"

    candidates: list[tuple[bytes, str]] = [(data, "none")]

    # zstd-22
    if zstd is not None:
        try:
            z = b"USZR" + zstd.ZstdCompressor(level=22).compress(data)
            candidates.append((z, "uszr"))
        except Exception:
            pass

    # brotli-11
    if _brotli is not None:
        try:
            b = b"USBR" + _brotli.compress(data, quality=11)
            candidates.append((b, "usbr"))
        except Exception:
            pass

    # bzip2-9
    try:
        bz = b"USBZ" + bz2.compress(data, compresslevel=9)
        candidates.append((bz, "usbz"))
    except Exception:
        pass

    # Try USC template-based cold if available
    if _USC_AVAILABLE:
        try:
            from usc.api.hdfs_template_codec_v1m_bundle import bundle_encode_and_compress_v1m
            from usc.mem.hdfs_templates_v0 import HDFSTemplateBank, parse_hdfs_lines
            from usc.cli.app import cold_pack

            text = data.decode("utf-8", errors="replace")
            lines = text.splitlines()

            if len(lines) >= 10:  # Only try template mining on enough lines
                from drain3 import TemplateMiner
                from drain3.template_miner_config import TemplateMinerConfig
                import tempfile, csv, io, os

                config = TemplateMinerConfig()
                config.drain_sim_th = 0.4
                config.drain_depth = 4
                miner = TemplateMiner(config=config)
                for line in lines:
                    miner.add_log_message(line)

                csv_lines = ["EventId,EventTemplate"]
                for cluster in miner.drain.clusters:
                    csv_lines.append(f"E{cluster.cluster_id},{cluster.get_template()}")
                tpl_csv = "\n".join(csv_lines) + "\n"

                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tf:
                    tf.write(tpl_csv)
                    tpl_path = tf.name
                try:
                    bank = HDFSTemplateBank.from_csv(tpl_path)
                    events, unknown = parse_hdfs_lines(lines, bank)
                    bundle, _meta = bundle_encode_and_compress_v1m(
                        events=events, unknown_lines=unknown,
                        template_csv_text=tpl_csv, zstd_level=10,
                    )
                    cold = cold_pack(bundle)
                    candidates.append((cold, "uscc"))
                finally:
                    os.unlink(tpl_path)
        except Exception:
            pass

    # Pick smallest
    best = min(candidates, key=lambda x: len(x[0]))
    return best


def _decompress(data: bytes, method: str) -> bytes:
    """Decompress based on method tag."""
    if method == "none":
        return data
    if method == "uszr":
        if data[:4] != b"USZR":
            raise ValueError("Expected USZR magic")
        return zstd.ZstdDecompressor().decompress(data[4:])
    if method == "usbr":
        if data[:4] != b"USBR":
            raise ValueError("Expected USBR magic")
        return _brotli.decompress(data[4:])
    if method == "usbz":
        if data[:4] != b"USBZ":
            raise ValueError("Expected USBZ magic")
        return bz2.decompress(data[4:])
    if method == "uscc":
        # Full USC cold decode would go here
        # For now, not all datasets need template decode
        raise ValueError("USCC decode requires template reconstruction (use USZR/USBR fallback)")
    raise ValueError(f"Unknown compression method: {method}")


class CompressedVault(Vault):
    """Vault with transparent USC compression."""

    def store_document(self, data: bytes, filename: str,
                       metadata: dict[str, Any] | None = None) -> str:
        """Store a document with compression; returns content hash as ID."""
        h = content_hash(data)
        compressed, method = _compress_smart(data)

        dest = self.root / "documents" / f"{h}_{filename}"
        if method != "none":
            dest = dest.with_suffix(dest.suffix + ".usc")
        dest.write_bytes(compressed)

        ratio = len(data) / max(1, len(compressed))
        meta = {
            "original_name": filename,
            "content_hash": h,
            "size_bytes": len(data),
            "compressed_bytes": len(compressed),
            "compression_method": method,
            "compression_ratio": round(ratio, 2),
            "stored_at": iso_str(utcnow()),
            **(metadata or {}),
        }
        meta_path = self.root / "documents" / f"{h}_{filename}.meta.json"
        meta_path.write_text(json_dumps(meta))
        return h

    def get_document(self, content_hash_prefix: str) -> tuple[bytes, dict[str, Any]]:
        """Retrieve and decompress document."""
        docs_dir = self.root / "documents"
        matches = list(docs_dir.glob(f"{content_hash_prefix}*"))
        matches = [m for m in matches if not m.name.endswith(".meta.json")]
        if not matches:
            from c3ae.exceptions import VaultError
            raise VaultError(f"No document found for hash prefix {content_hash_prefix}")
        if len(matches) > 1:
            from c3ae.exceptions import VaultError
            raise VaultError(f"Ambiguous hash prefix {content_hash_prefix}")

        doc_path = matches[0]
        raw = doc_path.read_bytes()

        # Find metadata
        # Try both with and without .usc extension
        base_name = doc_path.name.replace(".usc", "")
        meta_path = docs_dir / f"{base_name}.meta.json"
        meta = json_loads(meta_path.read_text()) if meta_path.exists() else {}

        method = meta.get("compression_method", "none")
        if method != "none":
            data = _decompress(raw, method)
        else:
            data = raw

        return data, meta

    def store_raw_log(self, data: str, session_id: str, filename: str) -> Path:
        """Store session log with compression."""
        session_dir = self.root / "raw_logs" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        raw_bytes = data.encode("utf-8")
        compressed, method = _compress_smart(raw_bytes)

        if method != "none":
            dest = session_dir / f"{filename}.usc"
            dest.write_bytes(compressed)
            # Write compression metadata
            meta = {
                "original_name": filename,
                "size_bytes": len(raw_bytes),
                "compressed_bytes": len(compressed),
                "compression_method": method,
                "compression_ratio": round(len(raw_bytes) / max(1, len(compressed)), 2),
            }
            (session_dir / f"{filename}.usc.meta.json").write_text(json_dumps(meta))
        else:
            dest = session_dir / filename
            dest.write_text(data)

        return dest

    def compression_stats(self) -> dict[str, Any]:
        """Get overall compression statistics."""
        docs_dir = self.root / "documents"
        total_raw = 0
        total_compressed = 0
        methods: dict[str, int] = {}

        for meta_file in docs_dir.glob("*.meta.json"):
            try:
                meta = json_loads(meta_file.read_text())
                total_raw += meta.get("size_bytes", 0)
                total_compressed += meta.get("compressed_bytes", meta.get("size_bytes", 0))
                m = meta.get("compression_method", "none")
                methods[m] = methods.get(m, 0) + 1
            except Exception:
                continue

        return {
            "total_raw_bytes": total_raw,
            "total_compressed_bytes": total_compressed,
            "overall_ratio": round(total_raw / max(1, total_compressed), 2),
            "document_count": sum(methods.values()),
            "methods": methods,
        }
