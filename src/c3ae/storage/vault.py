"""Filesystem vault for large artifacts."""

from __future__ import annotations

import mimetypes
import shutil
from pathlib import Path
from typing import Any

from c3ae.exceptions import VaultError
from c3ae.utils import content_hash, json_dumps, json_loads, utcnow, iso_str


class Vault:
    """Content-addressed filesystem vault."""

    def __init__(self, vault_dir: Path | str) -> None:
        self.root = Path(vault_dir)
        for sub in ["documents", "evidence", "raw_logs", "code_snapshots"]:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def store_document(self, data: bytes, filename: str,
                       metadata: dict[str, Any] | None = None) -> str:
        """Store a document; returns content hash as ID."""
        h = content_hash(data)
        dest = self.root / "documents" / f"{h}_{filename}"
        dest.write_bytes(data)
        # Write sidecar metadata
        meta = {
            "original_name": filename,
            "content_hash": h,
            "size_bytes": len(data),
            "mime_type": mimetypes.guess_type(filename)[0] or "",
            "stored_at": iso_str(utcnow()),
            **(metadata or {}),
        }
        dest.with_suffix(dest.suffix + ".meta.json").write_text(json_dumps(meta))
        return h

    def get_document(self, content_hash_prefix: str) -> tuple[bytes, dict[str, Any]]:
        """Retrieve document by content hash prefix."""
        docs_dir = self.root / "documents"
        matches = list(docs_dir.glob(f"{content_hash_prefix}*"))
        # Filter out .meta.json files
        matches = [m for m in matches if not m.name.endswith(".meta.json")]
        if not matches:
            raise VaultError(f"No document found for hash prefix {content_hash_prefix}")
        if len(matches) > 1:
            raise VaultError(f"Ambiguous hash prefix {content_hash_prefix}: {len(matches)} matches")
        doc_path = matches[0]
        meta_path = doc_path.with_suffix(doc_path.suffix + ".meta.json")
        meta = json_loads(meta_path.read_text()) if meta_path.exists() else {}
        return doc_path.read_bytes(), meta

    def store_evidence(self, data: bytes, evidence_id: str) -> Path:
        dest = self.root / "evidence" / evidence_id
        dest.write_bytes(data)
        return dest

    def store_raw_log(self, data: str, session_id: str, filename: str) -> Path:
        session_dir = self.root / "raw_logs" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        dest = session_dir / filename
        dest.write_text(data)
        return dest

    def store_code_snapshot(self, source_dir: Path, snapshot_id: str) -> Path:
        dest = self.root / "code_snapshots" / snapshot_id
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source_dir, dest)
        return dest

    def list_documents(self) -> list[dict[str, Any]]:
        docs_dir = self.root / "documents"
        results = []
        for meta_file in docs_dir.glob("*.meta.json"):
            results.append(json_loads(meta_file.read_text()))
        return results

    def delete_document(self, content_hash_prefix: str) -> bool:
        docs_dir = self.root / "documents"
        matches = list(docs_dir.glob(f"{content_hash_prefix}*"))
        if not matches:
            return False
        for m in matches:
            m.unlink()
        return True
