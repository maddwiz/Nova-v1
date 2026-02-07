"""Self-Compressing Context Windows — save tokens before LLM calls.

Upgrade #7: When sending prompts to an LLM, chunks that the cogstore has
seen many times (high ref_count) are replaced with compact shorthand
references. The LLM never sees the raw repeated content — only a small
token placeholder. On response processing, placeholders are expanded back.

This is NOT traditional compression (bytes → bytes). It's *semantic*
compression: we remove content the *system* has memorized, because the
LLM doesn't need to see boilerplate the agent has encountered 50 times.

Usage:
    compactor = ContextCompactor(store, min_ref_count=3)

    # Before sending to LLM:
    compressed, savings = compactor.compress_prompt(prompt_text)
    # compressed has high-frequency chunks replaced with «REF:42»

    # After LLM responds (if response contains refs):
    expanded = compactor.expand_response(response_text)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from usc.cogdedup.hasher import content_defined_chunks, sha256_hash, simhash64
from usc.cogdedup.store import CogStore


# Placeholder format: «REF:chunk_id» — uses angle quotes to avoid
# collision with any normal text or code
_REF_PATTERN = re.compile(r"«REF:(\d+)»")
_REF_FMT = "«REF:{}»"


@dataclass
class CompactionResult:
    """Result of compressing a prompt."""
    text: str                    # Compressed prompt text
    original_tokens: int         # Estimated original token count
    compressed_tokens: int       # Estimated compressed token count
    savings_pct: float           # Percentage of tokens saved
    refs_inserted: int           # Number of REF placeholders inserted
    chunks_total: int            # Total chunks analyzed


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


class ContextCompactor:
    """Replace high-frequency chunks in prompts with compact references.

    Args:
        store: CogStore to look up chunk frequencies
        min_ref_count: Minimum ref_count for a chunk to be replaced
        min_chunk_len: Minimum chunk length (chars) worth replacing
    """

    def __init__(
        self,
        store: CogStore,
        *,
        min_ref_count: int = 3,
        min_chunk_len: int = 100,
    ) -> None:
        self._store = store
        self._min_ref_count = min_ref_count
        self._min_chunk_len = min_chunk_len
        # Cache of chunk_id -> original text for expansion
        self._expansion_cache: Dict[int, str] = {}
        # Stats
        self._total_saved_tokens: int = 0
        self._total_compressions: int = 0

    def compress_prompt(self, prompt: str) -> CompactionResult:
        """Compress a prompt by replacing high-frequency chunks with refs.

        Returns CompactionResult with compressed text and savings stats.
        """
        data = prompt.encode("utf-8")
        chunks = content_defined_chunks(data)
        if not chunks:
            return CompactionResult(
                text=prompt,
                original_tokens=_estimate_tokens(prompt),
                compressed_tokens=_estimate_tokens(prompt),
                savings_pct=0.0,
                refs_inserted=0,
                chunks_total=0,
            )

        # Build output: for each chunk, either keep it or replace with ref
        parts: List[str] = []
        refs_inserted = 0

        for chunk in chunks:
            chunk_text = chunk.decode("utf-8", errors="replace")
            sha = sha256_hash(chunk)

            # Look up in store
            entry = self._store.lookup_exact(sha)
            if (
                entry is not None
                and entry.ref_count >= self._min_ref_count
                and len(chunk_text) >= self._min_chunk_len
            ):
                # High-frequency chunk — replace with compact reference
                ref_placeholder = _REF_FMT.format(entry.chunk_id)
                parts.append(ref_placeholder)
                self._expansion_cache[entry.chunk_id] = chunk_text
                refs_inserted += 1
            else:
                parts.append(chunk_text)

        compressed_text = "".join(parts)
        orig_tokens = _estimate_tokens(prompt)
        comp_tokens = _estimate_tokens(compressed_text)
        savings = ((orig_tokens - comp_tokens) / max(1, orig_tokens)) * 100.0

        self._total_saved_tokens += max(0, orig_tokens - comp_tokens)
        self._total_compressions += 1

        return CompactionResult(
            text=compressed_text,
            original_tokens=orig_tokens,
            compressed_tokens=comp_tokens,
            savings_pct=round(savings, 1),
            refs_inserted=refs_inserted,
            chunks_total=len(chunks),
        )

    def expand_response(self, text: str) -> str:
        """Expand any «REF:N» placeholders back to original content.

        Used when LLM output echoes back references (rare but possible).
        Also used to reconstruct the full prompt if needed for logging.
        """
        def _replace(m: re.Match) -> str:
            chunk_id = int(m.group(1))
            if chunk_id in self._expansion_cache:
                return self._expansion_cache[chunk_id]
            # Try store
            entry = self._store.get(chunk_id)
            if entry is not None and entry.data:
                return entry.data.decode("utf-8", errors="replace")
            return m.group(0)  # Can't expand — leave as-is

        return _REF_PATTERN.sub(_replace, text)

    def expand_prompt(self, compressed_text: str) -> str:
        """Alias for expand_response — works on any text with refs."""
        return self.expand_response(compressed_text)

    @property
    def total_saved_tokens(self) -> int:
        return self._total_saved_tokens

    @property
    def total_compressions(self) -> int:
        return self._total_compressions

    def stats(self) -> dict:
        return {
            "total_compressions": self._total_compressions,
            "total_saved_tokens": self._total_saved_tokens,
            "expansion_cache_size": len(self._expansion_cache),
        }
