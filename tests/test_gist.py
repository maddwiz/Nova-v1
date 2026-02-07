"""Tests for G7: Utility Gisting — Agent Summaries."""
import pytest
from usc.gist import (
    extract_gist,
    encode_with_gist,
    extract_gist_from_blob,
    decode_gist_blob,
    MAGIC,
)


def _sample_text():
    return (
        "Project: USC\n"
        "Goal: build a unified state codec.\n"
        "Decision: start with USC-MEM v0.\n"
        "Next: implement skeleton + witnesses + residual patches.\n"
        "Note: always keep ROADMAP, FILEMAP, MASTER_HANDOFF, CHANGES updated.\n"
        "\n"
        "Session recap:\n"
        "- Created repo scaffold.\n"
        "- Fixed pyenv python version.\n"
        "- Installed usc editable.\n"
        "- Installed pytest.\n"
        "- Ran tests successfully.\n"
        "\n"
        "Reminder: compress memories safely using anchors and residuals.\n"
        "Reminder: compress memories safely using anchors and residuals.\n"
        "Reminder: compress memories safely using anchors and residuals.\n"
    )


class TestExtractGist:
    def test_gist_shorter_than_original(self):
        text = _sample_text()
        gist = extract_gist(text, max_lines=3)
        assert len(gist) < len(text)
        assert len(gist.splitlines()) <= 3

    def test_gist_preserves_order(self):
        text = "Line A\nLine B\nLine C\nLine D\nLine E\nLine F"
        gist = extract_gist(text, max_lines=3)
        lines = gist.splitlines()
        assert len(lines) == 3
        # Lines should be in original order
        original_indices = [text.splitlines().index(ln) for ln in lines]
        assert original_indices == sorted(original_indices)

    def test_gist_short_text_returns_all(self):
        text = "Line A\nLine B"
        gist = extract_gist(text, max_lines=5)
        assert gist == "Line A\nLine B"

    def test_gist_empty_text(self):
        assert extract_gist("") == ""
        assert extract_gist("  \n  \n  ") == ""

    def test_gist_picks_representative_lines(self):
        # Repeated words should boost those lines' scores
        text = (
            "error: connection failed\n"
            "info: starting server\n"
            "error: timeout on connection\n"
            "debug: internal state\n"
            "error: connection reset\n"
            "info: server ready\n"
        )
        gist = extract_gist(text, max_lines=3)
        # "error" and "connection" appear most → those lines should be picked
        lines = gist.splitlines()
        error_lines = [ln for ln in lines if "error" in ln.lower()]
        assert len(error_lines) >= 2


class TestGistCodec:
    def test_encode_decode_roundtrip(self):
        gist = "This is a summary"
        inner = b"\x01\x02\x03\x04\x05" * 100
        blob = encode_with_gist(inner, gist)
        decoded_gist, decoded_inner = decode_gist_blob(blob)
        assert decoded_gist == gist
        assert decoded_inner == inner

    def test_magic_header(self):
        blob = encode_with_gist(b"data", "summary")
        assert blob[:4] == MAGIC

    def test_extract_gist_without_full_decode(self):
        gist = "Quick summary for preview"
        inner = b"\x00" * 10000  # large inner blob
        blob = encode_with_gist(inner, gist)
        # This should NOT need to process the inner blob
        extracted = extract_gist_from_blob(blob)
        assert extracted == gist

    def test_empty_gist(self):
        blob = encode_with_gist(b"inner-data", "")
        gist, inner = decode_gist_blob(blob)
        assert gist == ""
        assert inner == b"inner-data"

    def test_empty_inner(self):
        blob = encode_with_gist(b"", "summary text")
        gist, inner = decode_gist_blob(blob)
        assert gist == "summary text"
        assert inner == b""

    def test_unicode_gist(self):
        gist = "Summary: datos procesados correctamente"
        blob = encode_with_gist(b"data", gist)
        extracted = extract_gist_from_blob(blob)
        assert extracted == gist

    def test_bad_magic_raises(self):
        with pytest.raises(ValueError, match="bad magic"):
            decode_gist_blob(b"XXXX" + b"\x00" * 10)

    def test_too_small_raises(self):
        with pytest.raises(ValueError, match="too small"):
            decode_gist_blob(b"UG")
