"""Tests for G3: TOKEN Mode."""
import pytest
from usc.token import token_encode, token_decode, WhitespaceTokenizer, MAGIC
from usc.token.token_codec import _delta_encode, _delta_decode


class TestDeltaCoding:
    def test_roundtrip(self):
        values = [10, 15, 12, 20, 18]
        deltas = _delta_encode(values)
        restored = _delta_decode(deltas)
        assert restored == values

    def test_empty(self):
        assert _delta_encode([]) == []
        assert _delta_decode([]) == []

    def test_single_value(self):
        assert _delta_decode(_delta_encode([42])) == [42]

    def test_monotonic_increasing(self):
        values = list(range(100))
        deltas = _delta_encode(values)
        # All deltas after first should be 1
        assert all(d == 1 for d in deltas[1:])
        assert _delta_decode(deltas) == values


class TestWhitespaceTokenizer:
    def test_roundtrip(self):
        tok = WhitespaceTokenizer()
        text = "hello world foo bar"
        tokens = tok.encode(text)
        decoded = tok.decode(tokens)
        assert decoded == text

    def test_repeated_words_same_id(self):
        tok = WhitespaceTokenizer()
        tokens = tok.encode("hello hello hello")
        assert tokens[0] == tokens[1] == tokens[2]

    def test_unique_ids(self):
        tok = WhitespaceTokenizer()
        tokens = tok.encode("a b c d")
        assert len(set(tokens)) == 4


class TestTokenCodec:
    def test_basic_roundtrip(self):
        tok = WhitespaceTokenizer()
        text = "hello world this is a test of token mode encoding"
        blob = token_encode(text, tokenizer=tok)
        decoded = token_decode(blob, tokenizer=tok)
        assert decoded == text

    def test_magic_header(self):
        blob = token_encode("test data", tokenizer=WhitespaceTokenizer())
        assert blob[:4] == MAGIC

    def test_compression_smaller_than_raw(self):
        tok = WhitespaceTokenizer()
        # Repetitive text should compress well
        text = " ".join(["hello world test data"] * 50)
        blob = token_encode(text, tokenizer=tok)
        raw_size = len(text.encode("utf-8"))
        assert len(blob) < raw_size

    def test_shared_tokenizer_instance(self):
        # WhitespaceTokenizer builds vocab during encode, so same instance needed
        tok = WhitespaceTokenizer()
        text = "simple test text"
        blob = token_encode(text, tokenizer=tok)
        decoded = token_decode(blob, tokenizer=tok)
        assert decoded == text

    def test_bad_magic_raises(self):
        with pytest.raises(ValueError, match="bad magic"):
            token_decode(b"XXXX" + b"\x00" * 20)

    def test_too_small_raises(self):
        with pytest.raises(ValueError, match="too small"):
            token_decode(b"UT")

    def test_varied_text_roundtrip(self):
        tok = WhitespaceTokenizer()
        text = "User alice logged in from 10.0.0.1 at 2024-01-15T10:30:00Z"
        blob = token_encode(text, tokenizer=tok)
        decoded = token_decode(blob, tokenizer=tok)
        assert decoded == text

    def test_single_word(self):
        tok = WhitespaceTokenizer()
        text = "hello"
        blob = token_encode(text, tokenizer=tok)
        decoded = token_decode(blob, tokenizer=tok)
        assert decoded == text

    def test_many_tokens(self):
        tok = WhitespaceTokenizer()
        words = [f"word{i}" for i in range(500)]
        text = " ".join(words)
        blob = token_encode(text, tokenizer=tok, level=3)
        decoded = token_decode(blob, tokenizer=tok)
        assert decoded == text

    def test_different_compression_levels(self):
        tok = WhitespaceTokenizer()
        text = " ".join(["hello world"] * 100)
        blob_low = token_encode(text, tokenizer=tok, level=1)
        blob_high = token_encode(text, tokenizer=tok, level=19)
        # Both should decode correctly
        assert token_decode(blob_low, tokenizer=tok) == text
        assert token_decode(blob_high, tokenizer=tok) == text
        # High level should be similar size (tiny data can have marginal overhead)
        assert len(blob_high) <= len(blob_low) + 5
