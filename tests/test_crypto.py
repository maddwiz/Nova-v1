"""Tests for N3: PII Detection & Encryption."""
import pytest
from usc.crypto.pii_detect import detect_pii, redact_pii, PIIMatch

# Check if cryptography is available
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

requires_crypto = pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")


class TestPIIDetection:
    def test_detect_email(self):
        text = "Contact alice@example.com for details"
        matches = detect_pii(text)
        assert any(m.kind == "email" and m.value == "alice@example.com" for m in matches)

    def test_detect_phone(self):
        text = "Call (555) 123-4567 for support"
        matches = detect_pii(text)
        assert any(m.kind == "phone" for m in matches)

    def test_detect_ssn(self):
        text = "SSN: 123-45-6789"
        matches = detect_pii(text)
        assert any(m.kind == "ssn" and m.value == "123-45-6789" for m in matches)

    def test_detect_ip(self):
        text = "Server at 192.168.1.100 is down"
        matches = detect_pii(text)
        assert any(m.kind == "ip" and m.value == "192.168.1.100" for m in matches)

    def test_detect_credit_card(self):
        text = "Card: 4111-1111-1111-1111"
        matches = detect_pii(text)
        assert any(m.kind == "credit_card" for m in matches)

    def test_detect_multiple_types(self):
        text = "Email: bob@test.org, IP: 10.0.0.1, SSN: 999-88-7777"
        matches = detect_pii(text)
        kinds = {m.kind for m in matches}
        assert "email" in kinds
        assert "ip" in kinds
        assert "ssn" in kinds

    def test_no_pii(self):
        text = "This is a clean text with no PII at all."
        matches = detect_pii(text)
        assert len(matches) == 0

    def test_positions_correct(self):
        text = "User: alice@test.com"
        matches = detect_pii(text)
        for m in matches:
            assert text[m.start:m.end] == m.value


class TestRedaction:
    def test_redact_replaces_pii(self):
        text = "Email: alice@example.com"
        redacted = redact_pii(text)
        assert "alice@example.com" not in redacted
        assert "[REDACTED]" in redacted

    def test_redact_no_pii_returns_same(self):
        text = "Clean text here"
        assert redact_pii(text) == text

    def test_custom_replacement(self):
        text = "IP: 10.0.0.1"
        redacted = redact_pii(text, replacement="***")
        assert "***" in redacted
        assert "10.0.0.1" not in redacted


@requires_crypto
class TestEncryption:
    def test_encrypt_decrypt_roundtrip(self):
        from usc.crypto import encrypt_fields, decrypt_fields, generate_key
        key = generate_key()
        fields = ["alice@test.com", "normal text", "123-45-6789", "more text"]
        pii_indices = [0, 2]  # encrypt email and SSN
        blob = encrypt_fields(fields, key, pii_indices)
        decrypted = decrypt_fields(blob, key)
        assert decrypted == fields

    def test_wrong_key_fails(self):
        from usc.crypto import encrypt_fields, decrypt_fields, generate_key
        key1 = generate_key()
        key2 = generate_key()
        fields = ["secret@email.com", "public"]
        blob = encrypt_fields(fields, key1, [0])
        with pytest.raises(Exception):
            decrypt_fields(blob, key2)

    def test_magic_header(self):
        from usc.crypto import encrypt_fields, generate_key, MAGIC
        key = generate_key()
        blob = encrypt_fields(["test"], key, [0])
        assert blob[:4] == MAGIC

    def test_no_pii_passthrough(self):
        from usc.crypto import encrypt_fields, decrypt_fields, generate_key
        key = generate_key()
        fields = ["plain", "text", "only"]
        blob = encrypt_fields(fields, key, [])  # no PII indices
        decrypted = decrypt_fields(blob, key)
        assert decrypted == fields

    def test_all_pii_encrypted(self):
        from usc.crypto import encrypt_fields, decrypt_fields, generate_key
        key = generate_key()
        fields = ["secret1", "secret2", "secret3"]
        blob = encrypt_fields(fields, key, [0, 1, 2])
        decrypted = decrypt_fields(blob, key)
        assert decrypted == fields

    def test_bad_magic_raises(self):
        from usc.crypto import decrypt_fields, generate_key
        with pytest.raises(ValueError, match="bad magic"):
            decrypt_fields(b"XXXX" + b"\x00" * 20, generate_key())
