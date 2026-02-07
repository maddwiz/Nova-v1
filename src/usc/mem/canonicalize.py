import re


# ISO-ish timestamps: 2026-01-18 05:50:12 or 2026-01-18T05:50:12Z
RE_ISO_TS = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b"
)

# UUIDs
RE_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

# Long hex blobs / hashes (>= 16 chars)
RE_LONG_HEX = re.compile(r"\b[0-9a-fA-F]{16,}\b")

# Long integers (>= 7 digits) often IDs
RE_LONG_INT = re.compile(r"\b\d{7,}\b")


def canonicalize_lossy(line: str) -> str:
    """
    Canonicalize logs to increase repetition.
    LOSSY v0 (experimental):
    - Replace timestamps/UUIDs/long hex/long ints with placeholders.
    - Normalize whitespace.
    """
    s = line

    # Replace common high-entropy patterns
    s = RE_ISO_TS.sub("<TS>", s)
    s = RE_UUID.sub("<UUID>", s)
    s = RE_LONG_HEX.sub("<HEX>", s)
    s = RE_LONG_INT.sub("<INT>", s)

    # Normalize whitespace (also lossy)
    s = " ".join(s.split())

    return s
