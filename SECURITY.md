# USC Security Model

## Threat Model

USC's cognitive deduplication uses delta compression and content-addressed
storage. This creates attack surfaces that pure compression doesn't have:

### 1. Delta Chain Corruption
**Threat**: A corrupted or adversarial chunk becomes a delta base, causing
all chunks that reference it to decode incorrectly.

**Mitigation**: `IntegrityVerifier` computes fast hashes (xxHash-64 or CRC32)
after delta decompression and rejects corrupt outputs. Enable via:
```python
from usc.cogdedup.integrity import IntegrityVerifier, SecurityPolicy
verifier = IntegrityVerifier(SecurityPolicy(verify_deltas=True))
```

### 2. Decompression Bombs
**Threat**: A crafted delta that expands to enormous output, exhausting memory.

**Mitigation**: `SecurityPolicy.max_delta_expansion` limits the expansion
ratio. Default: 100x (a 1KB delta can produce at most 100KB output).

### 3. Reference Flooding
**Threat**: An adversarial chunk with artificially high ref_count becomes
the similarity target for everything, degrading compression quality.

**Mitigation**: `SecurityPolicy.max_ref_count_for_similarity` caps how
many times a chunk can be used as a delta base. Default: 1000.

### 4. Cross-Agent Leakage (Federation)
**Threat**: In federated mode, one agent's private data could leak to
another via the shared tier.

**Mitigation**: Only chunks exceeding `promote_threshold` ref_count
get promoted. Promotion copies data (not references), so the shared
tier holds common patterns, not unique content. Content-specific data
stays in each agent's local tier.

### 5. Timing Side Channels
**Threat**: Compression ratio differences reveal information about input
similarity to stored data.

**Mitigation**: This is inherent to deduplication and cannot be fully
eliminated. For sensitive applications, use `FULL` encoding only
(disable similarity search) which behaves like standard compression.

## Configuration

```python
policy = SecurityPolicy(
    max_ref_count_for_similarity=1000,  # Limit delta base reuse
    verify_deltas=True,                 # Hash check after decompress
    max_delta_expansion=100.0,          # Expansion ratio limit
)
verifier = IntegrityVerifier(policy)
```

## Wire Format Integrity

All USC wire formats use magic bytes for identification:
- `UCOG` — Cognitive dedup
- `USST` — Stream mode
- `TPF3` — Hot-lite-full
- `USZR` — Raw zstd fallback
- `USZD` — Trained-dictionary zstd
- `USBR` — Raw brotli fallback
- `USBZ` — Raw bzip2 fallback

Unknown magic bytes are rejected at decode time.
