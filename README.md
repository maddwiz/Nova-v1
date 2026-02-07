# Nova v1 — Cognitive Memory System for AI Agents

A unified memory system that **compresses, remembers, and learns** from AI agent sessions. The more Nova runs, the better it gets.

## What This Is

Nova combines two systems into one:

- **USC (Unified State Codec)** — a compression engine with cognitive deduplication. Instead of compressing data from scratch each time, it remembers what it's seen before. Repeated patterns get stored once and referenced thereafter.

- **C3/Ae (Cognitive Compression Core)** — a persistent memory spine with SQLite storage, FAISS vector search, reasoning bank, governance, and audit logging. It provides the long-term memory that USC's compressor learns from.

Together they form a **learning compressor** — a system where compression improves over time as the cognitive dictionary grows.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  Nova v1                          │
│                                                   │
│  ┌─────────────┐    ┌──────────────────────────┐ │
│  │ USC Engine   │    │ C3/Ae Memory Spine       │ │
│  │              │    │                          │ │
│  │ CDC Chunking │◄──►│ SQLite (chunks, LSH)    │ │
│  │ SimHash LSH  │    │ FAISS (vectors)         │ │
│  │ Delta Coding │    │ Reasoning Bank          │ │
│  │ Predictive   │    │ Governance + Audit      │ │
│  │ Compression  │    │ CompressedVault         │ │
│  └─────────────┘    └──────────────────────────┘ │
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │ Cogdedup Upgrades                           │ │
│  │                                             │ │
│  │ Integrity Verification  │ Anomaly Detection │ │
│  │ Context Compaction      │ Temporal Patterns │ │
│  │ Recursive Compression   │ Federation       │ │
│  └─────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

## Real-World Results

Tested on 64 actual Nova agent sessions (45.9 MB total):

| Metric | Value |
|--------|-------|
| Sessions compressed | 64 |
| Total raw size | 45.9 MB |
| Compressed size | 6.96 MB (6.6x) |
| Roundtrip integrity | 64/64 PASSED |
| First-quarter avg ratio | 3.7x (system still learning) |
| Last-quarter avg ratio | 6.1x (system has learned) |
| REF reuse (early) | 0.9% |
| REF reuse (late) | 29.9% |
| Largest session (23 MB) | 8.0x compression, 41.9% REF reuse |

The learning curve is real — compression improves 1.65x as the cognitive dictionary grows.

## Key Features

### Cognitive Deduplication
Content-defined chunking (Rabin fingerprint) splits data into ~4KB chunks. Each chunk is hashed (SHA-256 for exact match, SimHash for similarity). Known chunks become REF pointers. Similar chunks become delta-compressed. Only truly novel data gets full compression.

### Cross-Session Learning
The C3CogStore persists in SQLite with hot/warm/cold tiers. Chunks that appear frequently get promoted to the hot tier for instant lookup. The more sessions flow through, the more patterns are recognized.

### Integrity Verification
Every compression includes a hash of the original data. Decompression can verify integrity, catching any corruption in the delta chain.

### Anomaly Detection
Compression ratios are monitored with z-score analysis. Unusual data (much more or less compressible than expected) triggers alerts — a free signal for detecting drift or novel inputs.

### Temporal Pattern Detection
Event sequences (tool calls, searches, errors) are tracked for recurring motifs. The system learns the rhythms of agent behavior.

### Self-Compressing Memory
C3's own memories, reasoning entries, and audit logs can be compressed through the same cognitive dedup pipeline — the system compresses itself.

## Quick Start

```bash
# Install
pip install -e .

# Compress a session
nova compress-session path/to/session.jsonl -o compressed/

# Check cognitive dedup stats
nova cogdedup-stats

# View anomaly report
nova anomaly-report

# Search memory
nova search "error handling patterns"

# System status
nova status
```

## Python API

```python
from c3ae.memory_spine.spine import MemorySpine
from c3ae.config import Config

spine = MemorySpine(Config())

# Compress a session (all upgrades active)
result = spine.compress_session(session_bytes, session_id="session-42")
# result: {blob, stats, compressed_size, original_size}

# Decompress with integrity check
data = spine.decompress_with_dedup(blob, expected_hash=result["stats"]["integrity_hash"])

# Compress an LLM prompt (replace known chunks with REF placeholders)
compacted = spine.compress_prompt("Your long prompt here...")
# compacted: {compressed, token_savings, refs_used, original_tokens}

# Track temporal events
spine.track_event("tool_call")
spine.track_event("tool_result")
motifs = spine.track_events_batch([])  # Get detected patterns

# Compress C3's own reasoning bank
blob, stats = spine.compress_reasoning_bank()

# Full system status
status = spine.status()
```

## OpenClaw Integration

Nova includes a post-session hook for OpenClaw:

```bash
python scripts/openclaw_session_hook.py ~/.openclaw/agents/main/sessions/session-id.jsonl
```

This compresses each session transcript through the cognitive dedup pipeline and stores the compressed blob, building the cognitive dictionary over time.

## Wire Format

USC uses self-describing wire formats with magic byte identification:

| Magic | Format |
|-------|--------|
| `UCOG` | Cognitive dedup (REF/DELTA/FULL/PRED_DELTA chunks) |
| `USST` | Stream mode |
| `TPF3` | Hot-lite-full template compression |
| `USZR` | Raw zstd fallback |
| `USZD` | Trained-dictionary zstd |

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

555 tests across USC core, C3/Ae, and integration.

## License

MIT
