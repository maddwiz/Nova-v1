# NovaSpine — Long-Term Memory for LLM Agents

Give any AI agent persistent memory with hybrid search (vector + keyword fusion). NovaSpine automatically stores, indexes, and retrieves relevant context from past conversations.

## Why NovaSpine?

Most LLM memory solutions use only vector search. NovaSpine combines **FAISS vector search** with **SQLite FTS5 keyword search** via **Reciprocal Rank Fusion** — catching both semantic similarity AND exact keyword matches.

| Feature | NovaSpine | Vector-only (Mem0, etc.) |
|---------|-----------|--------------------------|
| "Find messages about Desmond" | Keyword match + semantic | May miss if embedding is weak |
| "What did we discuss about auth?" | Semantic match + keyword boost | Works, but no keyword boost |
| Pre-formatted LLM injection | `/augment` endpoint | DIY formatting |
| Role filtering | Built-in (user/assistant only) | Manual |
| Content deduplication | Built-in | Manual |

## Quick Start

```bash
pip install novaspine
novaspine serve  # starts API on :8420
```

### Store a memory
```bash
curl -X POST localhost:8420/api/v1/memory/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "The user prefers dark mode and uses vim keybindings"}'
```

### Search memories
```bash
curl -X POST localhost:8420/api/v1/memory/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "user preferences", "top_k": 5}'
```

### Auto-inject into LLM context
```bash
curl -X POST localhost:8420/api/v1/memory/augment \
  -H "Content-Type: application/json" \
  -d '{"query": "what does the user like?", "top_k": 5, "format": "xml"}'
```

Returns pre-formatted `<relevant-memories>` block ready for LLM context injection.

## Architecture

```
┌─────────────────────────────────────────────┐
│            REST API (FastAPI)                │
│                                             │
│  /api/v1/memory/augment  ← Main product API │
│  /api/v1/memory/recall   ← Search memories  │
│  /api/v1/memory/ingest   ← Store text       │
│  /api/v1/health          ← Health check     │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│           MemorySpine (Core Engine)          │
│                                             │
│  ingest_text()   → chunk + embed + store    │
│  search()        → hybrid RRF search        │
│  search_keyword()→ FTS5 fallback            │
└──────────┬──────────┬───────────────────────┘
           │          │
    ┌──────▼───┐  ┌───▼──────────────────────┐
    │ Hybrid   │  │ Storage                  │
    │ Search   │  │                          │
    │          │  │ SQLiteStore → chunks,FTS5 │
    │ RRF merge│  │ FAISSStore → vectors     │
    │ 70/30    │  │ EmbeddingCache           │
    └──────────┘  └──────────────────────────┘
```

## How Hybrid Search Works

1. Run **keyword search** (FTS5) → ranked list
2. Run **vector search** (FAISS) → ranked list
3. Merge via **Reciprocal Rank Fusion**:
   - `score(doc) = Σ weight / (k + rank + 1)`
   - Vector weight: 0.7, Keyword weight: 0.3
4. Return top-k by combined score

Results that appear in **both** lists get a significant boost.

## Python API

```python
from c3ae.memory_spine.spine import MemorySpine
from c3ae.config import Config

spine = MemorySpine(Config())

# Store
await spine.ingest_text("User prefers dark mode")

# Search (hybrid)
results = await spine.search("user preferences", top_k=5)
for r in results:
    print(f"[{r.score:.2f}] {r.content}")

# Keyword-only fallback (no embedding API needed)
results = spine.search_keyword("dark mode", top_k=5)
```

## Configuration

```bash
# Environment variables
C3AE_DATA_DIR=/path/to/data          # Database + index location
C3AE_API_TOKEN=your-secret           # Bearer token for API auth
VENICE_API_KEY=your-key              # Embedding provider API key
```

## Also Includes: USC Compression Engine

NovaSpine includes **USC (Unified State Codec)** — a cognitive compression engine that learns from repeated patterns across sessions. Compression ratios improve over time as the dictionary grows (3.7x → 6.1x across 64 sessions).

## Development

See [HANDOFF-MVP.md](HANDOFF-MVP.md) for detailed technical documentation, architecture guide, and MVP roadmap.

```bash
git clone https://github.com/maddwiz/Nova-v1.git
cd Nova-v1
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
