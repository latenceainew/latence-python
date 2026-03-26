# Chunking Service

Split processed documents into semantically meaningful chunks for granular processing,
retrieval, and embedding. Chunking is the foundational step that transforms full-text
documents into ingestible segments for downstream AI services.

## Overview

The chunking service supports 4 strategies with automatic Markdown structure preservation:

| Strategy | Description | Pricing |
|----------|-------------|---------|
| `character` | Fixed character-length splits with sentence-boundary alignment | **Free** |
| `token` | Token-boundary splits using tiktoken (o200k_base) | **Free** |
| `semantic` | Embedding-based grouping — chunks share semantic coherence | $0.10 / 1M chars |
| `hybrid` | Character splits refined with semantic coherence scoring | $0.10 / 1M chars |

All strategies automatically detect Markdown, HTML, and structured document formats
and preserve heading hierarchies, code blocks, lists, and section boundaries.

## Pipeline Position

Chunking always runs immediately after Document Intelligence and before all other
services:

```
Document Intelligence → Chunking → Redaction → Extraction → Ontology → Compression
```

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `strategy` | string | `"hybrid"` | character, token, semantic, hybrid | Splitting strategy |
| `chunk_size` | int | `512` | 64–8192 | Target chunk size (chars or tokens) |
| `chunk_overlap` | int | `50` | 0–4096 | Overlap between adjacent chunks |
| `min_chunk_size` | int | `64` | 1–8192 | Minimum chunk size (smaller chunks discarded) |

## Response

Each chunk includes:

| Field | Type | Description |
|-------|------|-------------|
| `content` | string | Chunk text content |
| `index` | int | 0-based chunk position |
| `start` | int | Start character offset in source text |
| `end` | int | End character offset in source text |
| `char_count` | int | Character count |
| `token_count` | int | Token count (tiktoken o200k_base) |
| `semantic_score` | float | Intra-chunk coherence (0–1, semantic/hybrid only) |
| `section_path` | list | Heading hierarchy at chunk position |

## Examples

### Direct API (Experimental)

```python
from latence import Latence

client = Latence(api_key="lat_xxx")

# Character chunking (free)
result = client.experimental.chunking.chunk(
    text="Long document text...",
    strategy="character",
    chunk_size=1000,
)
for chunk in result.data.chunks:
    print(f"[{chunk.index}] {chunk.char_count} chars: {chunk.content[:50]}...")

# Semantic chunking (charged)
result = client.experimental.chunking.chunk(
    text="Long document text...",
    strategy="semantic",
    chunk_size=512,
    chunk_overlap=50,
)
print(f"{result.data.num_chunks} semantic chunks")
```

### Pipeline Builder

```python
from latence import Latence, PipelineBuilder

client = Latence(api_key="lat_xxx")

config = (
    PipelineBuilder()
    .doc_intel(mode="performance")
    .chunking(strategy="hybrid", chunk_size=512)
    .extraction(threshold=0.3)
    .build()
)

job = client.pipeline.submit(config, files=["report.pdf"])
pkg = job.wait_for_completion()

print(f"{pkg.chunking.summary.num_chunks} chunks produced")
for chunk in pkg.chunking.chunks[:3]:
    print(f"  [{chunk['index']}] {chunk['char_count']} chars")
```

### Async Usage

```python
from latence import AsyncLatence

async with AsyncLatence(api_key="lat_xxx") as client:
    result = await client.experimental.chunking.chunk(
        text="Document...", strategy="hybrid"
    )
```

## Strategies in Depth

### Character (`character`)

Splits text at character boundaries with sentence alignment. Fast and deterministic.
Best for uniform chunk sizes when semantic grouping is not needed.

### Token (`token`)

Splits at token boundaries using tiktoken's o200k_base tokenizer. Ensures each chunk
stays within a token budget — ideal for embedding models with strict token limits.

### Semantic (`semantic`)

Uses embedding similarity to group sentences into coherent chunks. Produces
variable-length chunks that respect topic boundaries. Each chunk includes a
`semantic_score` indicating internal coherence.

### Hybrid (`hybrid`)

Combines character-based splitting with semantic refinement. First splits at character
boundaries, then adjusts boundaries using embedding similarity. Balances consistent
sizing with semantic coherence.

## Markdown Awareness

All strategies automatically detect structured document formats and preserve:

- **Heading hierarchies** — chunks respect section boundaries
- **Code blocks** — fenced code blocks are never split mid-block
- **Lists** — list items are kept together when possible
- **Tables** — table rows are preserved
- **Section paths** — each chunk reports its heading hierarchy via `section_path`
