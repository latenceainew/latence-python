<p align="center">
  <img src="https://latence.ai/logo.svg" alt="Latence AI" width="200">
</p>

<h1 align="center">Latence AI Python SDK</h1>

<p align="center">
  <strong>Documents in. Intelligence out.</strong><br>
  Turn unstructured documents into structured knowledge graphs, entities, and RAG-ready data -- in a single pipeline call.
</p>

<p align="center">
  <a href="https://pypi.org/project/latence/"><img src="https://img.shields.io/pypi/v/latence?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/latence/"><img src="https://img.shields.io/pypi/pyversions/latence" alt="Python"></a>
  <a href="https://github.com/latenceai/latence-python/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#the-pipeline">The Pipeline</a> &bull;
  <a href="#the-data-package">The Data Package</a> &bull;
  <a href="#pipeline-builder">Builder API</a> &bull;
  <a href="#direct-api-access">Direct API</a> &bull;
  <a href="#dataset-intelligence">Dataset Intelligence</a> &bull;
  <a href="SDK_TUTORIAL.md">Full Tutorial</a>
</p>

---

## Quick Start

```bash
pip install latence
```

```python
from latence import Latence

client = Latence()  # reads LATENCE_API_KEY from environment

job = client.pipeline.run(files=["contract.pdf"])
pkg = job.wait_for_completion()

print(pkg.document.markdown)                        # clean extracted text
print(pkg.entities.summary)                         # {"total": 142, "by_type": {"PERSON": 23, ...}}
print(pkg.knowledge_graph.summary.total_relations)  # 87
pkg.download_archive("./results.zip")               # organized ZIP with everything
```

That's it. Four lines from PDF to structured knowledge.

---

## The Pipeline

The Latence AI **Data Intelligence Pipeline** chains multiple AI services into a single async job. You submit documents, configure which services to run, and get back a structured `DataPackage` with everything organized and summarized.

### How it works

The pipeline executes services as a **directed acyclic graph (DAG)**, not a linear chain. Independent branches run in parallel:

```
                    ┌─── extraction ──── relation_extraction
                    │
document_intelligence ─┼─── redaction
                    │
                    └─── compression
```

You declare which services you want. The pipeline handles ordering, dependencies, and parallel execution automatically.

### Smart defaults

Provide only files -- the pipeline auto-applies: **OCR -> Entity Extraction -> Relation Extraction**

```python
job = client.pipeline.run(files=["report.pdf"])
```

### Explicit steps

Configure each service individually:

```python
job = client.pipeline.run(
    files=["contract.pdf"],
    name="Legal Analysis",
    steps={
        "ocr": {"mode": "performance", "output_format": "markdown"},
        "redaction": {"mode": "balanced", "redact": True, "redaction_mode": "mask"},
        "extraction": {
            "label_mode": "hybrid",
            "user_labels": ["person", "organization", "date", "monetary_amount"],
            "threshold": 0.3,
        },
        "relation_extraction": {"resolve_entities": True, "optimize_relations": True},
        "compression": {"compression_rate": 0.5},
    },
)
pkg = job.wait_for_completion()
```

### Available pipeline services

| Step | Aliases | What it does |
|------|---------|-------------|
| `document_intelligence` | `ocr`, `doc_intel` | OCR, layout detection, markdown extraction |
| `extraction` | `extract` | Zero-shot named entity recognition |
| `relation_extraction` | `ontology`, `knowledge_graph`, `graph` | Relation extraction, knowledge graph construction |
| `redaction` | `redact` | PII detection, masking, or synthetic replacement |
| `compression` | `compress` | Intelligent token-level text compression |

Steps are automatically sorted into the correct DAG execution order.

### From text or entities

Pipelines don't require files. You can start from raw text or pre-extracted entities:

```python
# Text input (skips OCR automatically)
job = client.pipeline.run(
    text="Apple Inc. was founded by Steve Jobs in Cupertino, California.",
    steps={"extraction": {"label_mode": "generated"}},
)

# Entity input (relation extraction only)
job = client.pipeline.run(
    entities=[{"text": "Apple", "label": "ORG", "start": 0, "end": 5, "score": 0.98}],
    steps={"knowledge_graph": {"resolve_entities": True}},
)
```

### Async / await

```python
from latence import AsyncLatence

async with AsyncLatence() as client:
    job = await client.pipeline.run(files=["doc.pdf"])
    pkg = await job.wait_for_completion()
```

---

## The Data Package

Every pipeline returns a `DataPackage` -- structured, summarized, and ready for downstream use.

| Section | What's inside | When present |
|---------|-------------|--------------|
| `pkg.document` | Markdown text, per-page content, metadata | OCR ran |
| `pkg.entities` | Entity list, summary (total, by_type, avg_confidence) | Extraction ran |
| `pkg.knowledge_graph` | Entities, relations, graph summary | Relation Extraction ran |
| `pkg.redaction` | Cleaned text, PII list, summary | Redaction ran |
| `pkg.compression` | Compressed text, ratio, tokens saved | Compression ran |
| `pkg.quality` | Per-stage report, confidence scores, cost | Always |

### Explore results

```python
pkg = job.wait_for_completion()

# Document
print(pkg.document.markdown)
print(pkg.document.metadata.pages_processed)

# Entities
print(pkg.entities.summary.total)           # 142
print(pkg.entities.summary.by_type)         # {"PERSON": 23, "ORG": 18, ...}
print(pkg.entities.summary.avg_confidence)  # 0.87
for e in pkg.entities.items:
    print(f"  {e.text} [{e.label}] {e.score:.2f}")

# Knowledge graph
print(pkg.knowledge_graph.summary.total_relations)
for r in pkg.knowledge_graph.relations:
    print(f"  {r.entity1} --[{r.relation_type}]--> {r.entity2}")

# Quality & cost
print(f"Cost: ${pkg.quality.total_cost_usd:.4f}")
print(f"Time: {pkg.quality.total_processing_time_ms:.0f}ms")
```

### Export

```python
# Organized ZIP archive
pkg.download_archive("./results.zip")
# -> Legal_Analysis/
#      README.md, document.md, entities.json, knowledge_graph.json,
#      quality_report.json, metadata.json, pages/page_001.md, ...

# Single consolidated JSON (document-centric, zero redundancy)
merged = pkg.merge(save_to="./results.json")
```

---

## Pipeline Builder

For power users who want a typed, chainable API with client-side validation:

```python
from latence import PipelineBuilder

config = (
    PipelineBuilder()
    .doc_intel(mode="performance")
    .extraction(
        label_mode="hybrid",
        user_labels=["person", "organization", "date"],
        threshold=0.3,
    )
    .relation_extraction(resolve_entities=True, optimize_relations=True)
    .compression(compression_rate=0.5)
    .store_intermediate()
    .build()
)

job = client.pipeline.submit(config, files=["contract.pdf"])
pkg = job.wait_for_completion()
```

The builder validates parameters client-side (threshold ranges, valid modes, valid dimensions) and rejects duplicates before the request leaves your machine.

### YAML configuration

Define pipelines in version-controlled YAML:

```yaml
# pipeline.yml
steps:
  document_intelligence:
    mode: performance
  extraction:
    label_mode: hybrid
    user_labels: [person, organization, date]
  relation_extraction:
    resolve_entities: true
```

```python
config = PipelineBuilder.from_yaml("pipeline.yml")
job = client.pipeline.submit(config, files=["contract.pdf"])
```

### Validate before running

```python
result = client.pipeline.validate(config, files=["doc.pdf"])
print(result.valid)          # True
print(result.auto_injected)  # ["document_intelligence"]
print(result.warnings)       # []
```

---

## Job Lifecycle

Pipelines are async jobs. You get a handle immediately and control the lifecycle.

```python
job = client.pipeline.run(files=["doc.pdf"])

# Poll status
status = job.status()
print(f"{status.stages_completed}/{status.total_stages}: {status.current_service}")

# Wait with progress callback
pkg = job.wait_for_completion(
    poll_interval=5.0,
    timeout=1800.0,
    on_progress=lambda status, elapsed: print(f"  {status} ({elapsed:.0f}s)"),
)

# Cancel
job.cancel()
```

### Resumable pipelines

If a pipeline fails partway through, completed stages are checkpointed:

```python
from latence import JobError

try:
    pkg = job.wait_for_completion()
except JobError as e:
    if e.is_resumable:
        pkg = job.resume().wait_for_completion()  # continues from checkpoint
    else:
        raise
```

### Job statuses

| Status | Meaning |
|--------|---------|
| `QUEUED` | Waiting to start |
| `IN_PROGRESS` | Processing |
| `COMPLETED` | Finished successfully |
| `CACHED` / `PULLED` | Results from cache/storage |
| `RESUMABLE` | Failed mid-pipeline; call `job.resume()` |
| `FAILED` | Pipeline failed |
| `CANCELLED` | Cancelled by user |

---

## Direct API Access

Every Latence AI service is also available individually via `client.experimental` -- full granular control, no pipeline overhead. This is expert / developer mode: you call exactly the service you need with exactly the parameters you want.

**Document intelligence services:**

```python
# Document processing (OCR, layout, markdown)
result = client.experimental.document_intelligence.process(file_path="doc.pdf")

# Entity extraction
result = client.experimental.extraction.extract(
    text="Apple Inc. was founded by Steve Jobs in Cupertino.",
    config={"label_mode": "generated"},
)

# Relation extraction / knowledge graph
result = client.experimental.ontology.build_graph(text="...", entities=[...])

# PII detection and redaction
result = client.experimental.redaction.detect_pii(
    text="John Smith, SSN 123-45-6789",
    config={"mode": "balanced", "redact": True, "redaction_mode": "mask"},
)

# Text chunking (4 strategies: character, token, semantic, hybrid)
result = client.experimental.chunking.chunk(text="...", strategy="hybrid", chunk_size=512)
```

**Embedding and retrieval services:**

```python
# Dense embeddings (256-1024d, Matryoshka)
result = client.experimental.embed.dense(text="Hello world", dimension=512)

# ColBERT token-level embeddings (late interaction retrieval)
result = client.experimental.colbert.embed(text="Hello world")

# ColPali vision-language page embeddings
result = client.experimental.colpali.embed(file_path="page.png")
```

**Groundedness & phantom-hallucination scoring ([Trace](docs/trace.md)):**

```python
# RAG groundedness -- was the response actually supported by the context?
r = client.experimental.trace.rag(
    response_text="Paris is the capital of France.",
    raw_context="France's capital city is Paris.",
)
print(r.score, r.band, r.context_coverage_ratio)

# Agentic-code phantom scoring with cross-turn session chaining
t1 = client.experimental.trace.code(
    response_text="def add(a, b): return a + b",
    raw_context="# utils.py\ndef sub(a, b): return a - b",
    response_language_hint="python",
)
t2 = client.experimental.trace.code(
    response_text="def mul(a, b): return a * b",
    raw_context="# utils.py\ndef sub(a, b): return a - b",
    response_language_hint="python",
    session_state=t1.next_session_state,   # round-trip the opaque state
)

# Stateless session rollup -- noise / drift / waste / reason-code histogram
rollup = client.experimental.trace.rollup(turns=[t1, t2])
print(rollup.noise_pct, rollup.recommendations)
```

> For production document intelligence workloads, use `client.pipeline`. Pipelines provide structured data packages, quality metrics, resumability, and are covered by Enterprise SLAs.
>
> The Direct API is open to all. We actively welcome feedback -- if something is missing or could work better, [let us know](https://github.com/latenceai/latence-python/issues).

See [SDK_TUTORIAL.md](SDK_TUTORIAL.md) for complete documentation of every service and parameter.

---

## Dataset Intelligence

Turn pipeline outputs into corpus-level knowledge graphs, ontologies, and structured datasets with incremental ingestion. Feed the output of any Latence pipeline into Dataset Intelligence to extract entities, resolve duplicates, build knowledge graphs with RotatE link prediction, and induce ontological concepts.

```python
# Dataset Intelligence consumes pipeline stage outputs.
# Use the portal's Dataset Intelligence UI to upload pipeline results,
# or submit programmatically via the SDK:
di = client.experimental.dataset_intelligence_service

# Create a new dataset from pipeline output (dict with stage keys)
job = di.run(input_data=pipeline_output, return_job=True)
print(f"Job submitted: {job.job_id}")
# Poll status at GET /api/v1/pipeline/{job.job_id}

# Append new documents to an existing dataset
delta = di.run(
    input_data=new_pipeline_output,
    dataset_id="ds_existing_id",  # appends to existing dataset
    return_job=True,
)
```

Four processing tiers:

| Tier | Method | What it does |
|------|--------|-------------|
| Tier 1 | `di.enrich()` | Semantic feature vectors (CPU-only, fast) |
| Tier 2 | `di.build_graph()` | Entity resolution, knowledge graph, link prediction |
| Tier 3 | `di.build_ontology()` | Concept clustering, hierarchy induction |
| Full | `di.run()` | All 3 tiers sequentially |

See [docs/dataset_intelligence.md](docs/dataset_intelligence.md) for the complete API reference, input format, delta ingestion details, and pricing.

---

## Error Handling

```python
from latence import (
    LatenceError,           # base for all SDK errors
    AuthenticationError,    # 401
    InsufficientCreditsError,  # 402
    RateLimitError,         # 429 (has retry_after)
    JobError,               # pipeline failed (has job_id, is_resumable)
    JobTimeoutError,        # wait exceeded timeout
    TransportError,         # network / DNS / connection errors
)

try:
    job = client.pipeline.run(files=["doc.pdf"])
    pkg = job.wait_for_completion(timeout=600)
except AuthenticationError:
    print("Invalid API key")
except InsufficientCreditsError:
    print("No credits remaining")
except RateLimitError as e:
    print(f"Rate limited -- retry after {e.retry_after}s")
except JobTimeoutError as e:
    print(f"Pipeline {e.job_id} did not finish in time")
except JobError as e:
    if e.is_resumable:
        print(f"Resumable failure at {e.error_code}")
    else:
        print(f"Pipeline failed: {e.message}")
except TransportError:
    print("Network error")
```

The SDK automatically retries on 429, 5xx with exponential backoff and jitter (default: 2 retries, respects `Retry-After`).

---

## Configuration

```bash
export LATENCE_API_KEY="lat_your_key"
```

```python
client = Latence(
    api_key="lat_...",       # or LATENCE_API_KEY env var
    base_url="https://...",  # or LATENCE_BASE_URL env var
    timeout=60.0,            # request timeout (default: 60s)
    max_retries=2,           # retry attempts (default: 2)
)
```

### Debug logging

```python
import latence
latence.setup_logging("DEBUG")  # logs all HTTP requests and responses
```

---

## Resources

| | |
|---|---|
| **Full Tutorial** | [SDK_TUTORIAL.md](SDK_TUTORIAL.md) -- every feature, every parameter |
| **API Reference** | [docs.latence.ai](https://docs.latence.ai) |
| **Portal** | [app.latence.ai](https://app.latence.ai) |

---

<p align="center">
  <sub>MIT License &bull; <a href="https://latence.ai">latence.ai</a></sub>
</p>
