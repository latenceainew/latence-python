# Pipelines

The Data Intelligence Pipeline chains multiple AI services into a single async job. You submit documents, configure services, and get back a structured `DataPackage`.

The pipeline executes services as a **directed acyclic graph (DAG)**, not a linear chain. Independent branches run in parallel:

```
                    ┌─── extraction ──── relation_extraction
                    │
document_intelligence ─┼─── redaction
                    │
                    └─── compression
```

You declare which services you want. The pipeline handles ordering, dependencies, and parallel execution automatically.

## Quick Start

```python
from latence import Latence

client = Latence()  # reads LATENCE_API_KEY from environment

# Smart defaults: OCR -> Entity Extraction -> Relation Extraction
job = client.pipeline.run(files=["contract.pdf"])
pkg = job.wait_for_completion()

print(pkg.document.markdown)
print(pkg.entities.summary)
print(pkg.knowledge_graph.summary.total_relations)
pkg.download_archive("./results.zip")
```

## Smart Defaults

When you provide files with no `steps`, the pipeline auto-applies:

**Document Intelligence -> Entity Extraction -> Relation Extraction**

```python
job = client.pipeline.run(files=["report.pdf"])
```

## Explicit Steps

Configure each service individually via `steps`:

```python
job = client.pipeline.run(
    files=["contract.pdf"],
    name="Legal Analysis",
    steps={
        "ocr": {"mode": "performance", "output_format": "markdown"},
        "redaction": {"mode": "balanced", "redact": True, "redaction_mode": "mask"},
        "extraction": {
            "label_mode": "hybrid",
            "user_labels": ["person", "organization", "date"],
            "threshold": 0.3,
        },
        "relation_extraction": {"resolve_entities": True, "optimize_relations": True},
        "compression": {"compression_rate": 0.5},
    },
)
pkg = job.wait_for_completion()
```

### Available Pipeline Services

| Step | Aliases | Description |
|------|---------|-------------|
| `document_intelligence` | `ocr`, `doc_intel` | OCR, layout detection, markdown extraction |
| `extraction` | `extract` | Zero-shot named entity recognition |
| `relation_extraction` | `ontology`, `knowledge_graph`, `graph` | Relation extraction, knowledge graph construction |
| `redaction` | `redact` | PII detection, masking, or synthetic replacement |
| `compression` | `compress` | Intelligent token-level text compression |

Steps are automatically sorted into the correct DAG execution order.

## Input Types

### Files

```python
job = client.pipeline.run(files=["doc.pdf"])
job = client.pipeline.run(file_urls=["https://example.com/doc.pdf"])
```

### Text (skips OCR)

```python
job = client.pipeline.run(
    text="Apple Inc. was founded by Steve Jobs in Cupertino.",
    steps={"extraction": {"label_mode": "generated"}},
)
```

### Pre-extracted Entities (relation extraction only)

```python
job = client.pipeline.run(
    entities=[{"text": "Apple", "label": "ORG", "start": 0, "end": 5, "score": 0.98}],
    steps={"knowledge_graph": {"resolve_entities": True}},
)
```

## Pipeline Builder

The `PipelineBuilder` provides a typed, chainable API with client-side validation:

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

The builder validates parameters client-side: threshold ranges, valid modes, valid embedding dimensions, and duplicate service detection. Invalid values raise `ValueError` before the request leaves your machine.

`store_intermediate` defaults to `True` -- every stage's output is preserved in the final `DataPackage`.

### YAML Configuration

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
config = PipelineBuilder.from_yaml("pipeline.yml")  # returns a PipelineBuilder
job = client.pipeline.submit(config, files=["contract.pdf"])
```

`from_yaml()` returns a `PipelineBuilder`, so you can chain additional methods before calling `.build()`.

### Validate Before Running

```python
result = client.pipeline.validate(config, files=["doc.pdf"])
print(result.valid)          # True
print(result.auto_injected)  # ["document_intelligence"]
print(result.warnings)       # []
```

## The Data Package

Every pipeline returns a `DataPackage`:

| Section | Contents | Present when |
|---------|----------|-------------|
| `pkg.document` | Markdown, per-page content, metadata | OCR ran |
| `pkg.entities` | Entity list, summary (total, by_type, avg_confidence) | Extraction ran |
| `pkg.knowledge_graph` | Entities, relations, graph summary | Relation Extraction ran |
| `pkg.redaction` | Cleaned text, PII list, summary | Redaction ran |
| `pkg.compression` | Compressed text, ratio, tokens saved | Compression ran |
| `pkg.quality` | Per-stage report, confidence scores, cost | Always |

### Export

```python
pkg.download_archive("./results.zip")                # organized ZIP
merged = pkg.merge(save_to="./results.json")          # consolidated JSON
```

## Job Lifecycle

Pipelines are async jobs. You get a handle immediately.

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

### Job Statuses

| Status | Meaning |
|--------|---------|
| `QUEUED` | Waiting to start |
| `IN_PROGRESS` | Processing |
| `COMPLETED` | Finished successfully |
| `CACHED` / `PULLED` | Results from cache/storage |
| `RESUMABLE` | Failed mid-pipeline; call `job.resume()` |
| `FAILED` | Pipeline failed |
| `CANCELLED` | Cancelled by user |

### Resumable Pipelines

If a pipeline fails partway through, completed stages are checkpointed:

```python
from latence import JobError

try:
    pkg = job.wait_for_completion()
except JobError as e:
    if e.is_resumable:
        pkg = job.resume().wait_for_completion()
    else:
        raise
```

## Async / Await

```python
from latence import AsyncLatence

async with AsyncLatence() as client:
    job = await client.pipeline.run(files=["doc.pdf"])
    pkg = await job.wait_for_completion()
```

## Error Handling

```python
from latence import (
    LatenceError,
    AuthenticationError,
    RateLimitError,
    InsufficientCreditsError,
    JobError,
    JobTimeoutError,
    TransportError,
    PipelineValidationError,
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
        print(f"Resumable failure: {e.message}")
    else:
        print(f"Pipeline failed: {e.message}")
except PipelineValidationError as e:
    print(f"Validation: {e.errors}")
except TransportError:
    print("Network error")
```

The SDK automatically retries on 429, 5xx with exponential backoff and jitter (default: 2 retries, respects `Retry-After`).

## Intermediate Results

Access per-stage download URLs while a job is running or after completion:

```python
stages = job.intermediate_results()
for stage in stages:
    print(f"{stage.service}: {stage.download_url}")
```

## API Reference

### Submit Pipeline

```
POST /api/v1/pipeline/execute
```

### Poll Status

```
GET /api/v1/pipeline/{job_id}
```

### Retrieve Result

```
GET /api/v1/pipeline/{job_id}/result
```

### Cancel Pipeline

```
DELETE /api/v1/pipeline/{job_id}
```

### Resume Pipeline

```
POST /api/v1/pipeline/{job_id}/resume
```

See [SDK_TUTORIAL.md](../SDK_TUTORIAL.md) for complete parameter documentation.
