# Latence AI Python SDK -- Complete Tutorial

This tutorial covers every feature of the Latence AI Python SDK: the **Data Intelligence Pipeline** for multi-stage document processing, **direct API access** to individual services, job management, credits, async usage, file handling, error handling, and configuration.

> **Prerequisites**: Python 3.10+, a Latence API key from [latence.ai](https://latence.ai/keys)

---

## Table of Contents

1. [Installation and Authentication](#1-installation-and-authentication)
2. [Quick Start](#2-quick-start)
3. [Pipeline: Smart Defaults](#3-pipeline-smart-defaults)
4. [Pipeline: Fluent Builder](#4-pipeline-fluent-builder)
5. [Pipeline: PipelineConfig Object](#5-pipeline-pipelineconfig-object)
6. [Pipeline: YAML Config](#6-pipeline-yaml-config)
7. [Job Lifecycle](#7-job-lifecycle)
8. [DataPackage](#8-datapackage)
9. [Direct API: Document Intelligence](#9-direct-api-document-intelligence)
10. [Direct API: Entity Extraction](#10-direct-api-entity-extraction)
11. [Direct API: Redaction](#11-direct-api-redaction)
12. [Direct API: Relation Extraction](#12-direct-api-relation-extraction)
13. [Direct API: Compression](#13-direct-api-compression)
14. [Direct API: Chunking](#14-direct-api-chunking)
15. [Direct API: Enrichment](#15-direct-api-enrichment)
16. [Direct API: Embeddings (Unified)](#16-direct-api-embeddings-unified)
17. [Direct API: Legacy Embeddings](#17-direct-api-legacy-embeddings)
18. [Jobs Service](#18-jobs-service)
19. [Credits](#19-credits)
20. [Async Usage](#20-async-usage)
21. [File Handling](#21-file-handling)
22. [Error Handling](#22-error-handling)
23. [Configuration](#23-configuration)

---

## 1. Installation and Authentication

```bash
pip install latence
```

### API Key

```python
from latence import Latence

# Option A: Pass directly
client = Latence(api_key="lat_your_key_here")

# Option B: Environment variable (recommended)
#   export LATENCE_API_KEY=lat_your_key_here
client = Latence()  # reads from LATENCE_API_KEY

```

### Context Manager

```python
with Latence() as client:
    result = client.experimental.extraction.extract(text="Apple Inc. in Cupertino")
    print(result.entities)
# client.close() is called automatically
```

---

## 2. Quick Start

Turn a PDF into structured knowledge in 4 lines:

```python
from latence import Latence

client = Latence()
job = client.pipeline.run(files=["contract.pdf"])
pkg = job.wait_for_completion()

print(pkg.document.markdown)       # clean extracted text
print(pkg.entities.summary)        # entity counts by type
print(pkg.knowledge_graph.relations)  # subject-predicate-object triples
```

---

## 3. Pipeline: Smart Defaults

`client.pipeline.run()` is the primary entry point. It auto-injects Document Intelligence as the first stage and applies sensible defaults.

### From files

```python
# Single file -- auto-applies: OCR -> Extraction -> Relation Extraction
job = client.pipeline.run(files=["report.pdf"])

# Multiple files
job = client.pipeline.run(files=["doc1.pdf", "doc2.pdf", "doc3.pdf"])

# Named pipeline
job = client.pipeline.run(
    files=["contract.pdf"],
    name="Legal Contract Analysis",
)
```

### From text

```python
job = client.pipeline.run(
    text="Apple Inc. was founded in 1976 by Steve Jobs in Cupertino, California.",
    steps={"extraction": {"label_mode": "hybrid", "user_labels": ["person", "organization", "location"]}},
)
pkg = job.wait_for_completion()
print(pkg.entities.items)
```

### From entities (relation extraction only)

```python
entities = [
    {"text": "Apple", "label": "ORG", "start": 0, "end": 5, "score": 0.98},
    {"text": "Cupertino", "label": "LOC", "start": 35, "end": 44, "score": 0.95},
]
job = client.pipeline.run(
    entities=entities,
    steps={"knowledge_graph": {"resolve_entities": True}},
)
```

### With explicit steps

The `steps` dict lets you configure each stage. Keys are short aliases (case-insensitive):

| Service | Accepted aliases |
|---------|-----------------|
| Document Intelligence | `document_intelligence`, `ocr`, `doc_intel` |
| Entity Extraction | `extraction`, `extract` |
| Relation Extraction | `relation_extraction`, `ontology`, `knowledge_graph`, `graph` |
| PII Redaction | `redaction`, `redact` |
| Text Compression | `compression`, `compress` |

> **Note**: Chunking and Enrichment are standalone APIs only -- they are not available as pipeline steps. Use `client.experimental.chunking.chunk()` and `client.experimental.enrichment.enrich()` respectively.

```python
job = client.pipeline.run(
    files=["financial_report.pdf"],
    steps={
        "ocr": {"mode": "performance", "output_format": "markdown"},
        "redaction": {"mode": "strict", "redact": True, "redaction_mode": "mask"},
        "extraction": {
            "label_mode": "hybrid",
            "user_labels": ["person", "organization", "monetary_amount", "date"],
            "threshold": 0.3,
        },
        "relation_extraction": {
            "resolve_entities": True,
            "optimize_relations": True,
            "kg_output_format": "property_graph",
        },
    },
    name="Financial Report Analysis",
)
pkg = job.wait_for_completion()
```

### With file URLs

```python
job = client.pipeline.run(
    file_urls=["https://example.com/document.pdf"],
    steps={"ocr": {"mode": "default"}},
)
```

---

## 4. Pipeline: Fluent Builder

The `PipelineBuilder` provides a chainable API for constructing pipelines:

```python
from latence import PipelineBuilder

config = (
    PipelineBuilder()
    .doc_intel(mode="performance", output_format="markdown")
    .redaction(mode="balanced", redact=True, redaction_mode="mask")
    .extraction(
        label_mode="hybrid",
        user_labels=["person", "organization", "location"],
        threshold=0.3,
        enable_refinement=True,
    )
    .relation_extraction(
        resolve_entities=True,
        optimize_relations=True,
        predict_missing_relations=True,
        kg_output_format="custom",
    )
    .store_intermediate()
    .build()
)

job = client.pipeline.submit(config, files=["contract.pdf"])
pkg = job.wait_for_completion()
```

### Available builder methods

```python
builder = PipelineBuilder()

# Document Intelligence
builder.doc_intel(
    mode="default",              # "default" or "performance"
    output_format="markdown",    # "markdown", "json", "html", "xlsx"
    max_pages=None,              # limit pages processed
    use_ocr_for_image_block=False,  # extract text from embedded images (+$0.25/1k pages)
)
# Layout detection, chart/seal recognition, and auto-rotate are pre-configured
# for optimal pipeline results. For full control, use the direct API.

# Entity Extraction
builder.extraction(
    threshold=0.3,               # confidence threshold (0.0-1.0)
    user_labels=["person"],      # labels to extract
    label_mode="generated",      # "user", "hybrid", "generated"
    enable_refinement=False,     # LLM refinement (1.5x credits)
    enforce_refinement=False,    # refine ALL entities (2.5x credits)
    refinement_threshold=0.5,
    chunk_size=1024,
    flat_ner=True,               # non-overlapping entities
    multi_label=False,           # multiple labels per span
)

# Relation Extraction (also available as .ontology() or .knowledge_graph())
builder.relation_extraction(
    resolve_entities=False,            # merge duplicates (2.0x credits)
    optimize_relations=True,           # refine relation labels (1.5x credits)
    predict_missing_relations=False,   # predict implicit links (2.5x credits)
    relation_threshold=0.5,
    kg_output_format="custom",         # "custom", "property_graph", "rdf"
)

# Redaction
builder.redaction(
    mode="balanced",             # "balanced", "strict", "recall", "precision"
    threshold=0.3,
    redact=True,
    redaction_mode="mask",       # "mask" or "replace"
    chunk_size=1024,
)
# Full LLM refinement is always enabled in pipeline redaction for quality.
# For manual refinement control, use the direct API.

# Compression
builder.compression(
    compression_rate=0.5,        # fraction of tokens to remove (0.0-1.0)
    force_preserve_digit=True,   # preserve numeric tokens (default: True)
    force_tokens=None,           # tokens to always keep, e.g. ["API", "JSON"]
    apply_toon=False,            # TOON encoding (+$0.50/1M tokens)
    chunk_size=4096,             # max tokens per chunk (default: 4096)
    fallback_mode=True,
)

# Embedding (experimental pipeline step)
builder.embedding(
    dimension=512,               # 256, 512, 768, or 1024
    encoding_format="float",     # "float" or "base64"
)

# ColBERT (experimental pipeline step)
builder.colbert(
    is_query=False,
    query_expansion=False,
)

# ColPali (experimental pipeline step)
builder.colpali(is_query=False)

# Pipeline options
builder.store_intermediate()     # keep results from each stage
builder.strict()                 # disable auto-injection of services

config = builder.build()
```

> **Note**: Chunking is not available as a pipeline step -- `builder.chunking()` raises `NotImplementedError`. Use `client.experimental.chunking.chunk()` for standalone text chunking.

### Dynamic pipeline construction with `add()`

For programmatic pipeline building (e.g., from user configuration or feature flags), use the generic `add()` method. It resolves aliases and rejects duplicates:

```python
builder = PipelineBuilder()

# Build a pipeline from a list of step names
requested_steps = ["ocr", "extraction", "relation_extraction"]
for step in requested_steps:
    builder.add(step)

# add() accepts keyword config just like the named methods
builder.add("redaction", mode="balanced", redact=True)

config = builder.build()
job = client.pipeline.submit(config, files=["doc.pdf"])
```

Duplicate services raise immediately:

```python
builder = PipelineBuilder().extraction()
builder.add("extract")  # ValueError: Service 'extraction' already added
```

### Strict mode vs auto-injection

By default, `build()` auto-injects missing DAG parents. For example, if you only add `extraction` with file input, `document_intelligence` is injected automatically. In **strict mode**, missing parents raise `PipelineValidationError` instead:

```python
from latence import PipelineValidationError

# Auto-injection (default) -- document_intelligence added for you
config = PipelineBuilder().extraction().relation_extraction().build()
job = client.pipeline.submit(config, files=["doc.pdf"])  # works

# Strict mode -- must be explicit
try:
    config = PipelineBuilder().extraction().relation_extraction().strict().build()
    client.pipeline.submit(config, files=["doc.pdf"])
except PipelineValidationError as e:
    print(e.errors)  # missing document_intelligence
```

### Pipeline execution model

The pipeline worker executes services as a **directed acyclic graph (DAG)**, not a linear chain. Services that share the same parent can run concurrently:

```
                    ┌─── extraction ──── relation_extraction
                    │
document_intelligence ─┼─── redaction
                    │
                    └─── compression
```

This means `extraction`, `redaction`, and `compression` all run in parallel once `document_intelligence` completes. The SDK and worker automatically handle ordering -- you just declare which services you want.

---

## 5. Pipeline: PipelineConfig Object

For maximum control, construct a `PipelineConfig` directly:

```python
from latence import PipelineConfig, ServiceConfig

config = PipelineConfig(
    services=[
        ServiceConfig(service="document_intelligence", config={
            "mode": "performance",
            "output_format": "markdown",
        }),
        ServiceConfig(service="extraction", config={
            "label_mode": "hybrid",
            "user_labels": ["person", "organization", "date"],
            "threshold": 0.3,
        }),
        ServiceConfig(service="ontology", config={  # Relation Extraction
            "resolve_entities": True,
            "kg_output_format": "property_graph",
        }),
    ],
    store_intermediate=True,
    strict_mode=False,
    name="My Custom Pipeline",
)

job = client.pipeline.submit(config, files=["doc.pdf"])
pkg = job.wait_for_completion()
```

---

## 6. Pipeline: YAML Config

Define pipelines in YAML files for version control and reuse:

```yaml
# pipeline.yml
name: contract-analysis
store_intermediate: true

steps:
  document_intelligence:
    mode: performance
    output_format: markdown

  extraction:
    label_mode: hybrid
    user_labels:
      - person
      - organization
      - location
      - date
      - monetary_amount
    threshold: 0.3

  relation_extraction:
    resolve_entities: true
    optimize_relations: true
    kg_output_format: custom
```

Load and run:

```python
from latence import PipelineBuilder

config = PipelineBuilder.from_yaml("pipeline.yml")
job = client.pipeline.submit(config, files=["contract.pdf"])
pkg = job.wait_for_completion()
```

> Requires `pyyaml`: `pip install pyyaml`

---

## 7. Job Lifecycle

Pipelines are async/job-based. `run()` and `submit()` return a `Job` handle immediately.

```python
job = client.pipeline.run(files=["doc.pdf"], name="My Analysis")

# Properties
print(job.id)      # "pipe_abc123"
print(job.name)    # "My Analysis"

# Poll status
status = job.status()
print(status.status)            # "QUEUED", "IN_PROGRESS", "COMPLETED", "FAILED", "CANCELLED"
print(status.stages_completed)  # 1
print(status.total_stages)      # 3
print(status.current_service)   # "extraction"

# Wait for completion (blocking) -- returns DataPackage
pkg = job.wait_for_completion(
    poll_interval=5.0,    # seconds between polls (default 5)
    timeout=1800.0,       # max wait in seconds (default 1800 = 30 min)
)

# With progress callback -- invoked on each poll
pkg = job.wait_for_completion(
    on_progress=lambda status, elapsed: print(f"  {status} ({elapsed:.0f}s)"),
)

# Optionally save results to disk during wait
pkg = job.wait_for_completion(save_to_disk="./results.zip")

# Cancel a running job
job.cancel()

# Lazy access to DataPackage (cached after first call)
pkg = job.data_package
```

### Resumable jobs

If a pipeline fails partway through, it may enter `RESUMABLE` status. Completed stages are checkpointed and only remaining stages re-execute on resume:

```python
try:
    pkg = job.wait_for_completion()
except JobError as e:
    if e.is_resumable:
        print(f"Job failed at a stage but is resumable: {e.message}")
        pkg = job.resume().wait_for_completion()
    else:
        raise
```

### Intermediate results and report

Access per-stage download URLs and the structured pipeline report while a job is running or after completion:

```python
# Per-stage download URLs (presigned B2 URLs to results.jsonl)
stages = job.intermediate_results()
for stage in stages:
    print(f"{stage.service}: {stage.download_url}")

# Structured pipeline report (dataset facts, per-stage metrics)
report = job.report
if report:
    print(report)
```

### Validate before running

Check a pipeline configuration without executing it:

```python
from latence import PipelineBuilder

builder = PipelineBuilder().doc_intel().extraction().relation_extraction()
result = client.pipeline.validate(builder, files=["doc.pdf"])
print(result.valid)       # True/False
print(result.errors)      # list of errors
print(result.warnings)    # list of warnings
print(result.auto_injected)  # services auto-added
```

### Get available stages

List the per-stage download links for a completed job:

```python
stages = client.pipeline.stages("pipe_abc123")
for s in stages:
    print(f"{s.service}: {s.download_url}")
```

### Status values

| Status | Meaning |
|--------|---------|
| `QUEUED` | Waiting to start |
| `IN_PROGRESS` | Currently processing |
| `COMPLETED` | Finished successfully |
| `CACHED` | Results retrieved from cache |
| `PULLED` | Results pulled from storage |
| `RESUMABLE` | Failed partway through; call `job.resume()` to continue |
| `FAILED` | Pipeline failed |
| `CANCELLED` | Cancelled by user |

---

## 8. DataPackage

The `DataPackage` is the structured output of a pipeline. It composes raw stage outputs into organized sections.

```python
pkg = job.wait_for_completion()
```

### Document section

```python
if pkg.document:
    print(pkg.document.markdown)                    # full extracted text
    print(pkg.document.pages)                       # per-page markdown list
    print(pkg.document.metadata.filename)            # original filename
    print(pkg.document.metadata.pages_processed)     # number of pages
    print(pkg.document.metadata.content_type)        # "markdown", "json", etc.
    print(pkg.document.metadata.processing_mode)     # "default" or "performance"
```

### Entities section

```python
if pkg.entities:
    print(pkg.entities.summary.total)           # total entity count
    print(pkg.entities.summary.by_type)         # {"person": 5, "organization": 3, ...}
    print(pkg.entities.summary.unique_labels)   # ["person", "organization", ...]
    print(pkg.entities.summary.avg_confidence)  # 0.87

    for entity in pkg.entities.items:
        print(f"{entity.text} [{entity.label}] score={entity.score}")
```

### Knowledge graph section

```python
if pkg.knowledge_graph:
    print(pkg.knowledge_graph.summary.total_entities)
    print(pkg.knowledge_graph.summary.total_relations)
    print(pkg.knowledge_graph.summary.entity_types)    # ["PERSON", "ORG", ...]
    print(pkg.knowledge_graph.summary.relation_types)  # ["works_at", "located_in", ...]

    for entity in pkg.knowledge_graph.entities:
        print(f"{entity.text} [{entity.label}]")

    for relation in pkg.knowledge_graph.relations:
        print(f"{relation.entity1} --[{relation.relation_type}]--> {relation.entity2}")
```

### Redaction section

```python
if pkg.redaction:
    print(pkg.redaction.redacted_text)              # cleaned text with PII masked
    print(pkg.redaction.summary.total_pii)          # number of PII entities found

    for pii in pkg.redaction.pii_detected:
        print(f"{pii.text} [{pii.label}] at {pii.start}:{pii.end}")
```

### Compression section

```python
if pkg.compression:
    print(pkg.compression.compressed_text)
    print(pkg.compression.summary.compression_ratio)  # e.g. 0.45
    print(pkg.compression.summary.tokens_saved)       # e.g. 1250
```

### Chunking section

```python
if pkg.chunking:
    print(pkg.chunking.summary.num_chunks)
    print(pkg.chunking.summary.strategy)       # "hybrid"
    print(pkg.chunking.summary.chunk_size)     # target chunk size parameter

    for chunk in pkg.chunking.chunks:
        print(f"Chunk: {chunk}")
```

### Enrichment section

```python
if pkg.enrichment:
    print(pkg.enrichment.summary.num_chunks)
    print(pkg.enrichment.summary.strategy)
    print(pkg.enrichment.summary.features_computed)  # ["quality", "density", ...]

    for chunk in pkg.enrichment.chunks:
        print(chunk)

    for name, data in pkg.enrichment.features.items():
        print(f"{name}: {data}")
```

### Quality report

```python
print(pkg.quality.total_cost_usd)
print(pkg.quality.total_processing_time_ms)

for stage in pkg.quality.stages:
    print(f"{stage.service}: {stage.status} ({stage.processing_time_ms}ms, ${stage.credits_used})")

# Confidence scores
print(pkg.quality.confidence.entity_avg_confidence)
print(pkg.quality.confidence.graph_completeness)
print(pkg.quality.confidence.ocr_quality)
```

### Download as ZIP archive

```python
path = pkg.download_archive("./results.zip")
print(f"Saved to {path}")
```

Archive structure:
```
{pipeline_name}/
  README.md
  document.md
  pages/
    page_001.md
    page_002.md
  entities.json
  knowledge_graph.json
  redaction.json          (if redaction ran)
  compression.json        (if compression ran)
  chunking.json           (if chunking ran)
  enrichment.json         (if enrichment ran)
  quality_report.json
  metadata.json
```

### Merge into flat dict

```python
merged = pkg.merge()
# {
#   "id": "pipe_xxx",
#   "name": "My Pipeline",
#   "status": "COMPLETED",
#   "created_at": "2025-...",
#   "documents": [{"filename": "doc.pdf", "markdown": "...", "entities": [...], ...}],
#   "summary": {"documents": 1, "pages": 5, "entities": {...}, "relations": {...}, ...},
# }

# Save merged output directly to a JSON file:
pkg.merge(save_to="./results.json")
```

---

## 9. Direct API: Document Intelligence

Process documents directly without a full pipeline.

```python
di = client.experimental.document_intelligence
```

### From file path

```python
result = di.process(
    file_path="report.pdf",
    mode="default",             # "default" or "performance"
    output_format="markdown",   # "markdown", "json", "html", "xlsx"
)
print(result.text)
print(result.pages)
print(result.credits_used)
```

### From base64

```python
import base64
with open("doc.pdf", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

result = di.process(
    file_base64=b64,
    filename="doc.pdf",
    mode="performance",
)
```

### From URL

```python
result = di.process(
    file_url="https://example.com/report.pdf",
    mode="default",
)
```

### Performance mode with all options

```python
result = di.process(
    file_path="complex_report.pdf",
    mode="performance",
    output_format="markdown",
    max_pages=50,
    use_layout_detection=True,
    use_chart_recognition=True,
    pipeline_options={
        "use_seal_recognition": False,
        "use_ocr_for_image_block": True,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
    },
)
```

### Refine previously processed pages

```python
result = di.refine(
    pages_result=result.raw_pages,
    output_format="markdown",
    refine_options={
        "merge_tables": True,
        "relevel_titles": True,
        "concatenate_pages": False,
    },
)
```

### Async job mode

```python
submitted = di.process(
    file_path="big_doc.pdf",
    mode="performance",
    return_job=True,
)
print(submitted.job_id)

# Poll with the jobs service
result = client.jobs.wait(submitted.job_id)
data = client.jobs.retrieve(submitted.job_id)
```

---

## 10. Direct API: Entity Extraction

```python
ext = client.experimental.extraction
```

### Auto-generated labels

```python
result = ext.extract(
    text="Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976.",
    config={"label_mode": "generated"},
)
for entity in result.entities:
    print(f"{entity.text} [{entity.label}] confidence={entity.score:.2f}")
```

### User-defined labels

```python
result = ext.extract(
    text="Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976.",
    config={
        "label_mode": "user",
        "user_labels": ["person", "organization", "location", "date"],
        "threshold": 0.3,
    },
)
```

### Hybrid mode

```python
result = ext.extract(
    text="The contract between Acme Corp and John Doe was signed on January 15, 2025 for $500,000.",
    config={
        "label_mode": "hybrid",
        "user_labels": ["person", "organization", "monetary_amount"],
        "threshold": 0.25,
    },
)
```

### With LLM refinement

```python
result = ext.extract(
    text="Dr. Sarah Chen published her findings at MIT last Tuesday.",
    config={
        "label_mode": "hybrid",
        "user_labels": ["person", "organization", "date"],
        "enable_refinement": True,
        "refinement_threshold": 0.5,
    },
)
```

### Custom labels

```python
result = ext.extract(
    text="The Tesla Model 3 uses a 75 kWh battery pack.",
    custom_labels=[
        {"name": "vehicle_model", "description": "Car make and model"},
        {"name": "technical_spec", "description": "Technical specification or measurement"},
    ],
)
```

### All configuration options

```python
result = ext.extract(
    text="...",
    config={
        "label_mode": "hybrid",      # "user" | "hybrid" | "generated"
        "user_labels": ["person"],    # required for "user" and "hybrid" modes
        "threshold": 0.3,            # confidence threshold (0.0-1.0)
        "flat_ner": True,            # non-overlapping entities
        "multi_label": False,        # one label per span
        "chunk_size": 1024,          # token chunk size for processing
        "enable_refinement": False,  # LLM refinement (1.5x credits)
        "enforce_refinement": False, # refine ALL entities (2.5x credits)
        "refinement_threshold": 0.5, # score cutoff for refinement
    },
    request_id="my-tracking-id",
)
```

---

## 11. Direct API: Redaction

```python
red = client.experimental.redaction
```

### Detect PII

```python
result = red.detect_pii(
    text="John Smith lives at 123 Main Street. His SSN is 123-45-6789. Contact: john@example.com",
    config={"mode": "balanced", "redact": False},
)
for pii in result.entities:
    print(f"{pii.text} [{pii.label}] score={pii.score:.2f}")
```

### Detect and mask PII

```python
result = red.detect_pii(
    text="John Smith lives at 123 Main Street. His SSN is 123-45-6789.",
    config={
        "mode": "balanced",
        "redact": True,
        "redaction_mode": "mask",  # replaces PII with [LABEL] tokens
    },
)
print(result.redacted_text)
# "[PERSON] lives at [ADDRESS]. His SSN is [SSN]."
```

### Replace PII with synthetic data

```python
result = red.detect_pii(
    text="John Smith lives at 123 Main Street. His SSN is 123-45-6789.",
    config={
        "mode": "balanced",
        "redact": True,
        "redaction_mode": "replace",  # generates synthetic replacement data
    },
)
print(result.redacted_text)
# "Jane Doe lives at 456 Oak Avenue. His SSN is 987-65-4321."
```

### Strict mode with refinement

```python
result = red.detect_pii(
    text="Dr. Chen mentioned that patient #4521 has an appointment at Mass General on Tuesday.",
    config={
        "mode": "strict",               # higher recall (1.3x credits)
        "redact": True,
        "redaction_mode": "mask",
        "enable_refinement": True,       # LLM refinement (1.5x credits)
        "refinement_threshold": 0.5,
    },
)
```

### All configuration options

```python
result = red.detect_pii(
    text="...",
    config={
        "mode": "balanced",              # "balanced" | "strict" | "recall" | "precision"
        "threshold": 0.3,               # confidence threshold (0.0-1.0)
        "redact": True,                 # detect only vs. detect and redact
        "redaction_mode": "mask",        # "mask" | "replace"
        "normalize_scores": True,        # Platt scaling for scores
        "chunk_size": 1024,             # token chunk size
        "enable_refinement": False,      # LLM refinement (1.5x credits)
        "enforce_refinement": False,     # refine ALL entities (2.5x credits)
        "refinement_threshold": 0.5,
    },
    request_id="my-tracking-id",
)
```

---

## 12. Direct API: Relation Extraction

Extract relations and build knowledge graphs from text and entities.

```python
ont = client.experimental.ontology
```

### Basic usage

```python
# First extract entities
entities_result = client.experimental.extraction.extract(
    text="Steve Jobs co-founded Apple in Cupertino. Tim Cook succeeded him as CEO.",
    config={"label_mode": "generated"},
)

# Then build the knowledge graph
result = ont.build_graph(
    text="Steve Jobs co-founded Apple in Cupertino. Tim Cook succeeded him as CEO.",
    entities=[e.model_dump() for e in entities_result.entities],
)
for rel in result.relations:
    print(f"{rel.subject} --[{rel.predicate}]--> {rel.object} (score={rel.score:.2f})")
```

### With entity resolution and link prediction

```python
result = ont.build_graph(
    text="...",
    entities=[...],
    config={
        "resolve_entities": True,              # merge duplicates (2.0x credits)
        "optimize_relations": True,            # refine labels with LLM (1.5x credits)
        "predict_missing_relations": True,     # predict implicit links (2.5x credits)
        "kg_output_format": "custom",          # "custom" | "rdf" | "property_graph"
        "relation_threshold": 0.6,
        "symmetric": True,
        "generate_knowledge_graph": True,
        "max_relations_per_decode": 30,
    },
)
```

### Property graph format (Neo4j-compatible)

```python
result = ont.build_graph(
    text="...",
    entities=[...],
    config={
        "kg_output_format": "property_graph",
        "resolve_entities": True,
    },
)
```

### RDF format

```python
result = ont.build_graph(
    text="...",
    entities=[...],
    config={
        "kg_output_format": "rdf",
        "namespace_uri": "http://example.org/ontology#",
    },
)
```

---

## 13. Direct API: Compression

Intelligently compress text while preserving key information.

```python
comp = client.experimental.compression
```

### Basic compression

```python
result = comp.compress(
    text="Your long document text here...",
    compression_rate=0.5,  # remove ~50% of tokens
)
print(result.compressed_text)
print(f"Original tokens: {result.original_tokens}")
print(f"Compressed tokens: {result.compressed_tokens}")
print(f"Actual ratio: {result.compression_ratio:.2%}")
```

### Preserve specific tokens

```python
result = comp.compress(
    text="The API returns JSON with HTTP status codes for REST endpoints.",
    compression_rate=0.4,
    force_preserve_digit=True,   # keep numbers
    force_tokens=["API", "JSON", "REST", "HTTP"],
)
```

### TOON encoding

```python
result = comp.compress(
    text="...",
    compression_rate=0.5,
    apply_toon=True,  # additional token reduction (+$0.50/1M tokens)
)
```

### All configuration options

```python
result = comp.compress(
    text="...",
    compression_rate=0.5,            # 0.0 (keep all) to 1.0 (remove all)
    force_preserve_digit=True,       # preserve numeric tokens
    force_tokens=["API", "JSON"],    # tokens to never remove
    apply_toon=False,                # TOON encoding (+$0.50/1M tokens)
    chunk_size=4096,                 # tokens per chunk (512-16384)
    fallback_mode=True,              # return original if compression fails
    request_id="my-id",
)
```

### Compress chat messages

```python
result = comp.compress_messages(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about quantum computing..."},
        {"role": "assistant", "content": "Quantum computing uses qubits..."},
        {"role": "user", "content": "How does entanglement work?"},
    ],
    target_compression=0.5,
    max_compression=0.7,           # never compress more than 70%
    force_preserve_digit=True,
    force_tokens=["quantum", "qubit"],
)
print(result.compressed_messages)
```

---

## 14. Direct API: Chunking

> **Standalone API only** -- Chunking is not available as a pipeline step. Use this endpoint directly for text splitting tasks outside of the pipeline.

Split text into semantically meaningful chunks for RAG and retrieval.

```python
ch = client.experimental.chunking
```

### Character strategy (free)

```python
result = ch.chunk(
    text="Your document text...",
    strategy="character",
    chunk_size=512,
    chunk_overlap=50,
)
for i, chunk in enumerate(result.chunks):
    print(f"Chunk {i}: {len(chunk['content'])} chars")
```

### Token strategy (free)

```python
result = ch.chunk(
    text="Your document text...",
    strategy="token",
    chunk_size=256,
    chunk_overlap=20,
)
```

### Semantic strategy (uses embeddings)

```python
result = ch.chunk(
    text="Your document text...",
    strategy="semantic",
    chunk_size=1024,
    semantic_threshold=0.5,     # lower = more aggressive splitting (0.1-0.95)
    semantic_window_size=3,     # sentences per window (1-10)
)
```

### Hybrid strategy (recommended)

```python
result = ch.chunk(
    text="Your document text...",
    strategy="hybrid",          # character splits refined with semantic coherence
    chunk_size=512,
    chunk_overlap=50,
    min_chunk_size=64,
    semantic_threshold=0.5,
    semantic_window_size=3,
)
```

### All configuration options

```python
result = ch.chunk(
    text="...",
    strategy="hybrid",          # "character" | "token" | "semantic" | "hybrid"
    chunk_size=512,             # target size (64-8192)
    chunk_overlap=50,           # overlap between chunks
    min_chunk_size=64,          # discard chunks smaller than this
    semantic_threshold=0.5,     # boundary detection threshold (0.1-0.95)
    semantic_window_size=3,     # sliding window size (1-10)
    request_id="my-id",
)
```

---

## 15. Direct API: Enrichment

> **Standalone API only** -- Enrichment is not available as a pipeline step. Corpus-level processing requires a dedicated architecture; this is coming in a future release.

Chunk text and compute retrieval-optimized features per chunk.

```python
enr = client.experimental.enrichment
```

### Chunk only

```python
result = enr.chunk(
    text="Your document text...",
    strategy="hybrid",
    chunk_size=512,
)
```

### Full enrichment with features

```python
result = enr.enrich(
    text="Your document text...",
    strategy="hybrid",
    chunk_size=512,
    chunk_overlap=50,
    min_chunk_size=64,
    features=["quality", "density", "structural", "semantic"],
)
print(f"Chunks: {len(result.chunks)}")
for chunk_data in result.chunks:
    print(chunk_data)
```

### All 10 feature groups

```python
result = enr.enrich(
    text="...",
    features=[
        "quality",       # text quality metrics
        "density",       # information density
        "structural",    # structural complexity
        "semantic",      # semantic richness
        "compression",   # compressibility
        "zipf",          # Zipf's law conformance
        "coherence",     # topic coherence
        "spectral",      # spectral properties
        "drift",         # topic drift
        "redundancy",    # redundancy metrics
    ],
    encoding_format="float",  # "float" or "base64"
)
```

---

## 16. Direct API: Embeddings (Unified)

The unified `embed` namespace provides access to all embedding types.

```python
emb = client.experimental.embed
```

### Dense embeddings

```python
# Single text
result = emb.dense(text="Hello world", dimension=512)
print(result.embeddings)  # [[0.012, -0.045, ...]]
print(result.shape)        # [1, 512]

# Batch
result = emb.dense(
    text=["Hello world", "Goodbye world"],
    dimension=1024,
)
print(result.shape)  # [2, 1024]
```

Supported dimensions: `256`, `512`, `768`, `1024`

### Late interaction (ColBERT-style)

```python
# Query embedding
query_result = emb.late_interaction(
    text="What is machine learning?",
    is_query=True,
    query_expansion=True,
)
print(query_result.shape)  # [tokens, 128]

# Document embedding
doc_result = emb.late_interaction(
    text="Machine learning is a subset of artificial intelligence...",
    is_query=False,
)
```

### Image embedding (ColPali-style)

```python
# Text query for visual retrieval
result = emb.image(text="Find invoices from 2024", is_query=True)

# Image embedding from file path
result = emb.image(image_path="page_scan.png", is_query=False)

# Image embedding from base64
result = emb.image(image="data:image/png;base64,...", is_query=False)

# Image from open file
with open("chart.png", "rb") as f:
    result = emb.image(image_path=f, is_query=False)
```

---

## 17. Direct API: Legacy Embeddings

These services are also available individually:

### Dense embedding (legacy)

```python
result = client.experimental.embedding.embed(
    text="Hello world",
    dimension=512,
)
print(result.embeddings)
print(result.shape)

# Batch
result = client.experimental.embedding.embed(
    text=["Hello", "World"],
    dimension=768,
)
```

### ColBERT embedding (legacy)

```python
result = client.experimental.colbert.embed(
    text="What is machine learning?",
    is_query=True,
    query_expansion=True,
)
print(result.embeddings)  # token-level embeddings
print(result.shape)        # [num_tokens, 128]
```

### ColPali embedding (legacy)

```python
# Text query
result = client.experimental.colpali.embed(text="Find invoices", is_query=True)

# Image from file
result = client.experimental.colpali.embed(image_path="page.png", is_query=False)

# Image from base64
result = client.experimental.colpali.embed(image="base64_string...", is_query=False)
```

---

## 18. Jobs Service

Manage background jobs across all services.

```python
jobs = client.jobs
```

### List jobs

```python
# All recent jobs
response = jobs.list(limit=50, offset=0)
for job in response.jobs:
    print(f"{job.job_id}: {job.status} ({job.service})")

# Filter by status
response = jobs.list(status="COMPLETED", limit=100)
```

### Iterate all jobs (auto-pagination)

```python
for job in jobs.list_iter(status="COMPLETED", page_size=100):
    print(f"{job.job_id}: {job.status}")
```

### Get specific job

```python
status = jobs.get("job_abc123")
print(status.status)
print(status.service)
print(status.created_at)
```

### Wait for job completion

```python
status = jobs.wait(
    "job_abc123",
    poll_interval=2.0,   # seconds between polls (default 2)
    timeout=300.0,       # max wait in seconds (default 300 = 5 min)
)
```

### Retrieve job result

```python
result = jobs.retrieve("job_abc123")
print(result)  # raw result dict
```

### Cancel a job

```python
response = jobs.cancel("job_abc123")
print(response.status)
```

---

## 19. Credits

Check your credit balance.

```python
balance = client.credits.balance()
print(f"Credits remaining: ${balance.credits_remaining:.2f}")
```

Every API response also includes credit information:

```python
result = client.experimental.extraction.extract(text="Apple Inc. in Cupertino")
print(f"Credits used: {result.credits_used}")
print(f"Credits remaining: {result.credits_remaining}")
print(f"Rate limit remaining: {result.rate_limit_remaining}")
```

---

## 20. Async Usage

Every feature has an async equivalent using `AsyncLatence`.

```python
import asyncio
from latence import AsyncLatence

async def main():
    async with AsyncLatence() as client:
        # Pipeline
        job = await client.pipeline.run(files=["doc.pdf"])
        pkg = await job.wait_for_completion()
        print(pkg.document.markdown)

        # Direct API
        result = await client.experimental.extraction.extract(
            text="Apple Inc. in Cupertino",
        )
        print(result.entities)

        # Embeddings
        result = await client.experimental.embed.dense(
            text=["Hello", "World"],
            dimension=512,
        )
        print(result.shape)

        # Credits
        balance = await client.credits.balance()
        print(balance.credits_remaining)

asyncio.run(main())
```

### Parallel direct API calls

```python
async def parallel_example():
    async with AsyncLatence() as client:
        extraction_task = client.experimental.extraction.extract(
            text="Apple Inc. was founded by Steve Jobs.",
            config={"label_mode": "generated"},
        )
        embedding_task = client.experimental.embed.dense(
            text="Apple Inc. was founded by Steve Jobs.",
            dimension=512,
        )
        chunking_task = client.experimental.chunking.chunk(
            text="A very long document...",
            strategy="hybrid",
        )

        extraction, embedding, chunking = await asyncio.gather(
            extraction_task, embedding_task, chunking_task
        )

asyncio.run(parallel_example())
```

### Async job mode for any service

```python
async def job_mode_example():
    async with AsyncLatence() as client:
        submitted = await client.experimental.document_intelligence.process(
            file_path="large_doc.pdf",
            mode="performance",
            return_job=True,
        )
        result = await client.jobs.wait(submitted.job_id)
        data = await client.jobs.retrieve(submitted.job_id)

asyncio.run(job_mode_example())
```

### Async API differences

The async `AsyncJob` mirrors `Job` with two differences:

| Feature | Sync (`Job`) | Async (`AsyncJob`) |
|---------|-------------|-------------------|
| Pipeline report | `job.report` (property) | `await job.get_report()` (method) |
| Lazy data package | `job.data_package` (fetches on first access) | `job.data_package` (only available after `wait_for_completion`) |

```python
# Sync
report = job.report

# Async -- must await the method, not the property
report = await job.get_report()
```

---

## 21. File Handling

### Local files (Path or string)

```python
# String path
job = client.pipeline.run(files=["report.pdf"])

# pathlib.Path
from pathlib import Path
job = client.pipeline.run(files=[Path("docs/report.pdf")])
```

### Open file objects (BinaryIO)

```python
with open("report.pdf", "rb") as f:
    job = client.pipeline.run(files=[f])
```

### Multiple files

```python
job = client.pipeline.run(files=[
    "contract_v1.pdf",
    "contract_v2.pdf",
    Path("annexes/annex_a.pdf"),
])
pkg = job.wait_for_completion()
```

### File URLs

```python
job = client.pipeline.run(
    file_urls=["https://example.com/document.pdf"],
)
```

### Automatic large file handling

The SDK automatically detects large files and uses presigned B2 uploads:

- **< 10 MB**: Encoded as base64 and sent inline in the JSON request body
- **>= 10 MB** (or batch total >= 10 MB): Uploaded directly to B2 via presigned URLs, bypassing payload size limits

This is fully transparent -- the same `files=[...]` API works for any file size.

```python
# This 500 MB file is automatically uploaded via presigned URL
job = client.pipeline.run(files=["huge_dataset.pdf"])
pkg = job.wait_for_completion()
```

### Direct API with files

```python
result = client.experimental.document_intelligence.process(
    file_path="report.pdf",  # auto-encodes to base64
)

# Or with explicit base64
import base64
with open("report.pdf", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
result = client.experimental.document_intelligence.process(file_base64=b64)
```

---

## 22. Error Handling

The SDK provides a granular exception hierarchy.

```python
from latence import (
    LatenceError,              # base for all SDK errors
    APIError,                  # base for HTTP errors (has status_code, error_code, request_id)
    AuthenticationError,       # 401 -- invalid or missing API key
    InsufficientCreditsError,  # 402 -- account balance is zero
    NotFoundError,             # 404 -- invalid endpoint or job not found
    ValidationError,           # 400 -- invalid request (bad JSON, missing fields)
    RateLimitError,            # 429 -- rate limit exceeded (has retry_after)
    ServerError,               # 5xx -- server-side errors
    TransportError,            # base for network-level errors (no HTTP status code)
    APIConnectionError,        # network issues, DNS failure, connection refused
    APITimeoutError,           # request exceeded configured timeout
    JobError,                  # pipeline/job failed (has job_id, error_code, is_resumable)
    JobTimeoutError,           # wait exceeded timeout (subclass of JobError)
    PipelineValidationError,   # strict mode validation failure (has errors, suggestion)
)
```

### Exception hierarchy

```
LatenceError
├── APIError (status_code, error_code, request_id, body)
│   ├── AuthenticationError      (401)
│   ├── InsufficientCreditsError (402)
│   ├── NotFoundError            (404)
│   ├── ValidationError          (400)
│   ├── RateLimitError           (429, retry_after)
│   └── ServerError              (5xx)
├── TransportError
│   ├── APIConnectionError
│   └── APITimeoutError
├── JobError (job_id, error_code, is_resumable)
│   └── JobTimeoutError
└── PipelineValidationError (errors, suggestion)
```

### Catching errors

```python
try:
    job = client.pipeline.run(files=["doc.pdf"])
    pkg = job.wait_for_completion(timeout=600)
except AuthenticationError:
    print("Invalid API key. Check LATENCE_API_KEY.")
except InsufficientCreditsError:
    print("No credits left. Top up at app.latence.ai")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except ValidationError as e:
    print(f"Bad request: {e.message} (code: {e.error_code})")
except JobTimeoutError as e:
    print(f"Job {e.job_id} didn't finish in time")
except JobError as e:
    if e.is_resumable:
        pkg = job.resume().wait_for_completion()
    else:
        print(f"Job {e.job_id} failed: {e.message} (code: {e.error_code})")
except TransportError:
    print("Network error -- check your connection or try again")
except ServerError as e:
    print(f"Server error {e.status_code}: {e.message}")
```

### Pipeline validation errors

When `strict_mode=True`, validation failures raise `PipelineValidationError` instead of auto-injecting missing services:

```python
from latence import PipelineBuilder, PipelineValidationError

try:
    config = PipelineBuilder().extraction().strict().build()
    job = client.pipeline.submit(config, files=["doc.pdf"])
except PipelineValidationError as e:
    print(e.message)       # "Pipeline validation failed"
    print(e.errors)        # ["Service 'extraction' requires its parent 'document_intelligence'..."]
    print(e.suggestion)    # "Add the following services: ['document_intelligence']"
```

### Inspecting API errors

```python
try:
    result = client.experimental.extraction.extract(text="...")
except APIError as e:
    print(e.status_code)    # 400
    print(e.error_code)     # "INVALID_INPUT"
    print(e.request_id)     # "req_abc123"
    print(e.body)           # raw response dict
    print(e.message)        # human-readable message
```

### Automatic retries

The SDK automatically retries on transient errors:

- **Retried status codes**: 408, 429, 500, 502, 503, 504
- **Max retries**: 2 (configurable)
- **Backoff**: Exponential with jitter (0.5s initial, 60s max, 2x factor, 25% jitter)
- **Retry-After**: Respected when present in the response

---

## 23. Configuration

### Client options

```python
client = Latence(
    api_key="lat_xxx",           # or set LATENCE_API_KEY env var
    base_url="https://api.latence.ai",  # or set LATENCE_BASE_URL env var
    timeout=60.0,                # request timeout in seconds
    max_retries=2,               # max retry attempts for transient errors
)
```

### Polling defaults

| Context | poll_interval | timeout |
|---------|---------------|---------|
| `job.wait_for_completion()` | 5.0s | 1800s (30 min) |
| `client.jobs.wait()` | 2.0s | 300s (5 min) |

Override per call:

```python
pkg = job.wait_for_completion(poll_interval=10.0, timeout=3600.0)
```

### Debug logging

```python
import latence
latence.setup_logging("DEBUG")

# Now all SDK HTTP requests and responses are logged
client = Latence()
result = client.experimental.extraction.extract(text="test")
```

Log levels: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`

Advanced options:

```python
import logging

# Custom format
latence.setup_logging("INFO", fmt="%(asctime)s [%(levelname)s] %(message)s")

# Custom handler (e.g., file logging)
handler = logging.FileHandler("latence_sdk.log")
latence.setup_logging("DEBUG", handler=handler)
```

### Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `LATENCE_API_KEY` | API key for authentication | (none) |
| `LATENCE_BASE_URL` | Override API base URL | `https://api.latence.ai` |
| `LATENCE_LOG_LEVEL` | Auto-configure logging on import | (none) |

Setting `LATENCE_LOG_LEVEL=DEBUG` is equivalent to calling `latence.setup_logging("DEBUG")` -- useful for debugging without code changes.

---

## Complete End-to-End Example

```python
from latence import Latence, PipelineBuilder, JobTimeoutError, JobError

with Latence() as client:
    # Check balance
    balance = client.credits.balance()
    print(f"Credits: ${balance.credits_remaining:.2f}")

    # Build pipeline
    config = (
        PipelineBuilder()
        .doc_intel(mode="performance")
        .redaction(mode="balanced", redact=True, redaction_mode="mask")
        .extraction(
            label_mode="hybrid",
            user_labels=["person", "organization", "location", "date", "monetary_amount"],
            threshold=0.3,
        )
        .relation_extraction(resolve_entities=True, optimize_relations=True)
        .compression(compression_rate=0.4, force_preserve_digit=True)
        .store_intermediate()
        .build()
    )

    # Run pipeline with progress tracking
    try:
        job = client.pipeline.submit(config, files=["contract.pdf"], name="Contract Analysis")
        print(f"Submitted: {job.id}")

        pkg = job.wait_for_completion(
            save_to_disk="./contract_results.zip",
            on_progress=lambda status, elapsed: print(f"  {status} ({elapsed:.0f}s)"),
        )

        # Explore results
        print(f"\nDocument: {pkg.document.metadata.pages_processed} pages")
        print(f"Entities: {pkg.entities.summary.total} ({pkg.entities.summary.by_type})")
        print(f"Relations: {pkg.knowledge_graph.summary.total_relations}")
        print(f"Redacted PII: {pkg.redaction.summary.total_pii}")
        print(f"Compression: {pkg.compression.summary.compression_ratio:.0%} ratio")
        print(f"Cost: ${pkg.quality.total_cost_usd:.4f}")

        # Single consolidated JSON for downstream use
        pkg.merge(save_to="./contract_output.json")

    except JobTimeoutError as e:
        print(f"Pipeline timed out: {e.job_id}")
    except JobError as e:
        if e.is_resumable:
            print(f"Resumable failure -- retrying from checkpoint")
            pkg = job.resume().wait_for_completion()
        else:
            print(f"Pipeline failed: {e.message}")
```
