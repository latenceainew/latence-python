# Dataset Intelligence

Corpus-level knowledge graph construction, ontology induction, and incremental dataset ingestion. Transform pipeline outputs into structured knowledge — entities, relations, graph embeddings, and ontological concepts — with built-in deduplication, link prediction, and delta-aware append mode.

```python
from latence import Latence
client = Latence(api_key="lat_xxx")

# Full pipeline — all tiers (enrichment + knowledge graph + ontology)
job = client.experimental.dataset_intelligence_service.run(
    input_data=pipeline_output,
    return_job=True,
)
print(f"Job submitted: {job.job_id}")
# Poll status at GET /api/v1/pipeline/{job.job_id}
```

> **Note:** Direct service APIs live under `client.experimental.*`. Dataset Intelligence requires pipeline output as input — run the [pipeline](pipelines.md) first, then feed its output here for corpus-level analysis.

## Processing Tiers

| Tier | Method | What it does | GPU |
|------|--------|-------------|-----|
| **Tier 1** | `enrich()` | Semantic feature vectors via EmbeddingGemma (chunking, embeddings) | No |
| **Tier 2** | `build_graph()` | Entity resolution, knowledge graph, RotatE link prediction | Yes |
| **Tier 3** | `build_ontology()` | Concept clustering, hierarchy induction, SHACL shapes | Yes |
| **Full** | `run()` | All 3 tiers sequentially | Yes |

---

## `client.experimental.dataset_intelligence_service.run()`

Full pipeline — runs all three tiers in sequence.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_data` | `dict` | required | Pipeline output payload (see [Input Format](#input-data-format) below) |
| `dataset_id` | `str \| None` | `None` | Existing dataset ID for append mode. If `None`, creates a new dataset. |
| `config_overrides` | `dict \| None` | `None` | Override default tier configurations |
| `request_id` | `str \| None` | `None` | Optional tracking ID |
| `return_job` | `bool` | `False` | Return `JobSubmittedResponse` for async polling |

```python
# Synchronous (blocks until done — may timeout for large datasets)
result = client.experimental.dataset_intelligence_service.run(
    input_data=pipeline_output,
)
print(f"Dataset: {result.dataset_id}")
print(f"Entities: {result.data['total_entities']}")
print(f"Cost: ${result.usage.credits:.4f}")

# Asynchronous (recommended for production)
job = client.experimental.dataset_intelligence_service.run(
    input_data=pipeline_output,
    return_job=True,
)
print(f"Job submitted: {job.job_id}")
# Poll at /api/v1/pipeline/{job.job_id} for status
```

## `client.experimental.dataset_intelligence_service.enrich()`

Tier 1 only — semantic enrichment with feature vectors. CPU-only, fast.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_data` | `dict` | required | Pipeline output payload |
| `dataset_id` | `str \| None` | `None` | Existing dataset ID for append mode |
| `config_overrides` | `dict \| None` | `None` | Override enrichment configuration |
| `request_id` | `str \| None` | `None` | Optional tracking ID |

```python
result = client.experimental.dataset_intelligence_service.enrich(
    input_data=pipeline_output,
)
print(f"Tier: {result.tier}")  # "tier1"
```

> `enrich()` is synchronous-only (no `return_job`). For large payloads, consider using `run(return_job=True)` with `tier="tier1"` via `config_overrides` if you need async submission.

## `client.experimental.dataset_intelligence_service.build_graph()`

Tier 2 — knowledge graph construction with entity resolution and RotatE link prediction.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_data` | `dict` | required | Pipeline output payload |
| `dataset_id` | `str \| None` | `None` | Existing dataset ID for append mode |
| `config_overrides` | `dict \| None` | `None` | Override graph configuration |
| `request_id` | `str \| None` | `None` | Optional tracking ID |
| `return_job` | `bool` | `False` | Return `JobSubmittedResponse` for async polling |

```python
job = client.experimental.dataset_intelligence_service.build_graph(
    input_data=pipeline_output,
    return_job=True,
)
```

## `client.experimental.dataset_intelligence_service.build_ontology()`

Tier 3 — ontology induction with concept clustering and SHACL constraint shapes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_data` | `dict` | required | Pipeline output payload |
| `dataset_id` | `str \| None` | `None` | Existing dataset ID for append mode |
| `config_overrides` | `dict \| None` | `None` | Override ontology configuration |
| `request_id` | `str \| None` | `None` | Optional tracking ID |

```python
result = client.experimental.dataset_intelligence_service.build_ontology(
    input_data=pipeline_output,
)
for concept in result.data.get("concepts", []):
    print(f"  {concept['label']} (level {concept.get('level', 0)})")
```

> `build_ontology()` is synchronous-only (no `return_job`). For large payloads, consider using `run(return_job=True)` with `tier="tier3"` via `config_overrides` if you need async submission.

---

## Input Data Format

Dataset Intelligence consumes the output of a Latence pipeline. The payload is a dict with stage keys mapping to lists of per-file records:

```python
{
    "result": { ... },  # Pipeline manifest (job_id, stages, timings)
    "stage_01_document_intelligence": [
        {
            "file_id": "f_abc",
            "filename": "report.pdf",
            "success": True,
            "output": {"content": "...", "pages": [...]},
        },
        ...
    ],
    "stage_02_extraction": [...],
    "stage_03_ontology": [...],
}
```

The minimum required stage is `stage_01_document_intelligence` with page content. Additional stages (extraction, ontology) improve graph quality when available.

### Large Payloads

Payloads exceeding 8 MB are automatically uploaded to B2 via a presigned URL. The SDK handles this transparently — no code changes needed regardless of payload size.

---

## Delta Ingestion (Append Mode)

Pass an existing `dataset_id` to incrementally update a dataset instead of rebuilding from scratch. The service detects new, updated, and unchanged files, merges entities, patches the knowledge graph, and retrains the RotatE model on the delta.

```python
# First run — creates the dataset
result = client.experimental.dataset_intelligence_service.run(
    input_data=initial_pipeline_output,
    return_job=True,
)
# ... wait for completion ...
dataset_id = result.dataset_id  # e.g. "ds_abc123"

# Later — append new documents
result = client.experimental.dataset_intelligence_service.run(
    input_data=new_pipeline_output,
    dataset_id=dataset_id,
    return_job=True,
)
# result.mode == "append"
# result.delta_summary contains change details
```

### Delta Summary Fields

| Field | Type | Description |
|-------|------|-------------|
| `files_added` | `int` | New files ingested |
| `files_updated` | `int` | Files re-processed with changes |
| `files_unchanged` | `int` | Files skipped (no changes) |
| `new_entities` | `int` | Newly discovered entities |
| `merged_entities` | `int` | Entities merged with existing ones |
| `new_edges` | `int` | New knowledge graph edges |
| `removed_edges` | `int` | Edges removed due to updated files |
| `ontology_types_added` | `int` | New concept types in ontology |
| `rotate_retrain` | `str` | RotatE retrain strategy (`"full"`, `"delta"`, `"skip"`) |
| `delta_ratio` | `float` | Fraction of dataset changed (0.0–1.0) |

---

## Async Usage

```python
from latence import AsyncLatence
import asyncio

async def main():
    async with AsyncLatence(api_key="lat_xxx") as client:
        job = await client.experimental.dataset_intelligence_service.run(
            input_data=pipeline_output,
            return_job=True,
        )
        print(f"Submitted: {job.job_id}")

asyncio.run(main())
```

---

## Response: `DatasetIntelligenceResponse`

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether processing succeeded |
| `tier` | `str` | Tier that was executed (`tier1`, `tier2`, `tier3`, `full`) |
| `dataset_id` | `str` | Dataset identifier (new or existing) |
| `mode` | `str` | `"create"` or `"append"` |
| `data` | `dict` | Result payload (entities, relations, graph, concepts) |
| `usage` | `DatasetIntelligenceUsage` | Credit usage breakdown |
| `stage_timings` | `list[DatasetIntelligenceStageTiming]` | Per-stage timing info |
| `delta_summary` | `DatasetIntelligenceDeltaSummary \| None` | Change summary (append mode only) |
| `processing_time_ms` | `float` | Total processing time in milliseconds |
| `version` | `str` | Service version |
| `error` | `str \| None` | Error message (if `success=False`) |
| `error_code` | `str \| None` | Machine-readable error code |

### `DatasetIntelligenceUsage`

| Field | Type | Description |
|-------|------|-------------|
| `credits` | `float` | Total credits consumed |
| `calculation` | `str` | Human-readable cost formula |
| `details` | `dict` | Breakdown (`num_pages`, `num_files`, `num_entities`, `num_concepts`) |

### `DatasetIntelligenceStageTiming`

| Field | Type | Description |
|-------|------|-------------|
| `stage` | `str` | Stage name (e.g. `"enrichment"`, `"graph"`, `"ontology"`) |
| `elapsed_ms` | `float` | Duration in milliseconds |
| `status` | `str` | Stage status (`"completed"`, `"skipped"`, `"failed"`) |

### Data Payload Keys (Full Tier)

| Key | Type | Description |
|-----|------|-------------|
| `entities` | `list[dict]` | Canonical entities with `entity_id`, `label`, `entity_type`, `confidence` |
| `relations` | `list[dict]` | Canonical relations with `source_entity_id`, `target_entity_id`, `relation_type` |
| `graph_nodes` | `list[dict]` | Knowledge graph nodes with embeddings and evidence counts |
| `graph_edges` | `list[dict]` | Knowledge graph edges with confidence and `is_predicted` flag |
| `concepts` | `list[dict]` | Ontology concepts with `concept_id`, `label`, `level`, `parent_concept_id` |
| `relation_types` | `list[dict]` | Ontology relation types with domain/range concept IDs |
| `predicted_edges` | `list[dict]` | Link-predicted edges (RotatE) |
| `total_entities` | `int` | Total entity count |
| `total_relations` | `int` | Total relation count |
| `total_graph_nodes` | `int` | Total graph node count |

---

## Pricing

Billed per 1,000 pages in the input data. Page count is determined by the `document_intelligence` stage output.

| Tier | Cost per 1K pages | What's included |
|------|-------------------|-----------------|
| Tier 1 (`enrich`) | $1.00 | Semantic enrichment, feature vectors |
| Tier 2 (`build_graph`) | $10.00 | Entity resolution, knowledge graph, RotatE link prediction |
| Tier 3 (`build_ontology`) | $50.00 | Concept clustering, hierarchy induction, SHACL shapes |
| Full (`run`) | $51.85 | All tiers (15% bundle discount) |

**Append mode discount:** 30% off all tiers for incremental ingestion.
