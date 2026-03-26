# Entity Extraction

Zero-shot entity extraction from text. No training data required -- provide labels or let the model discover entities automatically.

```python
from latence import Latence
client = Latence(api_key="lat_xxx")

result = client.experimental.extraction.extract(
    text="Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976.",
)
for entity in result.entities:
    print(f"{entity.label}: {entity.text} (score: {entity.score:.2f})")
```

> **Note:** Direct service APIs live under `client.experimental.*`. For production workloads, prefer the [pipeline](pipelines.md).

## `client.experimental.extraction.extract()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Input text |
| `config` | `dict \| ExtractionConfig \| None` | `None` | Extraction configuration (see below) |
| `custom_labels` | `list[dict] \| list[CustomLabel] \| None` | `None` | Regex-based custom extractors |
| `request_id` | `str \| None` | `None` | Optional tracking ID |
| `return_job` | `bool` | `False` | Return job ID for async polling |

### Config Options (`ExtractionConfig`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `label_mode` | `"user" \| "hybrid" \| "generated"` | `"generated"` | Label generation mode |
| `user_labels` | `list[str] \| None` | `None` | Labels to extract (required for `user` mode) |
| `threshold` | `float` | `0.3` | Confidence threshold (0.0-1.0) |
| `flat_ner` | `bool` | `True` | Disable nested entities |
| `multi_label` | `bool` | `False` | Allow multiple labels per span |
| `chunk_size` | `int` | `1024` | Chunk size in tokens |
| `enable_refinement` | `bool` | `False` | Refine low-confidence entities with LLM |
| `refinement_threshold` | `float` | `0.5` | Threshold below which entities get refined |
| `enforce_refinement` | `bool` | `False` | Refine all entities (not just low-confidence) |

### Label Modes

| Mode | Description |
|------|-------------|
| `generated` | Model auto-discovers entity types (default) |
| `user` | Extract only specified labels (`user_labels` required) |
| `hybrid` | Your labels + auto-discovery combined |

### Custom Labels (Regex)

```python
result = client.experimental.extraction.extract(
    text="Contact us at support@example.com or call 555-0123.",
    custom_labels=[
        {"label_name": "EMAIL", "extractor": r"[\w.+-]+@[\w-]+\.[\w.]+"},
        {"label_name": "PHONE", "extractor": r"\d{3}-\d{4}"},
    ],
)
```

## Response: `ExtractResponse`

| Field | Type | Description |
|-------|------|-------------|
| `original_text` | `str` | Input text |
| `entities` | `list[Entity]` | Extracted entities |
| `entity_count` | `int` | Number of entities found |
| `unique_labels` | `list[str]` | Distinct label types found |
| `chunks_processed` | `int` | Number of chunks processed |
| `labels_generated` | `bool \| list[str]` | Auto-generated labels (if applicable) |
| `usage` | `Usage \| None` | Credits consumed |

### Entity Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Entity text |
| `label` | `str` | Entity type (e.g., "PERSON", "ORG") |
| `start` | `int \| None` | Start character position |
| `end` | `int \| None` | End character position |
| `score` | `float` | Confidence score (0-1) |
| `source` | `str \| None` | Source: `"model"` or `"regex"` |

## Pricing

| Component | Cost |
|-----------|------|
| Base | $1.00 / 1M tokens |
| Auto labels (`generated` / `hybrid` mode) | +$5.00 / 1M tokens |
| Label refinement (`enable_refinement=True`) | +$10.00 / 1M tokens |
