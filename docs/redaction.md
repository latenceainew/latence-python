# Redaction

Detect and optionally redact personally identifiable information (PII) from text.

```python
from latence import Latence
client = Latence(api_key="lat_xxx")

result = client.experimental.redaction.detect_pii(
    text="My name is John Smith and my SSN is 123-45-6789.",
    config={"redact": True},
)
print(result.redacted_text)  # "My name is [PERSON] and my SSN is [SSN]."
for entity in result.entities:
    print(f"{entity.label}: {entity.text} (score: {entity.score:.2f})")
```

> **Note:** Direct service APIs live under `client.experimental.*`. For production workloads, prefer the [pipeline](pipelines.md) which includes `redaction` as a pipeline step with automatic full refinement.

## `client.experimental.redaction.detect_pii()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Input text |
| `config` | `dict \| RedactionConfig \| None` | `None` | Detection/redaction configuration |
| `custom_labels` | `list[dict] \| list[CustomLabel] \| None` | `None` | Regex-based custom PII detectors |
| `request_id` | `str \| None` | `None` | Optional tracking ID |
| `return_job` | `bool` | `False` | Return job ID for async polling |

### Config Options (`RedactionConfig`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mode` | `"balanced" \| "strict" \| "recall" \| "precision"` | `"balanced"` | Detection sensitivity |
| `threshold` | `float` | `0.3` | Confidence threshold (0.0-1.0) |
| `redact` | `bool` | `False` | Replace PII with labels in output |
| `redaction_mode` | `"mask" \| "replace"` | `"mask"` | `mask`: `[LABEL]`, `replace`: synthetic data |
| `normalize_scores` | `bool` | `True` | Normalize confidence scores |
| `chunk_size` | `int` | `1024` | Chunk size in tokens |
| `enable_refinement` | `bool` | `False` | Refine low-confidence detections with LLM |
| `refinement_threshold` | `float` | `0.5` | Threshold for LLM refinement |
| `enforce_refinement` | `bool` | `False` | Refine all detections (always `True` in pipelines) |

> **Pipeline note:** When used inside a pipeline, full refinement (`enforce_refinement`) is always
> enabled automatically. This ensures production-grade PII filtering. To control refinement
> manually, use the direct API (`client.experimental.redaction.detect_pii()`) instead.

### Detection Modes

| Mode | Behavior |
|------|----------|
| `balanced` | Default -- good precision/recall tradeoff |
| `strict` | High precision, fewer false positives |
| `recall` | High recall, catches more PII (may over-detect) |
| `precision` | Maximum precision, only very confident detections |

### Custom PII Detectors

```python
result = client.experimental.redaction.detect_pii(
    text="Employee ID: EMP-12345, Badge: B-9876",
    custom_labels=[
        {"label_name": "EMPLOYEE_ID", "extractor": r"EMP-\d{5}"},
        {"label_name": "BADGE", "extractor": r"B-\d{4}"},
    ],
)
```

## Response: `DetectPIIResponse`

| Field | Type | Description |
|-------|------|-------------|
| `original_text` | `str` | Input text |
| `entities` | `list[Entity]` | Detected PII entities |
| `entity_count` | `int` | Number of PII entities found |
| `unique_labels` | `list[str]` | Types of PII found |
| `redacted_text` | `str \| None` | Redacted text (only when `redact=True`) |
| `chunks_processed` | `int` | Number of chunks processed |
| `usage` | `Usage \| None` | Credits consumed |

## Pricing

| Component | Cost |
|-----------|------|
| Base | $1.50 / 1M tokens |
| Synthetic PII replacement (`replace` mode) | +$5.00 / 1M tokens |
| LLM refinement (`enable_refinement=True`) | +$10.00 / 1M tokens |
