# Unified Embed

Single endpoint for all embedding types: dense vectors, token-level (ColBERT), and visual (ColPali).

```python
from latence import Latence
client = Latence(api_key="lat_xxx")

# Dense embeddings
result = client.experimental.embed.dense(text="Hello world", dimension=512)
print(result.embeddings)  # [[0.012, -0.034, ...]]

# Token-level (ColBERT) embeddings
result = client.experimental.embed.late_interaction(text="search query", is_query=True)
print(result.tokens)  # 5

# Visual (ColPali) embeddings
result = client.experimental.embed.image(image_path="document.png", is_query=False)
print(result.patches)  # 1024
```

> **Note:** Direct service APIs live under `client.experimental.*`. For production workloads, prefer the [pipeline](pipelines.md) which includes `embedding`, `colbert`, and `colpali` as pipeline steps.

## Methods

### `client.experimental.embed.dense()`

Generate dense vector embeddings for text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Input text |
| `dimension` | `int` | `512` | Vector dimension: `256`, `512`, `768`, or `1024` |
| `request_id` | `str \| None` | `None` | Optional tracking ID |
| `return_job` | `bool` | `False` | Return job ID for async polling |

**Pricing:** $0.10 / 1M tokens

### `client.experimental.embed.late_interaction()`

Generate token-level embeddings for fine-grained matching (ColBERT).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Input text |
| `is_query` | `bool` | `True` | `True` for queries, `False` for documents |
| `query_expansion` | `bool` | `True` | Expand query tokens (only when `is_query=True`) |
| `request_id` | `str \| None` | `None` | Optional tracking ID |
| `return_job` | `bool` | `False` | Return job ID for async polling |

**Pricing:** $0.10 / 1M tokens (query), $0.40 / 1M tokens (document)

### `client.experimental.embed.image()`

Generate visual embeddings for document images (ColPali).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| None` | `None` | Query text (required when `is_query=True`) |
| `image` | `str \| None` | `None` | Base64-encoded image |
| `image_path` | `str \| Path \| BinaryIO \| None` | `None` | Path to image file (auto-encoded) |
| `is_query` | `bool` | `True` | `True` for text queries, `False` for document images |
| `request_id` | `str \| None` | `None` | Optional tracking ID |
| `return_job` | `bool` | `False` | Return job ID for async polling |

**Pricing:** $0.10 / 1M tokens (query), $1.00 / 1K images

## Response: `UnifiedEmbedResponse`

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"dense" \| "late_interaction" \| "image"` | Embedding type |
| `embeddings` | `list[list[float]]` | Float arrays (base64 auto-decoded) |
| `shape` | `list[int]` | Array shape |
| `dimension` | `int \| None` | Dimension (dense only) |
| `is_query` | `bool \| None` | Query flag (late_interaction, image) |
| `tokens` | `int \| None` | Token count (late_interaction only) |
| `patches` | `int \| None` | Image patches (image only) |
| `model` | `str` | Model identifier |
| `usage` | `Usage \| None` | Credits consumed |

## Async

```python
from latence import AsyncLatence

async with AsyncLatence(api_key="lat_xxx") as client:
    result = await client.experimental.embed.dense(text="Hello world")
```

## Background Jobs

```python
job = client.experimental.embed.dense(text="Hello world", return_job=True)
print(job.job_id)  # "job_abc123"

result = client.jobs.wait(job.job_id)
```
