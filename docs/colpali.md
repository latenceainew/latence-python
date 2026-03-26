# ColPali (Legacy)

> **Prefer `client.experimental.embed.image()` instead.** This is the legacy direct endpoint for ColPali visual document embeddings.

Vision-language embeddings for document image retrieval. Understands layout, tables, charts, and figures without OCR.

```python
from latence import Latence
client = Latence(api_key="lat_xxx")

# Query embedding (text)
query = client.experimental.colpali.embed(text="revenue growth chart", is_query=True)

# Document image embedding
doc = client.experimental.colpali.embed(image_path="report.png", is_query=False)

# Or with base64
doc = client.experimental.colpali.embed(image="base64_encoded_string...", is_query=False)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| None` | `None` | Query text (required when `is_query=True`) |
| `image` | `str \| None` | `None` | Base64-encoded image |
| `image_path` | `str \| Path \| BinaryIO \| None` | `None` | Path to image file (auto-encoded) |
| `is_query` | `bool` | `True` | `True` for text queries, `False` for document images |
| `request_id` | `str \| None` | `None` | Optional tracking ID |
| `return_job` | `bool` | `False` | Return job ID for async polling |

**Rules:**
- When `is_query=True`: `text` is required, `image` is ignored
- When `is_query=False`: `image` or `image_path` is required

## Response: `ColPaliEmbedResponse`

| Field | Type | Description |
|-------|------|-------------|
| `embeddings` | `list[list[float]]` | Patch-level embeddings (base64 auto-decoded) |
| `shape` | `list[int]` | `[num_patches, dimension]` |
| `is_query` | `bool` | Whether this was a query embedding |
| `patches` | `int \| None` | Number of image patches |
| `model` | `str` | Model identifier |
| `usage` | `Usage \| None` | Credits consumed |

**Pricing:** $0.10 / 1M tokens (query), $1.00 / 1K images
