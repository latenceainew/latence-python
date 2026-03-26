# Embedding (Legacy)

> **Prefer `client.experimental.embed.dense()` instead.** This is the legacy direct endpoint for dense vector embeddings.

```python
from latence import Latence
client = Latence(api_key="lat_xxx")

result = client.experimental.embedding.embed(text="Hello world", dimension=512)
print(result.embeddings)  # [[0.012, -0.034, ...]]
print(result.dimension)   # 512
print(result.shape)       # [1, 512]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Input text (1-100,000 chars) |
| `dimension` | `int` | `512` | Vector dimension: `256`, `512`, `768`, or `1024` |
| `request_id` | `str \| None` | `None` | Optional tracking ID |
| `return_job` | `bool` | `False` | Return job ID for async polling |

## Response: `EmbedResponse`

| Field | Type | Description |
|-------|------|-------------|
| `embeddings` | `list[list[float]]` | Float arrays (base64 auto-decoded) |
| `dimension` | `int` | Embedding dimension |
| `shape` | `list[int]` | Array shape |
| `model` | `str` | Model identifier |
| `usage` | `Usage \| None` | Credits consumed |

**Pricing:** $0.10 / 1M tokens
