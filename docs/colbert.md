# ColBERT (Legacy)

> **Prefer `client.experimental.embed.late_interaction()` instead.** This is the legacy direct endpoint for ColBERT token-level embeddings.

Token-level embeddings for fine-grained neural retrieval. Each token gets its own vector, enabling precise matching between queries and documents.

```python
from latence import Latence
client = Latence(api_key="lat_xxx")

# Query embedding
query = client.experimental.colbert.embed(text="What is machine learning?", is_query=True)
print(query.shape)     # [7, 128] -> 7 tokens, 128-dim each
print(query.tokens)    # 7

# Document embedding
doc = client.experimental.colbert.embed(text="Machine learning is a subset of AI...", is_query=False)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Input text |
| `is_query` | `bool` | `True` | `True` for queries, `False` for documents |
| `query_expansion` | `bool` | `True` | Expand query tokens for better recall (only when `is_query=True`) |
| `request_id` | `str \| None` | `None` | Optional tracking ID |
| `return_job` | `bool` | `False` | Return job ID for async polling |

## Response: `ColBERTEmbedResponse`

| Field | Type | Description |
|-------|------|-------------|
| `embeddings` | `list[list[float]]` | Per-token embeddings (base64 auto-decoded) |
| `shape` | `list[int]` | `[num_tokens, dimension]` |
| `is_query` | `bool` | Whether this was a query embedding |
| `tokens` | `int \| None` | Number of tokens |
| `model` | `str` | Model identifier |
| `usage` | `Usage \| None` | Credits consumed |

**Pricing:** $0.10 / 1M tokens (query), $0.40 / 1M tokens (document)
