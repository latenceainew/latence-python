# Compression

Compress text and chat messages while preserving meaning. Reduces token costs for LLM pipelines by up to 80%.

```python
from latence import Latence
client = Latence(api_key="lat_xxx")

result = client.experimental.compression.compress(
    text="The Federal Reserve maintained its benchmark interest rate...",
    compression_rate=0.5,
)
print(result.compressed_text)
print(f"Saved {result.tokens_saved} tokens ({result.compression_ratio:.0%} ratio)")
```

> **Note:** Direct service APIs live under `client.experimental.*`. For production workloads, prefer the [pipeline](pipelines.md) which includes `compression` as a pipeline step.

## Text Compression

### `client.experimental.compression.compress()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Input text to compress |
| `compression_rate` | `float` | `0.5` | Target compression (0.0-1.0). 0.5 = remove ~50% of tokens |
| `force_preserve_digit` | `bool` | `True` | Preserve numbers |
| `force_tokens` | `list[str] \| None` | `None` | Tokens to always preserve |
| `apply_toon` | `bool` | `False` | Apply TOON encoding (+$0.50/1M tokens) |
| `chunk_size` | `int` | `4096` | Processing chunk size |
| `fallback_mode` | `bool` | `True` | Fallback if compression fails |
| `request_id` | `str \| None` | `None` | Optional tracking ID |
| `return_job` | `bool` | `False` | Return job ID for async polling |

### Response: `CompressResponse`

| Field | Type | Description |
|-------|------|-------------|
| `compressed_text` | `str` | Compressed text |
| `original_tokens` | `int` | Original token count |
| `compressed_tokens` | `int` | Compressed token count |
| `compression_ratio` | `float` | Achieved compression ratio |
| `tokens_saved` | `int` | Tokens saved |
| `toon_applied` | `bool` | Whether TOON encoding was applied |
| `usage` | `Usage \| None` | Credits consumed |

## Chat Message Compression

### `client.experimental.compression.compress_messages()`

Compresses a conversation with gradient annealing -- recent messages are preserved more than older ones.

```python
result = client.experimental.compression.compress_messages(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in detail..."},
        {"role": "assistant", "content": "Quantum computing uses qubits..."},
        {"role": "user", "content": "How does error correction work?"},
    ],
    target_compression=0.5,
)
for msg in result.compressed_messages:
    print(f"{msg.role}: {msg.content[:60]}...")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `list[dict] \| list[Message]` | required | Chat messages (`role` + `content`) |
| `target_compression` | `float` | `0.5` | Target compression (0.0-1.0) |
| `force_tokens` | `list[str] \| None` | `None` | Tokens to always preserve |
| `force_preserve_digit` | `bool` | `True` | Preserve numbers |
| `request_id` | `str \| None` | `None` | Optional tracking ID |
| `return_job` | `bool` | `False` | Return job ID for async polling |

### Response: `CompressMessagesResponse`

| Field | Type | Description |
|-------|------|-------------|
| `compressed_messages` | `list[Message]` | Compressed messages |
| `statistics` | `dict \| None` | Per-message compression stats |
| `original_total_length` | `int` | Original total character length |
| `compressed_total_length` | `int` | Compressed total character length |
| `average_compression` | `float` | Average compression ratio |
| `compression_percentage` | `float` | Overall compression percentage |
| `usage` | `Usage \| None` | Credits consumed |

## Pricing

| Component | Cost |
|-----------|------|
| Base | $0.25 / 1M tokens |
| Chat annealing | +$0.25 / 1M tokens |
| TOON encoding | +$0.50 / 1M tokens |
