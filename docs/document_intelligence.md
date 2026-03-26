# Document Intelligence

Extract structured content from PDFs, images, and office documents. Supports OCR, layout detection, chart recognition, and markdown/JSON/HTML/XLSX output.

```python
from latence import Latence
client = Latence(api_key="lat_xxx")

result = client.experimental.document_intelligence.process(file_path="report.pdf")
print(result.content)          # Markdown output
print(result.pages_processed)  # Number of pages processed
```

> **Note:** Direct service APIs live under `client.experimental.*`. For production workloads, prefer the [pipeline](pipelines.md) where Document Intelligence runs as the `ocr` / `doc_intel` step.

## `client.experimental.document_intelligence.process()`

### Input (one required)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str \| Path \| BinaryIO \| None` | `None` | Local file (auto-encoded to base64) |
| `file_base64` | `str \| None` | `None` | Base64-encoded file content |
| `file_url` | `str \| None` | `None` | URL to fetch file from |
| `filename` | `str \| None` | `None` | Original filename with extension (helps format detection) |

### Processing Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `"default" \| "performance"` | `"default"` | `performance` adds multi-page refinement |
| `output_format` | `"markdown" \| "json" \| "html" \| "xlsx"` | `"markdown"` | Output format |
| `max_pages` | `int \| None` | `None` | Limit pages to process |
| `target_longest` | `int \| None` | `None` | Target longest image dimension (px) |
| `request_id` | `str \| None` | `None` | Optional tracking ID |
| `return_job` | `bool` | `False` | Return job ID for async polling |

### Pipeline Options (`pipeline_options`)

Fine-grained control over the processing pipeline.

```python
result = client.experimental.document_intelligence.process(
    file_path="report.pdf",
    pipeline_options={
        "use_layout_detection": True,
        "use_chart_recognition": True,
        "use_doc_orientation_classify": True,
        "precision": "bf16",
    },
)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_layout_detection` | `bool` | `True` | Enable layout analysis (PP-DocLayoutV2) |
| `use_chart_recognition` | `bool` | `True` | Parse charts into data |
| `use_seal_recognition` | `bool` | `False` | Detect seals/stamps |
| `use_doc_orientation_classify` | `bool` | `True` | Auto-rotate pages |
| `use_doc_unwarping` | `bool` | `False` | Dewarp curved text |
| `use_ocr_for_image_block` | `bool` | `True` | OCR text within images |
| `format_block_content` | `bool` | `True` | Format blocks as Markdown |
| `merge_layout_blocks` | `bool` | `True` | Merge cross-column blocks |
| `markdown_ignore_labels` | `list[str] \| None` | `None` | Layout labels to skip |
| `use_queues` | `bool` | `True` | Async internal queues |
| `enable_hpi` | `bool` | `True` | High-performance inference |
| `precision` | `"bf16" \| "fp16" \| "fp32"` | `"bf16"` | Computation precision |

### Predict Options (`predict_options`)

Control model inference behavior.

```python
result = client.experimental.document_intelligence.process(
    file_path="scan.jpg",
    predict_options={
        "layout_threshold": 0.5,
        "temperature": 0.1,
        "max_new_tokens": 4096,
    },
)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `layout_threshold` | `float \| dict` | `0.3` | Layout detection threshold (or per-class dict `{cls_id: threshold}`) |
| `layout_nms` | `bool` | `True` | NMS post-processing for layout |
| `layout_shape_mode` | `"auto" \| "square" \| "preserve"` | `"auto"` | Shape representation mode |
| `temperature` | `float` | `0.1` | VLM sampling temperature (0.0-2.0) |
| `top_p` | `float \| None` | `None` | Top-p nucleus sampling |
| `repetition_penalty` | `float \| None` | `None` | Repetition penalty |
| `max_new_tokens` | `int \| None` | `None` | Max generated tokens |
| `min_pixels` | `int \| None` | `None` | Min pixels for VLM preprocessing |
| `max_pixels` | `int \| None` | `None` | Max pixels for VLM preprocessing |
| `vl_rec_max_concurrency` | `int \| None` | `None` | Max concurrency for VL recognition |

### Output Options (`output_options`)

```python
result = client.experimental.document_intelligence.process(
    file_path="report.pdf",
    output_options={
        "pretty": True,
        "include_images": True,
    },
)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `pretty` | `bool` | `True` | Beautify markdown (center charts, etc.) |
| `show_formula_number` | `bool` | `False` | Include formula numbers |
| `include_images` | `bool` | `False` | Return base64-encoded visualization images |

## Response: `ProcessDocumentResponse`

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Extracted content (markdown, JSON, HTML, or XLSX) |
| `content_type` | `str \| None` | Output format type |
| `metadata` | `DocumentMetadata \| None` | File info (filename, pages, file_type, model) |
| `pages_processed` | `int \| None` | Number of pages processed |
| `pages` | `list[dict] \| None` | Per-page structured results with layout blocks |
| `images` | `dict[str, str] \| None` | Base64 visualization images (when `include_images=True`) |
| `refinement_stats` | `RefinementStats \| None` | Refinement info (when `mode="performance"`) |
| `processing_time_ms` | `float \| None` | Processing time in ms |
| `usage` | `Usage \| None` | Credits consumed |

## Pricing

| Mode | Cost |
|------|------|
| `default` | $5.00 / 1K pages |
| `performance` | $7.50 / 1K pages |
| Layout detection addon | +$1.25 / 1K pages |
