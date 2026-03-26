"""Pydantic models for the Document Intelligence service (V2)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .common import BaseResponse, Usage


# =============================================================================
# V2 REQUEST OPTIONS
# =============================================================================


class PipelineOptions(BaseModel):
    """
    Pipeline configuration options for document processing.

    These control which modules are loaded and their default behavior.
    All fields are optional - only specify values you want to override from defaults.
    """

    use_layout_detection: bool | None = Field(
        default=None, description="Enable layout detection (PP-DocLayoutV2). Default: true"
    )
    use_chart_recognition: bool | None = Field(
        default=None, description="Enable chart parsing. Default: true"
    )
    use_seal_recognition: bool | None = Field(
        default=None, description="Enable seal recognition. Default: false"
    )
    use_doc_orientation_classify: bool | None = Field(
        default=None, description="Enable document orientation classification (auto-rotate). Default: true"
    )
    use_doc_unwarping: bool | None = Field(
        default=None, description="Enable text image rectification (dewarp). Default: false"
    )
    use_ocr_for_image_block: bool | None = Field(
        default=None, description="Perform OCR on text within image blocks. Default: true"
    )
    format_block_content: bool | None = Field(
        default=None, description="Format block_content as Markdown. Default: true"
    )
    merge_layout_blocks: bool | None = Field(
        default=None, description="Merge cross-column or staggered layout blocks. Default: true"
    )
    markdown_ignore_labels: list[str] | None = Field(
        default=None,
        description="Layout labels to ignore in Markdown output (e.g., ['number', 'footnote', 'header', 'footer'])",
    )
    use_queues: bool | None = Field(
        default=None, description="Enable async internal queues for efficiency. Default: true"
    )
    enable_hpi: bool | None = Field(
        default=None, description="Deprecated -- ignored by the backend (vLLM serves the model directly). Kept for backward compatibility."
    )
    precision: str | None = Field(
        default=None, description="Deprecated -- ignored by the backend (vLLM manages precision automatically). Kept for backward compatibility."
    )


class PredictOptions(BaseModel):
    """
    Per-request prediction options for document processing.

    These can override pipeline defaults on a per-request basis.
    All fields are optional - only specify values you want to override.
    """

    # Layout detection options
    layout_threshold: float | dict[int, float] | None = Field(
        default=None,
        description="Score threshold (0-1) for layout detection, or per-class dict {cls_id: threshold}. Default: 0.3",
    )
    layout_nms: bool | None = Field(
        default=None, description="Use NMS post-processing for layout detection. Default: true"
    )
    layout_shape_mode: str | None = Field(
        default=None,
        description="Shape representation mode: 'auto', 'square', or 'preserve'. Default: 'auto'",
    )

    # VLM sampling options
    temperature: float | None = Field(
        default=None, description="Temperature for VLM sampling. Default: 0.1"
    )
    top_p: float | None = Field(default=None, description="Top-p for VLM sampling")
    repetition_penalty: float | None = Field(
        default=None, description="Repetition penalty for VLM sampling"
    )
    max_new_tokens: int | None = Field(
        default=None, description="Maximum tokens to generate"
    )

    # Image preprocessing
    min_pixels: int | None = Field(
        default=None, description="Minimum pixels for VLM image preprocessing"
    )
    max_pixels: int | None = Field(
        default=None, description="Maximum pixels for VLM image preprocessing"
    )
    vl_rec_max_concurrency: int | None = Field(
        default=None, description="Deprecated -- ignored by the backend (vLLM handles concurrency internally). Kept for backward compatibility."
    )


class OutputOptions(BaseModel):
    """
    Output formatting options for document processing results.

    Control how the extracted content is formatted and what additional
    data is included in the response.
    """

    pretty: bool = Field(
        default=True, description="Beautify markdown output (center charts, etc.)"
    )
    show_formula_number: bool = Field(
        default=False, description="Include formula numbers in markdown output"
    )
    include_images: bool = Field(
        default=False,
        description="Include base64-encoded visualization images in response",
    )
    indent: int | None = Field(
        default=None,
        description="Indentation level for JSON-style output formatting",
    )
    ensure_ascii: bool | None = Field(
        default=None,
        description="Escape non-ASCII characters in JSON-style output",
    )


class RefineOptions(BaseModel):
    """Refinement behavior for multi-page post-processing."""

    merge_tables: bool | None = Field(
        default=None, description="Merge tables that continue across pages"
    )
    relevel_titles: bool | None = Field(
        default=None, description="Normalize heading hierarchy across pages"
    )
    concatenate_pages: bool | None = Field(
        default=None, description="Concatenate page outputs into a continuous flow"
    )


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class DocumentMetadata(BaseResponse):
    """Metadata about processed document."""

    filename: str | None = Field(default=None, description="Original filename")
    pages: int | None = Field(default=None, description="Number of pages")
    pages_processed: int | None = Field(default=None, description="Number of pages processed")
    file_type: str | None = Field(default=None, description="Detected file type")
    output_format: str | None = Field(default=None, description="Output format used")
    processing_mode: str | None = Field(default=None, description="Processing mode used")
    model: str | None = Field(default=None, description="Model name used for processing")
    target_longest: int | None = Field(
        default=None, description="Target longest dimension for image preprocessing"
    )


class RefinementStats(BaseModel):
    """Statistics from performance mode refinement."""

    tables_merged: bool | None = Field(default=None, description="Whether tables were merged")
    titles_releveled: bool | None = Field(
        default=None, description="Whether title hierarchy was reconstructed"
    )
    pages_concatenated: bool | None = Field(
        default=None, description="Whether pages were concatenated"
    )
    input_pages: int | None = Field(
        default=None, description="Number of input pages for refinement"
    )


class ProcessDocumentResponse(BaseResponse):
    """Response from document processing (V2)."""

    # Content
    content: str = Field(description="Extracted document content")
    content_type: str | None = Field(
        default=None, description="Output format type (markdown, json, html, xlsx)"
    )

    # Metadata
    metadata: DocumentMetadata | dict[str, Any] | None = Field(
        default=None, description="Document metadata"
    )
    pages_processed: int | None = Field(default=None, description="Number of pages processed")

    # Structured page results (V2)
    pages: list[dict[str, Any]] | None = Field(
        default=None,
        description="Per-page structured results with parsing_res_list containing layout blocks",
    )

    # Optional visualization images (V2)
    images: dict[str, str] | None = Field(
        default=None,
        description="Base64-encoded visualization images (if output_options.include_images=true)",
    )

    # Refinement stats (V2 - present if mode=performance)
    refinement_stats: RefinementStats | dict[str, Any] | None = Field(
        default=None,
        description="Refinement statistics (present when mode='performance')",
    )

    # Timing and usage
    processing_time_ms: float | None = Field(default=None, description="Processing time in ms")
    usage: Usage | None = Field(default=None, description="Credit usage")
