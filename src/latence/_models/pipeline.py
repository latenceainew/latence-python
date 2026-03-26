"""Pydantic models for the Pipeline service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import BaseResponse, Entity, Usage


# Service type literals
ServiceName = Literal[
    "document_intelligence",
    "extraction",
    "ontology",
    "redaction",
    "compression",
    "embedding",
    "colbert",
    "colpali",
]


class FileInput(BaseModel):
    """File input for pipeline processing."""

    base64: str | None = Field(default=None, description="Base64-encoded file content")
    url: str | None = Field(default=None, description="URL to fetch file from")
    filename: str | None = Field(default=None, description="Original filename with extension")

    @model_validator(mode="after")
    def _at_least_one_source(self) -> "FileInput":
        if self.base64 is None and self.url is None:
            raise ValueError("FileInput requires at least one of 'base64' or 'url'")
        return self


class ServiceConfig(BaseModel):
    """Per-service configuration in a pipeline."""

    service: ServiceName = Field(description="Service to execute")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Service-specific configuration parameters"
    )


class PipelineConfig(BaseModel):
    """Pipeline execution configuration."""

    services: list[ServiceConfig] = Field(
        description="Ordered list of services to execute"
    )
    store_intermediate: bool = Field(
        default=False, description="Store results from each stage"
    )
    strict_mode: bool = Field(
        default=False, description="Error on validation failure instead of auto-injecting services"
    )
    name: str | None = Field(
        default=None, description="Human-readable pipeline name for tracking and display"
    )


class PipelineInput(BaseModel):
    """Pipeline input data."""

    files: list[FileInput] | None = Field(
        default=None, description="Files to process (base64 or URLs)"
    )
    batch_id: str | None = Field(
        default=None, description="B2 batch upload ID (files already uploaded via presigned URLs)"
    )
    text: str | None = Field(default=None, description="Text input")
    entities: list[Entity] | None = Field(
        default=None, description="Pre-extracted entities (for ontology-only pipelines)"
    )


class StageResult(BaseModel):
    """Result from a single pipeline stage."""

    service: str = Field(description="Service that produced this result")
    status: str = Field(description="Stage execution status")
    output: dict[str, Any] | None = Field(default=None, description="Stage output data")
    credits_used: float | None = Field(default=None, description="Credits consumed by this stage")
    processing_time_ms: float | None = Field(
        default=None, description="Processing time in milliseconds"
    )
    error: str | None = Field(default=None, description="Error message if stage failed")


class PipelineExecutionSummary(BaseModel):
    """Summary of pipeline execution."""

    total_stages: int = Field(description="Total number of stages")
    completed_stages: int = Field(description="Number of completed stages")
    failed_stage: str | None = Field(default=None, description="Stage that failed (if any)")
    total_credits_used: float = Field(default=0.0, description="Total credits consumed")
    total_processing_time_ms: float = Field(
        default=0.0, description="Total processing time in milliseconds"
    )


class StageStatus(BaseModel):
    """Status of a single pipeline stage."""

    service: str = Field(description="Service name for this stage")
    status: Literal["pending", "processing", "completed", "failed", "skipped"] = Field(
        default="pending", description="Current status of this stage"
    )
    started_at: str | None = Field(default=None, description="Stage start timestamp")
    completed_at: str | None = Field(default=None, description="Stage completion timestamp")
    processing_time_ms: float | None = Field(
        default=None, description="Processing time in milliseconds"
    )
    error: str | None = Field(default=None, description="Error message if stage failed")


class StageDownload(BaseModel):
    """Per-stage download information for intermediate results."""

    service: str = Field(description="Service name")
    stage_index: int = Field(default=0, description="Stage index")
    status: str = Field(description="Stage completion status")
    download_url: str | None = Field(default=None, description="Presigned URL to JSONL results")
    files_total: int = Field(default=0, description="Total files in this stage")
    files_ok: int = Field(default=0, description="Files processed successfully")
    files_failed: int = Field(default=0, description="Files that failed")
    cost_usd: float = Field(default=0.0, description="Cost in USD for this stage")
    duration_s: float = Field(default=0.0, description="Duration in seconds")
    from_checkpoint: bool = Field(default=False, description="Whether this stage was loaded from checkpoint")


class PipelineReport(BaseModel):
    """Structured pipeline report with dataset facts and per-stage metrics."""

    stages: list[dict[str, Any]] = Field(default_factory=list, description="Per-stage metrics")
    dataset: dict[str, Any] = Field(default_factory=dict, description="Dataset-level facts")
    cost: dict[str, Any] = Field(default_factory=dict, description="Cost breakdown")
    timing: dict[str, Any] = Field(default_factory=dict, description="Timing breakdown")
    resume_count: int = Field(default=0, description="Number of times pipeline was resumed")


class PipelineSubmitResponse(BaseResponse):
    """Response when pipeline job is submitted."""

    job_id: str = Field(description="Pipeline job identifier")
    poll_url: str = Field(description="URL to poll for results")
    services: list[str] = Field(description="Final ordered list of services after validation")
    auto_injected: list[str] | None = Field(
        default=None, description="Services that were auto-added to satisfy dependencies"
    )
    name: str | None = Field(default=None, description="Pipeline name for tracking")
    message: str | None = Field(default=None, description="Status message")
    retention: str | None = Field(default=None, description="Result retention period")


class PipelineStatusResponse(BaseResponse):
    """Response when polling pipeline job status."""

    job_id: str = Field(description="Pipeline job identifier")
    status: Literal[
        "QUEUED", "IN_PROGRESS", "COMPLETED", "CACHED", "PULLED", "FAILED", "CANCELLED", "RESUMABLE"
    ] = Field(description="Current job status")
    current_stage: int | None = Field(default=None, description="Current stage index (0-based)")
    current_service: str | None = Field(
        default=None, description="Service currently being executed"
    )
    stages_completed: int = Field(default=0, description="Number of stages completed")
    total_stages: int = Field(default=0, description="Total number of stages")
    created_at: str | None = Field(default=None, description="Job creation timestamp")
    completed_at: str | None = Field(default=None, description="Job completion timestamp")
    error_code: str | None = Field(default=None, description="Error code if failed")
    error_message: str | None = Field(default=None, description="Error message if failed")
    message: str | None = Field(default=None, description="Status message")
    is_resumable: bool = Field(default=False, description="Whether the pipeline can be resumed")
    resume_count: int = Field(default=0, description="Number of times this pipeline has been resumed")
    pipeline_report: PipelineReport | None = Field(default=None, description="Structured pipeline report")
    failed_stage: str | None = Field(default=None, description="Stage where pipeline failed")


class PipelineResultResponse(BaseResponse):
    """Pipeline execution result."""

    model_config = ConfigDict(extra="allow")

    job_id: str = Field(description="Pipeline job identifier")
    status: str = Field(description="Final job status")
    created_at: str | None = Field(default=None, description="Job creation timestamp")
    final_output: dict[str, Any] | None = Field(
        default=None, description="Output from the final pipeline stage"
    )
    intermediate_results: dict[str, StageResult] | None = Field(
        default=None, description="Results from each stage (if store_intermediate=True)"
    )
    execution_summary: PipelineExecutionSummary = Field(
        description="Summary of pipeline execution"
    )
    output_url: str | None = Field(
        default=None, description="Presigned URL to fetch results (if B2-cached)"
    )
    download_url: str | None = Field(
        default=None, description="Direct download URL for pipeline output archive"
    )
    b2_prefix: str | None = Field(
        default=None, description="B2 storage prefix for per-stage access"
    )
    output_expires_at: str | None = Field(default=None, description="When output_url expires")
    usage: Usage | None = Field(default=None, description="Total credit usage")

    _zip_manifest: dict[str, Any] = {}
    _zip_documents: dict[str, Any] = {}


class PipelineValidationResult(BaseModel):
    """Result of pipeline validation."""

    valid: bool = Field(description="Whether the pipeline configuration is valid")
    services: list[str] = Field(description="Final ordered list of services")
    auto_injected: list[str] = Field(
        default_factory=list, description="Services that were auto-injected"
    )
    errors: list[str] = Field(
        default_factory=list, description="Validation error messages"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Validation warning messages"
    )
