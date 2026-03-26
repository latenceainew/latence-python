"""Pydantic models for the Jobs service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .common import BaseResponse


JobStatus = Literal[
    "QUEUED",
    "IN_PROGRESS",
    "COMPLETED",
    "CACHED",
    "PULLED",
    "FAILED",
    "CANCELLED",
    "RESUMABLE",
]


class JobSubmittedResponse(BaseResponse):
    """Response when a background job is submitted."""

    job_id: str = Field(description="Job identifier")
    status: JobStatus = Field(default="QUEUED", description="Job status")
    poll_url: str | None = Field(default=None, description="URL to poll for results")
    message: str | None = Field(default=None, description="Status message")
    retention: str | None = Field(default=None, description="Result retention period")


class JobStatusResponse(BaseResponse):
    """Response when polling job status."""

    job_id: str = Field(description="Job identifier")
    status: JobStatus = Field(description="Current job status")
    output: dict[str, Any] | None = Field(default=None, description="Job output (if complete)")
    output_url: str | None = Field(
        default=None, description="Presigned URL to fetch results (if B2-cached)"
    )
    output_expires_at: str | None = Field(
        default=None, description="When output_url expires"
    )
    output_size_bytes: int | None = Field(default=None, description="Size of cached result")
    created_at: str | None = Field(default=None, description="Job creation timestamp")
    completed_at: str | None = Field(default=None, description="Job completion timestamp")
    error_code: str | None = Field(default=None, description="Error code if failed")
    error_message: str | None = Field(default=None, description="Error message if failed")
    message: str | None = Field(default=None, description="Status message")


class JobListResponse(BaseResponse):
    """Response when listing jobs."""

    jobs: list[JobStatusResponse] = Field(default_factory=list, description="List of jobs")
    total: int = Field(default=0, description="Total number of jobs")
    limit: int = Field(default=100, description="Page size limit")
    offset: int = Field(default=0, description="Page offset")


class JobCancelResponse(BaseResponse):
    """Response when cancelling a job."""

    message: str = Field(description="Cancellation result message")
    warning: str | None = Field(default=None, description="Warning if any")


class CreditsResponse(BaseResponse):
    """Response from credits/balance check."""

    balance_usd: float = Field(description="Current USD balance")
