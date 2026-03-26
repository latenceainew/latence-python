"""Pydantic models for the Redaction service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .common import BaseResponse, Entity, Usage


class RedactionConfig(BaseModel):
    """Configuration for PII detection and redaction."""

    mode: Literal["balanced", "strict", "recall", "precision"] = Field(
        default="balanced", description="Detection sensitivity mode"
    )
    threshold: float = Field(default=0.3, description="Confidence threshold")
    redact: bool = Field(default=False, description="Whether to redact PII")
    redaction_mode: Literal["mask", "replace"] = Field(default="mask", description="Redaction mode")
    chunk_size: int = Field(default=1024, description="Chunk size in tokens")
    enable_refinement: bool = Field(default=False, description="Enable refinement")
    refinement_threshold: float = Field(default=0.5, description="Refinement threshold")
    enforce_refinement: bool = Field(default=False, description="Force refinement on all")
    normalize_scores: bool = Field(default=True, description="Normalize confidence scores")


class DetectPIIResponse(BaseResponse):
    """Response from PII detection."""

    original_text: str = Field(description="Input text")
    entities: list[Entity] = Field(default_factory=list, description="Detected PII entities")
    entity_count: int = Field(default=0, description="Number of PII entities found")
    unique_labels: list[str] = Field(default_factory=list, description="Types of PII found")
    redacted_text: str | None = Field(default=None, description="Redacted text (if redact=True)")
    chunks_processed: int = Field(default=1, description="Number of chunks processed")
    usage: Usage | None = Field(default=None, description="Credit usage")
