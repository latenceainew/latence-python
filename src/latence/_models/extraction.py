"""Pydantic models for the Extraction service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .common import BaseResponse, Entity, Usage


class ExtractionConfig(BaseModel):
    """Configuration for entity extraction."""

    label_mode: Literal["user", "hybrid", "generated"] = Field(
        default="generated", description="Label generation mode"
    )
    user_labels: list[str] | None = Field(
        default=None, description="User-provided entity labels"
    )
    threshold: float = Field(default=0.3, description="Confidence threshold")
    chunk_size: int = Field(default=1024, description="Chunk size in tokens")
    enable_refinement: bool = Field(default=False, description="Enable LLM refinement")
    refinement_threshold: float = Field(default=0.5, description="Refinement threshold")
    enforce_refinement: bool = Field(default=False, description="Force refinement on all")
    flat_ner: bool = Field(default=True, description="Disable nested entities")
    multi_label: bool = Field(default=False, description="Allow multiple labels per span")


class ExtractResponse(BaseResponse):
    """Response from entity extraction."""

    original_text: str = Field(description="Input text")
    entities: list[Entity] = Field(default_factory=list, description="Extracted entities")
    entity_count: int = Field(default=0, description="Number of entities found")
    unique_labels: list[str] = Field(default_factory=list, description="Unique labels found")
    chunks_processed: int = Field(default=1, description="Number of chunks processed")
    labels_generated: bool | list[str] = Field(default=False, description="Auto-generated labels (True/list of label names)")
    usage: Usage | None = Field(default=None, description="Credit usage")
