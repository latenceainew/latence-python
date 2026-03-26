"""Pydantic models for the Dataset Intelligence service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .common import BaseResponse


class DatasetIntelligenceStageTiming(BaseModel):
    """Timing information for a single pipeline stage."""

    model_config = ConfigDict(extra="allow")

    stage: str = ""
    elapsed_ms: float = 0.0
    status: str = "completed"


class DatasetIntelligenceUsage(BaseModel):
    """Credit usage breakdown."""

    model_config = ConfigDict(extra="allow")

    credits: float = 0.0
    calculation: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class DatasetIntelligenceDeltaSummary(BaseModel):
    """Summary of changes in an incremental (append) ingestion."""

    model_config = ConfigDict(extra="allow")

    files_added: int = 0
    files_updated: int = 0
    files_unchanged: int = 0
    new_entities: int = 0
    merged_entities: int = 0
    new_edges: int = 0
    removed_edges: int = 0
    ontology_types_added: int = 0
    rotate_retrain: str = ""
    delta_ratio: float = 0.0
    train_metrics: dict[str, Any] | None = None


class DatasetIntelligenceResponse(BaseResponse):
    """Response from the Dataset Intelligence service.

    Returned by ``enrich()``, ``build_graph()``, ``build_ontology()``,
    and ``run()`` on the SDK resource.
    """

    model_config = ConfigDict(extra="allow")

    endpoint_id: str = "dataset_intelligence"
    tier: str = ""
    dataset_id: str = ""
    mode: str = "create"
    data: dict[str, Any] = Field(default_factory=dict)
    usage: DatasetIntelligenceUsage = Field(
        default_factory=DatasetIntelligenceUsage
    )
    stage_timings: list[DatasetIntelligenceStageTiming] = Field(
        default_factory=list
    )
    delta_summary: DatasetIntelligenceDeltaSummary | None = None
    processing_time_ms: float = 0.0
    version: str = ""
    error: str | None = None
    error_code: str | None = None
