"""Common Pydantic models shared across services."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Usage(BaseModel):
    """Credit usage information returned with API responses."""

    model_config = ConfigDict(extra="allow")

    credits: float = Field(default=0.0, description="Credits consumed by this request")
    total_credits: float | None = Field(default=None, description="Total credits used")


class Entity(BaseModel):
    """An extracted entity from text."""

    model_config = ConfigDict(extra="allow")

    start: int | None = Field(default=None, description="Start character position")
    end: int | None = Field(default=None, description="End character position")
    text: str = Field(description="Entity text")
    label: str = Field(description="Entity label/type")
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score (0-1)")
    source: str | None = Field(default=None, description="Source of extraction (model/regex)")


class Relation(BaseModel):
    """A relation between two entities."""

    model_config = ConfigDict(extra="allow")

    source_entity: str = Field(description="Source entity text")
    target_entity: str = Field(description="Target entity text")
    relation_type: str = Field(description="Type of relation")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1)")


class KnowledgeGraph(BaseModel):
    """Knowledge graph structure."""

    model_config = ConfigDict(extra="allow")

    nodes: list[dict[str, Any]] = Field(default_factory=list, description="Graph nodes")
    edges: list[dict[str, Any]] = Field(default_factory=list, description="Graph edges")


class Message(BaseModel):
    """A chat message."""

    role: str = Field(description="Message role (user/assistant/system)")
    content: str = Field(description="Message content")


class CustomLabel(BaseModel):
    """Custom regex-based label extractor."""

    label_name: str = Field(description="Name for the label")
    extractor: str = Field(description="Regex pattern for extraction")


class BaseResponse(BaseModel):
    """Base response with common metadata fields."""

    model_config = ConfigDict(extra="allow")

    success: bool = Field(default=True, description="Whether the request succeeded")
    request_id: str | None = Field(default=None, description="Request tracking ID")

    # Response metadata from headers (injected by client)
    credits_used: float | None = Field(default=None, description="Credits used for this request")
    credits_remaining: float | None = Field(default=None, description="Remaining credit balance")
    rate_limit_remaining: int | None = Field(
        default=None, description="Remaining requests in rate limit window"
    )
