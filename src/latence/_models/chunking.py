"""Pydantic models for the Chunking service."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .common import BaseResponse, Usage

# ---------------------------------------------------------------------------
# Chunk item
# ---------------------------------------------------------------------------


class ChunkItem(BaseModel):
    """A single chunk with structural metadata."""

    model_config = ConfigDict(extra="allow")

    content: str = Field(description="Chunk text content")
    index: int = Field(description="0-based chunk position")
    start: int = Field(description="Start character offset in source text")
    end: int = Field(description="End character offset in source text")
    char_count: int = Field(description="Character count of chunk")
    token_count: int | None = Field(default=None, description="Token count (tiktoken o200k_base)")
    semantic_score: float | None = Field(
        default=None, description="Intra-chunk coherence score (0-1)"
    )
    section_path: list[str] | None = Field(
        default=None, description="Heading hierarchy at chunk position"
    )


# ---------------------------------------------------------------------------
# Chunk response
# ---------------------------------------------------------------------------


class ChunkData(BaseModel):
    """Data payload for chunk response."""

    model_config = ConfigDict(extra="allow")

    chunks: list[ChunkItem] = Field(description="Chunk objects with metadata")
    num_chunks: int = Field(description="Total number of chunks")
    strategy: str = Field(description="Chunking strategy used")
    chunk_size: int = Field(description="Target chunk size parameter")
    processing_time_ms: float = Field(description="Processing time in milliseconds")


class ChunkResponse(BaseResponse):
    """Response from the chunking service."""

    data: ChunkData = Field(description="Chunk data payload")
    usage: Usage | None = Field(default=None, description="Credit usage")
