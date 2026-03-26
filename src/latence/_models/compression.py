"""Pydantic models for the Compression service."""

from __future__ import annotations

from pydantic import Field

from .common import BaseResponse, Message, Usage


class CompressResponse(BaseResponse):
    """Response from text compression."""

    compressed_text: str = Field(description="Compressed text")
    original_tokens: int = Field(description="Original token count")
    compressed_tokens: int = Field(description="Compressed token count")
    compression_ratio: float = Field(description="Compression ratio achieved")
    tokens_saved: int = Field(description="Tokens saved")
    toon_applied: bool = Field(default=False, description="Whether TOON encoding was applied")
    usage: Usage | None = Field(default=None, description="Credit usage")


class CompressMessagesResponse(BaseResponse):
    """Response from chat message compression."""

    compressed_messages: list[Message] = Field(description="Compressed messages")
    statistics: dict[str, int] | None = Field(default=None, description="Compression statistics")
    original_total_length: int = Field(description="Original total character length")
    compressed_total_length: int = Field(description="Compressed total character length")
    average_compression: float = Field(default=0.0, description="Average compression ratio")
    compression_percentage: float = Field(description="Compression percentage achieved")
    usage: Usage | None = Field(default=None, description="Credit usage")
