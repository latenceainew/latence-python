"""Pydantic models for the ColBERT service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .common import BaseResponse, Usage
from .._utils import decode_base64_embeddings


class ColBERTEmbedResponse(BaseResponse):
    """Response from the ColBERT embedding service.
    
    Embeddings are always returned as float arrays, regardless of the
    encoding format used by the API. Base64 decoding happens automatically.
    """

    embeddings: list[list[float]] = Field(
        description="Token-level embeddings as float arrays"
    )
    shape: list[int] = Field(description="Shape of embeddings [tokens, dim]")
    encoding_format: Literal["float", "base64"] = Field(
        default="base64", description="API encoding format (decoding is automatic)"
    )
    is_query: bool = Field(description="Whether this was a query embedding")
    tokens: int | None = Field(default=None, description="Number of tokens")
    model: str = Field(description="Model used")
    usage: Usage | None = Field(default=None, description="Credit usage")

    @model_validator(mode="before")
    @classmethod
    def decode_embeddings(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Automatically decode base64 embeddings to float arrays."""
        import base64
        
        embeddings = data.get("embeddings")
        shape = data.get("shape")
        
        # Handle list with single base64 string (unwrap it)
        if isinstance(embeddings, list) and len(embeddings) == 1 and isinstance(embeddings[0], str):
            embeddings = embeddings[0]
            data["embeddings"] = embeddings
        
        # If embeddings is a string (base64), decode it
        if isinstance(embeddings, str) and shape:
            # Auto-detect dtype by checking byte size
            raw_bytes = base64.b64decode(embeddings)
            total_elements = 1
            for dim in shape:
                total_elements *= dim
            
            # Determine dtype from byte size
            bytes_per_element = len(raw_bytes) / total_elements
            if bytes_per_element == 2:
                dtype = "float16"
            elif bytes_per_element == 4:
                dtype = "float32"
            else:
                # Default to float32 if unclear
                dtype = "float32"
            
            data["embeddings"] = decode_base64_embeddings(embeddings, shape, dtype=dtype)
        
        # Set default model if missing
        if "model" not in data or data["model"] is None:
            data["model"] = "colbert"
        
        # Compute tokens from shape if missing
        if ("tokens" not in data or data["tokens"] is None) and shape:
            data["tokens"] = shape[0] if len(shape) > 0 else None
        
        return data
