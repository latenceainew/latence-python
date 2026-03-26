"""Pydantic models for the Embedding service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .common import BaseResponse, Usage
from .._utils import decode_base64_embeddings


class EmbedResponse(BaseResponse):
    """Response from the embedding service.
    
    Embeddings are always returned as float arrays, regardless of the
    encoding format used by the API. Base64 decoding happens automatically.
    """

    embeddings: list[list[float]] = Field(description="Generated embeddings as float arrays")
    dimension: int = Field(description="Embedding dimension")
    shape: list[int] = Field(description="Shape of embeddings array")
    encoding_format: Literal["float", "base64"] = Field(
        default="float", description="API encoding format (decoding is automatic)"
    )
    model: str = Field(description="Model used for embedding")
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

        # Handle batch base64 payloads where API returns one base64 blob per row:
        # embeddings = ["<row1_b64>", "<row2_b64>", ...], shape = [batch, dim]
        embeddings = data.get("embeddings")
        if (
            isinstance(embeddings, list)
            and embeddings
            and all(isinstance(item, str) for item in embeddings)
            and isinstance(shape, list)
            and len(shape) == 2
            and shape[0] == len(embeddings)
        ):
            decoded_rows: list[list[float]] = []
            for encoded_row in embeddings:
                raw_bytes = base64.b64decode(encoded_row)
                bytes_per_element = len(raw_bytes) / max(shape[1], 1)
                if bytes_per_element == 2:
                    dtype = "float16"
                elif bytes_per_element == 4:
                    dtype = "float32"
                else:
                    dtype = "float32"
                decoded_row = decode_base64_embeddings(encoded_row, [1, shape[1]], dtype=dtype)
                decoded_rows.append(decoded_row[0] if decoded_row else [])
            data["embeddings"] = decoded_rows
        
        # Normalize flat array to nested: [0.1, 0.2, ...] → [[0.1, 0.2, ...]]
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and len(embeddings) > 0 and not isinstance(embeddings[0], list):
            data["embeddings"] = [embeddings]
        
        # Set default model if missing (for compatibility with super-pod)
        if "model" not in data or data["model"] is None:
            data["model"] = "nomic-embed-text-v1.5"
        
        return data
