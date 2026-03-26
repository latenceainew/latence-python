"""Pydantic models for the unified Embed service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .common import BaseResponse, Usage
from .._utils import decode_base64_embeddings


# Type alias for embedding types
EmbedType = Literal["dense", "late_interaction", "image"]


class UnifiedEmbedResponse(BaseResponse):
    """Response from the unified embed service.

    This model handles responses from all three embedding types:
    - dense: Standard vector embeddings
    - late_interaction: ColBERT token-level embeddings
    - image: ColPali visual embeddings

    Embeddings are always returned as float arrays, regardless of the
    encoding format used by the API. Base64 decoding happens automatically.

    Attributes:
        type: The embedding type used (dense, late_interaction, image).
        embeddings: Generated embeddings as float arrays.
        shape: Shape of embeddings array (varies by type).
        encoding_format: API encoding format (decoding is automatic).
        dimension: Embedding dimension (dense type only).
        is_query: Whether this was a query embedding (late_interaction, image).
        tokens: Number of tokens (late_interaction type only).
        patches: Number of image patches (image type only).
        model: Model identifier.
        usage: Credit usage information.

    Example:
        >>> # Dense embeddings
        >>> result = client.embed.dense(text="Hello world", dimension=512)
        >>> print(result.type)  # "dense"
        >>> print(result.embeddings)  # [[0.123, -0.456, ...]]
        >>> print(result.dimension)  # 512

        >>> # Late interaction embeddings
        >>> result = client.embed.late_interaction(text="What is AI?")
        >>> print(result.type)  # "late_interaction"
        >>> print(result.tokens)  # Number of tokens

        >>> # Image embeddings
        >>> result = client.embed.image(image_path="doc.png")
        >>> print(result.type)  # "image"
        >>> print(result.patches)  # Number of image patches
    """

    type: EmbedType = Field(description="Embedding type used")
    embeddings: list[list[float]] = Field(description="Generated embeddings as float arrays")
    shape: list[int] = Field(description="Shape of embeddings array")
    encoding_format: Literal["float", "base64"] = Field(
        default="float", description="API encoding format (decoding is automatic)"
    )
    # Type-specific fields
    dimension: int | None = Field(default=None, description="Embedding dimension (dense only)")
    is_query: bool | None = Field(
        default=None, description="Whether this was a query (late_interaction, image)"
    )
    tokens: int | None = Field(
        default=None, description="Number of tokens (late_interaction only)"
    )
    patches: int | None = Field(default=None, description="Number of image patches (image only)")
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
        if (
            isinstance(embeddings, list)
            and len(embeddings) == 1
            and isinstance(embeddings[0], str)
        ):
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

        # Normalize flat array to nested: [0.1, 0.2, ...] → [[0.1, 0.2, ...]]
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and len(embeddings) > 0 and not isinstance(embeddings[0], list):
            data["embeddings"] = [embeddings]

        # Set default model based on type if missing
        embed_type = data.get("type")
        if "model" not in data or data["model"] is None:
            model_map = {
                "dense": "nomic-embed-text-v1.5",
                "late_interaction": "colbert",
                "image": "colpali",
            }
            data["model"] = model_map.get(embed_type, "embedding")

        # Compute tokens/patches from shape if missing
        if embed_type == "late_interaction":
            if ("tokens" not in data or data["tokens"] is None) and shape:
                data["tokens"] = shape[0] if len(shape) > 0 else None
        elif embed_type == "image":
            if ("patches" not in data or data["patches"] is None) and shape:
                data["patches"] = shape[0] if len(shape) > 0 else None

        return data
