"""Unified Embed service resource.

This module provides a unified interface to all embedding capabilities:
- Dense embeddings (standard vector embeddings)
- Late interaction embeddings (ColBERT token-level)
- Image embeddings (ColPali visual)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, Union, overload

from .._models import JobSubmittedResponse
from .._models.embed import UnifiedEmbedResponse
from .._utils import image_to_base64
from ._base import AsyncResource, SyncResource

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient


class Embed(SyncResource):
    """
    Unified embedding service - single interface for all embedding types.

    Provides three embedding modes:
    - dense(): Standard vector embeddings with Matryoshka dimension support
    - late_interaction(): ColBERT token-level embeddings for neural retrieval
    - image(): ColPali visual embeddings for document retrieval

    Example:
        >>> # Dense embeddings
        >>> result = client.embed.dense(text="Hello world", dimension=512)
        >>> print(result.type)  # "dense"
        >>> print(result.embeddings)  # [[0.123, -0.456, ...]]

        >>> # Late interaction (ColBERT)
        >>> result = client.embed.late_interaction(text="What is AI?", is_query=True)
        >>> print(result.tokens)  # Number of token embeddings

        >>> # Image embeddings (ColPali)
        >>> result = client.embed.image(image_path="document.png", is_query=False)
        >>> print(result.patches)  # Number of image patches
    """

    def __init__(self, client: BaseSyncClient) -> None:
        """Initialize the Embed resource.

        Args:
            client: The sync client instance.
        """
        super().__init__(client)

    # =========================================================================
    # DENSE EMBEDDINGS
    # =========================================================================

    @overload
    def dense(
        self,
        text: str | list[str],
        *,
        dimension: int = 512,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> UnifiedEmbedResponse: ...

    @overload
    def dense(
        self,
        text: str | list[str],
        *,
        dimension: int = 512,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def dense(
        self,
        text: str | list[str],
        *,
        dimension: int = 512,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[UnifiedEmbedResponse, JobSubmittedResponse]:
        """
        Generate dense vector embeddings for text.

        Dense embeddings are standard high-dimensional vectors ideal for
        semantic similarity search and retrieval tasks.

        Args:
            text: Input text or batch of texts (each 1-100,000 characters).
            dimension: Embedding dimension. Supports Matryoshka dimensions:
                256, 512, 768, or 1024. Default: 512.
            request_id: Optional request identifier for tracking.
            return_job: If True, returns a job ID for async polling
                instead of waiting for results.

        Returns:
            UnifiedEmbedResponse with embeddings as float arrays.
            If return_job=True, returns JobSubmittedResponse instead.

        Raises:
            ValidationError: If text is empty or too long.
            AuthenticationError: If API key is invalid.
            RateLimitError: If rate limit exceeded.

        Example:
            >>> result = client.embed.dense(
            ...     text="Machine learning transforms industries.",
            ...     dimension=512
            ... )
            >>> print(len(result.embeddings[0]))  # 512
            >>> print(result.dimension)  # 512
        """
        body = self._build_request_body(
            type="dense",
            text=text,
            dimension=dimension,
            encoding_format="base64",  # Always use base64 internally (efficient)
            request_id=request_id,
            return_job=return_job,
        )

        response = self._client.post("/api/v1/embed/embed", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = UnifiedEmbedResponse.model_validate(response.data)

        return self._inject_metadata(result, response)

    # =========================================================================
    # LATE INTERACTION EMBEDDINGS (ColBERT)
    # =========================================================================

    @overload
    def late_interaction(
        self,
        text: str,
        *,
        is_query: bool = True,
        query_expansion: bool = True,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> UnifiedEmbedResponse: ...

    @overload
    def late_interaction(
        self,
        text: str,
        *,
        is_query: bool = True,
        query_expansion: bool = True,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def late_interaction(
        self,
        text: str,
        *,
        is_query: bool = True,
        query_expansion: bool = True,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[UnifiedEmbedResponse, JobSubmittedResponse]:
        """
        Generate late interaction (ColBERT) token-level embeddings.

        Late interaction embeddings produce a vector per token, enabling
        more nuanced similarity matching. Ideal for neural retrieval where
        query-document matching benefits from token-level comparisons.

        Args:
            text: Input text to embed (1-100,000 characters).
            is_query: True for query embeddings, False for document embeddings.
                Query embeddings are optimized for search, document embeddings
                for indexing. Default: True.
            query_expansion: Enable query expansion for better retrieval.
                Only applies when is_query=True. Default: True.
            request_id: Optional request identifier for tracking.
            return_job: If True, returns a job ID for async polling
                instead of waiting for results.

        Returns:
            UnifiedEmbedResponse with token-level embeddings as float arrays.
            If return_job=True, returns JobSubmittedResponse instead.

        Raises:
            ValidationError: If text is empty or too long.
            AuthenticationError: If API key is invalid.
            RateLimitError: If rate limit exceeded.

        Example:
            >>> # Query embedding
            >>> query_result = client.embed.late_interaction(
            ...     text="What is machine learning?",
            ...     is_query=True
            ... )
            >>> print(query_result.tokens)  # e.g., 32

            >>> # Document embedding
            >>> doc_result = client.embed.late_interaction(
            ...     text="Machine learning is a subset of AI...",
            ...     is_query=False
            ... )
            >>> print(doc_result.shape)  # [tokens, 128]
        """
        body = self._build_request_body(
            type="late_interaction",
            text=text,
            is_query=is_query,
            query_expansion=query_expansion,
            encoding_format="base64",  # Always use base64 internally (efficient)
            request_id=request_id,
            return_job=return_job,
        )

        response = self._client.post("/api/v1/embed/embed", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = UnifiedEmbedResponse.model_validate(response.data)

        return self._inject_metadata(result, response)

    # =========================================================================
    # IMAGE EMBEDDINGS (ColPali)
    # =========================================================================

    @overload
    def image(
        self,
        *,
        text: str | None = None,
        image: str | None = None,
        image_path: str | Path | BinaryIO | None = None,
        is_query: bool = True,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> UnifiedEmbedResponse: ...

    @overload
    def image(
        self,
        *,
        text: str | None = None,
        image: str | None = None,
        image_path: str | Path | BinaryIO | None = None,
        is_query: bool = True,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def image(
        self,
        *,
        text: str | None = None,
        image: str | None = None,
        image_path: str | Path | BinaryIO | None = None,
        is_query: bool = True,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[UnifiedEmbedResponse, JobSubmittedResponse]:
        """
        Generate visual embeddings for images or text queries.

        Image embeddings (ColPali) enable multi-modal retrieval where you can:
        - Embed images for indexing (is_query=False)
        - Embed text queries to search across images (is_query=True)

        Args:
            text: Query text to embed. Required when is_query=True.
            image: Base64-encoded image data. Required when is_query=False
                unless image_path is provided.
            image_path: Path to local image file. The file will be automatically
                read and base64 encoded. Alternative to providing image directly.
            is_query: True for text queries, False for image documents.
                Default: True.
            request_id: Optional request identifier for tracking.
            return_job: If True, returns a job ID for async polling
                instead of waiting for results.

        Returns:
            UnifiedEmbedResponse with visual embeddings as float arrays.
            If return_job=True, returns JobSubmittedResponse instead.

        Raises:
            ValueError: If neither text, image, nor image_path is provided.
            ValidationError: If input is invalid.
            AuthenticationError: If API key is invalid.
            RateLimitError: If rate limit exceeded.

        Example:
            >>> # Text query to search images
            >>> query_result = client.embed.image(
            ...     text="Find documents about revenue",
            ...     is_query=True
            ... )

            >>> # Embed an image document
            >>> doc_result = client.embed.image(
            ...     image_path="report_page1.png",
            ...     is_query=False
            ... )
            >>> print(doc_result.patches)  # Number of image patches
        """
        # Handle image_path conversion (automatic base64 encoding)
        if image_path is not None:
            image = image_to_base64(image_path)

        if text is None and image is None:
            raise ValueError(
                "One of text, image, or image_path must be provided. "
                "Use text for queries (is_query=True) or image/image_path "
                "for documents (is_query=False)."
            )

        body = self._build_request_body(
            type="image",
            text=text,
            image=image,
            is_query=is_query,
            encoding_format="base64",  # Always use base64 internally (efficient)
            request_id=request_id,
            return_job=return_job,
        )

        response = self._client.post("/api/v1/embed/embed", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = UnifiedEmbedResponse.model_validate(response.data)

        return self._inject_metadata(result, response)


class AsyncEmbed(AsyncResource):
    """
    Async unified embedding service.

    Provides the same interface as Embed but with async/await support.

    Example:
        >>> async with AsyncLatence(api_key="...") as client:
        ...     result = await client.embed.dense(text="Hello world")
        ...     print(result.embeddings)
    """

    def __init__(self, client: BaseAsyncClient) -> None:
        """Initialize the async Embed resource.

        Args:
            client: The async client instance.
        """
        super().__init__(client)

    # =========================================================================
    # DENSE EMBEDDINGS (ASYNC)
    # =========================================================================

    @overload
    async def dense(
        self,
        text: str | list[str],
        *,
        dimension: int = 512,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> UnifiedEmbedResponse: ...

    @overload
    async def dense(
        self,
        text: str | list[str],
        *,
        dimension: int = 512,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def dense(
        self,
        text: str | list[str],
        *,
        dimension: int = 512,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[UnifiedEmbedResponse, JobSubmittedResponse]:
        """
        Generate dense vector embeddings for text (async).

        See Embed.dense() for full documentation.

        Args:
            text: Input text or batch of texts (each 1-100,000 characters).
            dimension: Embedding dimension (256, 512, 768, 1024). Default: 512.
            request_id: Optional request identifier for tracking.
            return_job: If True, returns a job ID for async polling.

        Returns:
            UnifiedEmbedResponse with embeddings as float arrays.
        """
        body = self._build_request_body(
            type="dense",
            text=text,
            dimension=dimension,
            encoding_format="base64",
            request_id=request_id,
            return_job=return_job,
        )

        response = await self._client.post("/api/v1/embed/embed", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = UnifiedEmbedResponse.model_validate(response.data)

        return self._inject_metadata(result, response)

    # =========================================================================
    # LATE INTERACTION EMBEDDINGS (ASYNC)
    # =========================================================================

    @overload
    async def late_interaction(
        self,
        text: str,
        *,
        is_query: bool = True,
        query_expansion: bool = True,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> UnifiedEmbedResponse: ...

    @overload
    async def late_interaction(
        self,
        text: str,
        *,
        is_query: bool = True,
        query_expansion: bool = True,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def late_interaction(
        self,
        text: str,
        *,
        is_query: bool = True,
        query_expansion: bool = True,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[UnifiedEmbedResponse, JobSubmittedResponse]:
        """
        Generate late interaction (ColBERT) token-level embeddings (async).

        See Embed.late_interaction() for full documentation.

        Args:
            text: Input text to embed (1-100,000 characters).
            is_query: True for queries, False for documents. Default: True.
            query_expansion: Enable query expansion. Default: True.
            request_id: Optional request identifier for tracking.
            return_job: If True, returns a job ID for async polling.

        Returns:
            UnifiedEmbedResponse with token-level embeddings.
        """
        body = self._build_request_body(
            type="late_interaction",
            text=text,
            is_query=is_query,
            query_expansion=query_expansion,
            encoding_format="base64",
            request_id=request_id,
            return_job=return_job,
        )

        response = await self._client.post("/api/v1/embed/embed", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = UnifiedEmbedResponse.model_validate(response.data)

        return self._inject_metadata(result, response)

    # =========================================================================
    # IMAGE EMBEDDINGS (ASYNC)
    # =========================================================================

    @overload
    async def image(
        self,
        *,
        text: str | None = None,
        image: str | None = None,
        image_path: str | Path | BinaryIO | None = None,
        is_query: bool = True,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> UnifiedEmbedResponse: ...

    @overload
    async def image(
        self,
        *,
        text: str | None = None,
        image: str | None = None,
        image_path: str | Path | BinaryIO | None = None,
        is_query: bool = True,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def image(
        self,
        *,
        text: str | None = None,
        image: str | None = None,
        image_path: str | Path | BinaryIO | None = None,
        is_query: bool = True,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[UnifiedEmbedResponse, JobSubmittedResponse]:
        """
        Generate visual embeddings for images or text queries (async).

        See Embed.image() for full documentation.

        Args:
            text: Query text (required when is_query=True).
            image: Base64-encoded image (required when is_query=False).
            image_path: Path to local image file.
            is_query: True for text queries, False for image documents.
            request_id: Optional request identifier for tracking.
            return_job: If True, returns a job ID for async polling.

        Returns:
            UnifiedEmbedResponse with visual embeddings.
        """
        # Handle image_path conversion
        if image_path is not None:
            image = image_to_base64(image_path)

        if text is None and image is None:
            raise ValueError(
                "One of text, image, or image_path must be provided. "
                "Use text for queries (is_query=True) or image/image_path "
                "for documents (is_query=False)."
            )

        body = self._build_request_body(
            type="image",
            text=text,
            image=image,
            is_query=is_query,
            encoding_format="base64",
            request_id=request_id,
            return_job=return_job,
        )

        response = await self._client.post("/api/v1/embed/embed", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = UnifiedEmbedResponse.model_validate(response.data)

        return self._inject_metadata(result, response)
