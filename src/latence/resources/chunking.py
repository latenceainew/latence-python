"""Chunking service resource — text splitting with 4 strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union, overload

from .._models import JobSubmittedResponse
from .._models.chunking import ChunkResponse
from ._base import AsyncResource, SyncResource

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient


class Chunking(SyncResource):
    """
    Chunking service — split text into semantically meaningful chunks.

    Supports 4 strategies:
    - **character**: Fixed character-length splits (free)
    - **token**: Token-boundary splits (free)
    - **semantic**: Embedding-based semantic grouping (charged)
    - **hybrid**: Character splits refined with semantic coherence (charged)

    All strategies automatically detect and preserve Markdown structure.

    Example:
        >>> result = client.experimental.chunking.chunk(
        ...     text="Long document...", strategy="hybrid"
        ... )
        >>> for chunk in result.data.chunks:
        ...     print(f"[{chunk.index}] {chunk.char_count} chars")
    """

    def __init__(self, client: BaseSyncClient) -> None:
        super().__init__(client)

    @overload
    def chunk(
        self,
        text: str,
        *,
        strategy: str = "hybrid",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 64,
        semantic_threshold: float | None = None,
        semantic_window_size: int | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> ChunkResponse: ...

    @overload
    def chunk(
        self,
        text: str,
        *,
        strategy: str = "hybrid",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 64,
        semantic_threshold: float | None = None,
        semantic_window_size: int | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def chunk(
        self,
        text: str,
        *,
        strategy: str = "hybrid",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 64,
        semantic_threshold: float | None = None,
        semantic_window_size: int | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[ChunkResponse, JobSubmittedResponse]:
        """
        Split text into chunks with structural metadata.

        Args:
            text: Input text to chunk (1-5,000,000 characters).
            strategy: Splitting strategy — "character", "token", "semantic",
                or "hybrid". Character and token are free; semantic and hybrid
                are charged at $0.10/1M characters.
            chunk_size: Target chunk size (64-8192). Characters for
                character/semantic/hybrid, tokens for token strategy.
            chunk_overlap: Overlap between adjacent chunks (0 to chunk_size-1).
            min_chunk_size: Minimum chunk size — smaller chunks are discarded.
            semantic_threshold: Similarity threshold for boundary detection
                (0.1-0.95, default 0.5). Lower values split more aggressively
                at topic shifts. Only applies to semantic and hybrid strategies.
            semantic_window_size: Sentences per sliding window for topic
                comparison (1-10, default 3). Only applies to semantic and
                hybrid strategies.
            request_id: Optional tracking ID.
            return_job: If True, return job ID for async polling.

        Returns:
            ChunkResponse with chunks and metadata.
        """
        body = self._build_request_body(
            task="chunk",
            text=text,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            semantic_threshold=semantic_threshold,
            semantic_window_size=semantic_window_size,
            request_id=request_id,
            return_job=return_job,
        )

        response = self._client.post("/api/v1/chunking/chunk", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = ChunkResponse.model_validate(response.data)

        return self._inject_metadata(result, response)


class AsyncChunking(AsyncResource):
    """
    Async chunking service — split text into semantically meaningful chunks.

    See :class:`Chunking` for full documentation.
    """

    def __init__(self, client: BaseAsyncClient) -> None:
        super().__init__(client)

    @overload
    async def chunk(
        self,
        text: str,
        *,
        strategy: str = "hybrid",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 64,
        semantic_threshold: float | None = None,
        semantic_window_size: int | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> ChunkResponse: ...

    @overload
    async def chunk(
        self,
        text: str,
        *,
        strategy: str = "hybrid",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 64,
        semantic_threshold: float | None = None,
        semantic_window_size: int | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def chunk(
        self,
        text: str,
        *,
        strategy: str = "hybrid",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 64,
        semantic_threshold: float | None = None,
        semantic_window_size: int | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[ChunkResponse, JobSubmittedResponse]:
        """
        Split text into chunks with structural metadata.

        See Chunking.chunk() for full documentation.
        """
        body = self._build_request_body(
            task="chunk",
            text=text,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            semantic_threshold=semantic_threshold,
            semantic_window_size=semantic_window_size,
            request_id=request_id,
            return_job=return_job,
        )

        response = await self._client.post("/api/v1/chunking/chunk", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = ChunkResponse.model_validate(response.data)

        return self._inject_metadata(result, response)
