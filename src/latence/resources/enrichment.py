"""Enrichment service resource — chunking and feature enrichment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Union, overload

from .._models import JobSubmittedResponse
from .._models.enrichment import ChunkResponse, EnrichResponse
from ._base import AsyncResource, SyncResource

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient


class Enrichment(SyncResource):
    """
    Enrichment service — chunk documents and compute retrieval features.

    Example:
        >>> result = client.experimental.enrichment.chunk(
        ...     text="Long document...", strategy="hybrid"
        ... )
        >>> print(f"{result.data.num_chunks} chunks")

        >>> result = client.experimental.enrichment.enrich(text="Long document...")
        >>> print(result.data.features.keys())
    """

    def __init__(self, client: BaseSyncClient) -> None:
        super().__init__(client)

    # -----------------------------------------------------------------
    # chunk()
    # -----------------------------------------------------------------

    @overload
    def chunk(
        self,
        text: str,
        *,
        strategy: str = "hybrid",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 64,
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
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[ChunkResponse, JobSubmittedResponse]:
        """
        Chunk text into segments with structural metadata.

        Args:
            text: Input text to chunk (1-5,000,000 characters).
            strategy: Splitting strategy — "character", "token", "semantic", or "hybrid".
            chunk_size: Target chunk size (64-8192).
            chunk_overlap: Overlap between adjacent chunks (0 to chunk_size-1).
            min_chunk_size: Minimum chunk size (1 to chunk_size).
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
            request_id=request_id,
            return_job=return_job,
        )

        response = self._client.post("/api/v1/enrichment/chunk", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = ChunkResponse.model_validate(response.data)

        return self._inject_metadata(result, response)

    # -----------------------------------------------------------------
    # enrich()
    # -----------------------------------------------------------------

    @overload
    def enrich(
        self,
        text: str,
        *,
        strategy: str = "hybrid",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 64,
        encoding_format: str = "float",
        features: list[str] | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> EnrichResponse: ...

    @overload
    def enrich(
        self,
        text: str,
        *,
        strategy: str = "hybrid",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 64,
        encoding_format: str = "float",
        features: list[str] | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def enrich(
        self,
        text: str,
        *,
        strategy: str = "hybrid",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 64,
        encoding_format: str = "float",
        features: list[str] | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[EnrichResponse, JobSubmittedResponse]:
        """
        Chunk, embed, and compute 10 enrichment feature groups.

        Feature groups: quality, density, structural, semantic, compression,
        zipf, coherence, spectral, drift, redundancy.

        Args:
            text: Input text (1-5,000,000 characters).
            strategy: Splitting strategy — "character", "token", "semantic", or "hybrid".
            chunk_size: Target chunk size (64-8192).
            chunk_overlap: Overlap between adjacent chunks.
            min_chunk_size: Minimum chunk size.
            encoding_format: Embedding output format — "float" or "base64".
            features: Feature groups to compute. Default: all 10.
            request_id: Optional tracking ID.
            return_job: If True, return job ID for async polling.

        Returns:
            EnrichResponse with chunks, embeddings, and feature results.
        """
        body = self._build_request_body(
            task="enrich",
            text=text,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            encoding_format=encoding_format,
            features=features,
            request_id=request_id,
            return_job=return_job,
        )

        response = self._client.post("/api/v1/enrichment/enrich", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = EnrichResponse.model_validate(response.data)

        return self._inject_metadata(result, response)


class AsyncEnrichment(AsyncResource):
    """
    Async enrichment service — chunk documents and compute retrieval features.

    Example:
        >>> result = await client.experimental.enrichment.chunk(
        ...     text="Long document...", strategy="hybrid"
        ... )
        >>> print(f"{result.data.num_chunks} chunks")
    """

    def __init__(self, client: BaseAsyncClient) -> None:
        super().__init__(client)

    # -----------------------------------------------------------------
    # chunk()
    # -----------------------------------------------------------------

    @overload
    async def chunk(
        self,
        text: str,
        *,
        strategy: str = "hybrid",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 64,
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
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[ChunkResponse, JobSubmittedResponse]:
        """
        Chunk text into segments with structural metadata.

        See Enrichment.chunk() for full documentation.
        """
        body = self._build_request_body(
            task="chunk",
            text=text,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            request_id=request_id,
            return_job=return_job,
        )

        response = await self._client.post("/api/v1/enrichment/chunk", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = ChunkResponse.model_validate(response.data)

        return self._inject_metadata(result, response)

    # -----------------------------------------------------------------
    # enrich()
    # -----------------------------------------------------------------

    @overload
    async def enrich(
        self,
        text: str,
        *,
        strategy: str = "hybrid",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 64,
        encoding_format: str = "float",
        features: list[str] | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> EnrichResponse: ...

    @overload
    async def enrich(
        self,
        text: str,
        *,
        strategy: str = "hybrid",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 64,
        encoding_format: str = "float",
        features: list[str] | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def enrich(
        self,
        text: str,
        *,
        strategy: str = "hybrid",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 64,
        encoding_format: str = "float",
        features: list[str] | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[EnrichResponse, JobSubmittedResponse]:
        """
        Chunk, embed, and compute 10 enrichment feature groups.

        See Enrichment.enrich() for full documentation.
        """
        body = self._build_request_body(
            task="enrich",
            text=text,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            encoding_format=encoding_format,
            features=features,
            request_id=request_id,
            return_job=return_job,
        )

        response = await self._client.post("/api/v1/enrichment/enrich", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = EnrichResponse.model_validate(response.data)

        return self._inject_metadata(result, response)
