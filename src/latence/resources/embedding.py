"""Embedding service resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union, overload

from .._models import EmbedResponse, JobSubmittedResponse
from ._base import AsyncResource, SyncResource

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient


class Embedding(SyncResource):
    """
    Embedding service - generate dense vector embeddings.

    Example:
        >>> result = client.embedding.embed(text="Hello world", dimension=512)
        >>> print(result.embeddings)  # Always float arrays
        >>> print(result.shape)  # [1, 512]
    """

    def __init__(self, client: BaseSyncClient) -> None:
        super().__init__(client)

    @overload
    def embed(
        self,
        text: str | list[str],
        *,
        dimension: int = 512,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> EmbedResponse: ...

    @overload
    def embed(
        self,
        text: str | list[str],
        *,
        dimension: int = 512,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def embed(
        self,
        text: str | list[str],
        *,
        dimension: int = 512,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[EmbedResponse, JobSubmittedResponse]:
        """
        Generate dense vector embeddings for text.

        Args:
            text: Input text or batch of texts (each 1-100,000 chars)
            dimension: Embedding dimension (256, 512, 768, 1024)
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling

        Returns:
            EmbedResponse with embeddings as float arrays
        """
        body = self._build_request_body(
            text=text,
            dimension=dimension,
            encoding_format="base64",  # Always use base64 internally (efficient)
            request_id=request_id,
            return_job=return_job,
        )

        response = self._client.post("/api/v1/embedding/embed", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = EmbedResponse.model_validate(response.data)

        return self._inject_metadata(result, response)


class AsyncEmbedding(AsyncResource):
    """Async embedding service."""

    def __init__(self, client: BaseAsyncClient) -> None:
        super().__init__(client)

    @overload
    async def embed(
        self,
        text: str | list[str],
        *,
        dimension: int = 512,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> EmbedResponse: ...

    @overload
    async def embed(
        self,
        text: str | list[str],
        *,
        dimension: int = 512,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def embed(
        self,
        text: str | list[str],
        *,
        dimension: int = 512,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[EmbedResponse, JobSubmittedResponse]:
        """
        Generate dense vector embeddings for text.

        Args:
            text: Input text or batch of texts (each 1-100,000 chars)
            dimension: Embedding dimension (256, 512, 768, 1024)
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling

        Returns:
            EmbedResponse with embeddings as float arrays
        """
        body = self._build_request_body(
            text=text,
            dimension=dimension,
            encoding_format="base64",  # Always use base64 internally (efficient)
            request_id=request_id,
            return_job=return_job,
        )

        response = await self._client.post("/api/v1/embedding/embed", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = EmbedResponse.model_validate(response.data)

        return self._inject_metadata(result, response)
