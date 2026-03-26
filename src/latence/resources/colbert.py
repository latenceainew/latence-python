"""ColBERT service resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union, overload

from .._models import ColBERTEmbedResponse, JobSubmittedResponse
from ._base import AsyncResource, SyncResource

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient


class ColBERT(SyncResource):
    """
    ColBERT service - token-level embeddings for neural retrieval.

    Example:
        >>> result = client.colbert.embed(text="What is AI?", is_query=True)
        >>> print(result.embeddings)  # Always float arrays
        >>> print(result.shape)  # [tokens, 128]
    """

    def __init__(self, client: BaseSyncClient) -> None:
        super().__init__(client)

    @overload
    def embed(
        self,
        text: str,
        *,
        is_query: bool = True,
        query_expansion: bool = True,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> ColBERTEmbedResponse: ...

    @overload
    def embed(
        self,
        text: str,
        *,
        is_query: bool = True,
        query_expansion: bool = True,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def embed(
        self,
        text: str,
        *,
        is_query: bool = True,
        query_expansion: bool = True,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[ColBERTEmbedResponse, JobSubmittedResponse]:
        """
        Generate ColBERT token-level embeddings.

        Args:
            text: Input text (1-100,000 chars)
            is_query: True for queries, False for documents
            query_expansion: Enable query expansion
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling

        Returns:
            ColBERTEmbedResponse with embeddings as float arrays
        """
        body = self._build_request_body(
            text=text,
            is_query=is_query,
            query_expansion=query_expansion,
            encoding_format="base64",  # Always use base64 internally (efficient)
            request_id=request_id,
            return_job=return_job,
        )

        response = self._client.post("/api/v1/colbert/embed", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = ColBERTEmbedResponse.model_validate(response.data)  # type: ignore[assignment]

        return self._inject_metadata(result, response)


class AsyncColBERT(AsyncResource):
    """Async ColBERT service."""

    def __init__(self, client: BaseAsyncClient) -> None:
        super().__init__(client)

    @overload
    async def embed(
        self,
        text: str,
        *,
        is_query: bool = True,
        query_expansion: bool = True,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> ColBERTEmbedResponse: ...

    @overload
    async def embed(
        self,
        text: str,
        *,
        is_query: bool = True,
        query_expansion: bool = True,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def embed(
        self,
        text: str,
        *,
        is_query: bool = True,
        query_expansion: bool = True,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[ColBERTEmbedResponse, JobSubmittedResponse]:
        """
        Generate ColBERT token-level embeddings.

        Args:
            text: Input text (1-100,000 chars)
            is_query: True for queries, False for documents
            query_expansion: Enable query expansion
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling

        Returns:
            ColBERTEmbedResponse with embeddings as float arrays
        """
        body = self._build_request_body(
            text=text,
            is_query=is_query,
            query_expansion=query_expansion,
            encoding_format="base64",  # Always use base64 internally (efficient)
            request_id=request_id,
            return_job=return_job,
        )

        response = await self._client.post("/api/v1/colbert/embed", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = ColBERTEmbedResponse.model_validate(response.data)  # type: ignore[assignment]

        return self._inject_metadata(result, response)
