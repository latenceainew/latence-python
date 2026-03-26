"""ColPali service resource."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, Union, overload

from .._models import ColPaliEmbedResponse, JobSubmittedResponse
from .._utils import image_to_base64
from ._base import AsyncResource, SyncResource

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient


class ColPali(SyncResource):
    """
    ColPali service - visual document retrieval embeddings.

    Example:
        >>> # Text query
        >>> result = client.colpali.embed(text="Find invoices from 2024")
        >>> print(result.embeddings)  # Always float arrays

        >>> # Image embedding from file
        >>> result = client.colpali.embed(image_path="/path/to/page.png", is_query=False)
        >>> print(result.shape)  # [patches, 128]
    """

    def __init__(self, client: BaseSyncClient) -> None:
        super().__init__(client)

    @overload
    def embed(
        self,
        *,
        text: str | None = None,
        image: str | None = None,
        image_path: str | Path | BinaryIO | None = None,
        is_query: bool = True,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> ColPaliEmbedResponse: ...

    @overload
    def embed(
        self,
        *,
        text: str | None = None,
        image: str | None = None,
        image_path: str | Path | BinaryIO | None = None,
        is_query: bool = True,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def embed(
        self,
        *,
        text: str | None = None,
        image: str | None = None,
        image_path: str | Path | BinaryIO | None = None,
        is_query: bool = True,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[ColPaliEmbedResponse, JobSubmittedResponse]:
        """
        Generate ColPali visual embeddings.

        Provide one of: text (for queries), image (base64), or image_path (local file).

        Args:
            text: Query text
            image: Base64-encoded image (with or without data URI prefix)
            image_path: Local image file path - automatically encoded
            is_query: True for queries, False for images
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling

        Returns:
            ColPaliEmbedResponse with embeddings as float arrays
        """
        # Handle image_path conversion (automatic base64 encoding)
        if image_path is not None:
            image = image_to_base64(image_path)

        if text is None and image is None:
            raise ValueError("One of text, image, or image_path must be provided")

        body = self._build_request_body(
            text=text,
            image=image,
            is_query=is_query,
            encoding_format="base64",  # Always use base64 internally (efficient)
            request_id=request_id,
            return_job=return_job,
        )

        response = self._client.post("/api/v1/colpali/embed", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = ColPaliEmbedResponse.model_validate(response.data)

        return self._inject_metadata(result, response)


class AsyncColPali(AsyncResource):
    """Async ColPali service."""

    def __init__(self, client: BaseAsyncClient) -> None:
        super().__init__(client)

    @overload
    async def embed(
        self,
        *,
        text: str | None = None,
        image: str | None = None,
        image_path: str | Path | BinaryIO | None = None,
        is_query: bool = True,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> ColPaliEmbedResponse: ...

    @overload
    async def embed(
        self,
        *,
        text: str | None = None,
        image: str | None = None,
        image_path: str | Path | BinaryIO | None = None,
        is_query: bool = True,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def embed(
        self,
        *,
        text: str | None = None,
        image: str | None = None,
        image_path: str | Path | BinaryIO | None = None,
        is_query: bool = True,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[ColPaliEmbedResponse, JobSubmittedResponse]:
        """
        Generate ColPali visual embeddings.

        Provide one of: text (for queries), image (base64), or image_path (local file).

        Args:
            text: Query text
            image: Base64-encoded image (with or without data URI prefix)
            image_path: Local image file path - automatically encoded
            is_query: True for queries, False for images
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling

        Returns:
            ColPaliEmbedResponse with embeddings as float arrays
        """
        # Handle image_path conversion (automatic base64 encoding)
        if image_path is not None:
            image = image_to_base64(image_path)

        if text is None and image is None:
            raise ValueError("One of text, image, or image_path must be provided")

        body = self._build_request_body(
            text=text,
            image=image,
            is_query=is_query,
            encoding_format="base64",  # Always use base64 internally (efficient)
            request_id=request_id,
            return_job=return_job,
        )

        response = await self._client.post("/api/v1/colpali/embed", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = ColPaliEmbedResponse.model_validate(response.data)

        return self._inject_metadata(result, response)
