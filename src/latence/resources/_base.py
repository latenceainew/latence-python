"""Base resource class for service resources."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import httpx
from pydantic import BaseModel

from .._base import APIResponse
from .._constants import B2_UPLOAD_TIMEOUT

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient

T = TypeVar("T", bound=BaseModel)


def _guess_content_type(filename: str) -> str:
    """Guess MIME type from filename, defaulting to application/octet-stream."""
    ct, _ = mimetypes.guess_type(filename)
    return ct or "application/octet-stream"


def _inject_response_metadata(model: T, response: APIResponse) -> T:
    """Inject response metadata into a Pydantic model — shared implementation."""
    if hasattr(model, "credits_used"):
        model.credits_used = response.metadata.credits_used
    if hasattr(model, "credits_remaining"):
        model.credits_remaining = response.metadata.credits_remaining
    if hasattr(model, "rate_limit_remaining"):
        model.rate_limit_remaining = response.metadata.rate_limit_remaining
    if hasattr(model, "request_id") and model.request_id is None:
        model.request_id = response.metadata.request_id
    return model


def _build_common_request_body(
    *,
    return_job: bool = False,
    request_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build request body with common parameters — shared implementation."""
    body: dict[str, Any] = {}

    for key, value in kwargs.items():
        if value is not None:
            body[key] = value

    if return_job:
        body["async"] = True

    if request_id:
        body["request_id"] = request_id

    return body


class SyncResource:
    """Base class for synchronous service resources."""

    def __init__(self, client: BaseSyncClient) -> None:
        self._client = client

    def _inject_metadata(self, model: T, response: APIResponse) -> T:
        return _inject_response_metadata(model, response)

    def _build_request_body(
        self,
        *,
        return_job: bool = False,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return _build_common_request_body(return_job=return_job, request_id=request_id, **kwargs)

    def _presigned_upload(self, file_path: str | Path, filename: str) -> str:
        """Upload a large file via presigned URL, return the file_url for processing."""
        content_type = _guess_content_type(filename)

        resp = self._client.post(
            "/api/v1/upload/presign",
            json={"filename": filename, "content_type": content_type},
        )
        upload_url: str = resp.data["upload_url"]
        file_url: str = resp.data["file_url"]

        with open(file_path, "rb") as f:
            put_resp = httpx.put(
                upload_url,
                content=f,
                headers={"Content-Type": content_type},
                timeout=httpx.Timeout(B2_UPLOAD_TIMEOUT),
            )
            put_resp.raise_for_status()

        return file_url


class AsyncResource:
    """Base class for asynchronous service resources."""

    def __init__(self, client: BaseAsyncClient) -> None:
        self._client = client

    def _inject_metadata(self, model: T, response: APIResponse) -> T:
        return _inject_response_metadata(model, response)

    def _build_request_body(
        self,
        *,
        return_job: bool = False,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return _build_common_request_body(return_job=return_job, request_id=request_id, **kwargs)

    async def _presigned_upload(self, file_path: str | Path, filename: str) -> str:
        """Upload a large file via presigned URL, return the file_url for processing."""
        content_type = _guess_content_type(filename)

        resp = await self._client.post(
            "/api/v1/upload/presign",
            json={"filename": filename, "content_type": content_type},
        )
        upload_url: str = resp.data["upload_url"]
        file_url: str = resp.data["file_url"]

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        async with httpx.AsyncClient(timeout=httpx.Timeout(B2_UPLOAD_TIMEOUT)) as upload_client:
            put_resp = await upload_client.put(
                upload_url,
                content=file_bytes,
                headers={"Content-Type": content_type},
            )
            put_resp.raise_for_status()

        return file_url
