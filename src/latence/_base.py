"""Base HTTP client implementations for sync and async operations."""

from __future__ import annotations

import os
from typing import Any, TypeVar

import httpx

from ._constants import DEFAULT_BASE_URL, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
from ._exceptions import (
    APIConnectionError,
    APITimeoutError,
    _create_api_error,
)
from ._logging import log_request, log_response
from ._retry import AsyncRetryHandler, RetryConfig, SyncRetryHandler
from ._version import VERSION

T = TypeVar("T")


class ResponseMetadata:
    """Metadata extracted from HTTP response headers."""

    __slots__ = (
        "credits_used",
        "credits_remaining",
        "rate_limit_limit",
        "rate_limit_remaining",
        "retry_after",
        "request_id",
    )

    def __init__(
        self,
        *,
        credits_used: float | None = None,
        credits_remaining: float | None = None,
        rate_limit_limit: int | None = None,
        rate_limit_remaining: int | None = None,
        retry_after: float | None = None,
        request_id: str | None = None,
    ) -> None:
        self.credits_used = credits_used
        self.credits_remaining = credits_remaining
        self.rate_limit_limit = rate_limit_limit
        self.rate_limit_remaining = rate_limit_remaining
        self.retry_after = retry_after
        self.request_id = request_id

    @classmethod
    def from_headers(cls, headers: httpx.Headers) -> ResponseMetadata:
        """Extract metadata from response headers."""

        def _float_or_none(val: str | None) -> float | None:
            if val is None:
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        def _int_or_none(val: str | None) -> int | None:
            if val is None:
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        return cls(
            credits_used=_float_or_none(headers.get("x-credits-used")),
            credits_remaining=_float_or_none(headers.get("x-credits-remaining")),
            rate_limit_limit=_int_or_none(headers.get("x-ratelimit-limit")),
            rate_limit_remaining=_int_or_none(headers.get("x-ratelimit-remaining")),
            retry_after=_float_or_none(headers.get("retry-after")),
            request_id=headers.get("x-request-id"),
        )


class APIResponse:
    """Container for API response data and metadata."""

    __slots__ = ("data", "metadata", "status_code")

    def __init__(
        self,
        data: dict[str, Any],
        metadata: ResponseMetadata,
        status_code: int,
    ) -> None:
        self.data = data
        self.metadata = metadata
        self.status_code = status_code


def _parse_api_response(response: httpx.Response) -> APIResponse:
    """Process HTTP response — shared between sync and async clients.

    Raises:
        APIError subclass: When the response indicates an error.
    """
    metadata = ResponseMetadata.from_headers(response.headers)

    try:
        data = response.json()
    except Exception:
        data = {"raw": response.text}

    is_success_response = response.is_success and (
        not isinstance(data, dict) or data.get("success", True) is not False
    )

    if is_success_response:
        return APIResponse(data, metadata, response.status_code)

    error_code = data.get("error") if isinstance(data, dict) else None
    message = (
        (data.get("message") or data.get("error") or response.reason_phrase or "Unknown error")
        if isinstance(data, dict)
        else str(data)
    )
    request_id = data.get("request_id") if isinstance(data, dict) else None

    raise _create_api_error(
        status_code=response.status_code,
        message=message,
        error_code=error_code,
        request_id=request_id or metadata.request_id,
        body=data if isinstance(data, dict) else None,
        retry_after=metadata.retry_after,
    )


class BaseSyncClient:
    """
    Synchronous HTTP client for the Latence API.

    Uses httpx.Client with HTTP/2 support and automatic retries.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._api_key = api_key or os.environ.get("LATENCE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "API key is required. Pass api_key or set LATENCE_API_KEY environment variable."
            )

        self.base_url = (base_url or os.environ.get("LATENCE_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.timeout = timeout
        self.max_retries = max_retries

        self._retry_config = RetryConfig(max_retries=max_retries)
        self._retry_handler = SyncRetryHandler(self._retry_config)

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            http2=True,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": f"latence-python/{VERSION}",
            },
        )

    @property
    def api_key(self) -> str:
        """The API key (read-only)."""
        return self._api_key

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> BaseSyncClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        key = self._api_key
        masked = f"{key[:7]}...{key[-3:]}" if key and len(key) > 10 else "***"
        return f"BaseSyncClient(base_url={self.base_url!r}, api_key={masked!r})"

    def _handle_response(self, response: httpx.Response) -> APIResponse:
        """Process HTTP response and raise appropriate errors."""
        return _parse_api_response(response)

    def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> APIResponse:
        """Make an HTTP request to the API with automatic retries."""
        log_request(method, path)

        def make_request() -> httpx.Response:
            return self._client.request(
                method=method,
                url=path,
                json=json,
                params=params,
                headers=headers,
            )

        try:
            if self.max_retries > 0:
                response = self._retry_handler.execute(make_request)
            else:
                response = make_request()

            result = self._handle_response(response)
            log_response(response.status_code, path, result.metadata.credits_used)
            return result
        except httpx.TimeoutException as e:
            raise APITimeoutError(f"Request timed out: {e}") from e
        except httpx.TransportError as e:
            raise APIConnectionError(f"Connection error: {e}") from e

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> APIResponse:
        """Make a GET request."""
        return self.request("GET", path, params=params)

    def post(
        self,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """Make a POST request."""
        extra_headers = {"Content-Type": "application/json"} if json is not None else None
        return self.request("POST", path, json=json, params=params, headers=extra_headers)

    def delete(self, path: str, *, params: dict[str, Any] | None = None) -> APIResponse:
        """Make a DELETE request."""
        return self.request("DELETE", path, params=params)


class BaseAsyncClient:
    """
    Asynchronous HTTP client for the Latence API.

    Uses httpx.AsyncClient with HTTP/2 support and automatic retries.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._api_key = api_key or os.environ.get("LATENCE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "API key is required. Pass api_key or set LATENCE_API_KEY environment variable."
            )

        self.base_url = (base_url or os.environ.get("LATENCE_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.timeout = timeout
        self.max_retries = max_retries

        self._retry_config = RetryConfig(max_retries=max_retries)
        self._retry_handler = AsyncRetryHandler(self._retry_config)

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            http2=True,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": f"latence-python/{VERSION}",
            },
        )

    @property
    def api_key(self) -> str:
        """The API key (read-only)."""
        return self._api_key

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> BaseAsyncClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        key = self._api_key
        masked = f"{key[:7]}...{key[-3:]}" if key and len(key) > 10 else "***"
        return f"BaseAsyncClient(base_url={self.base_url!r}, api_key={masked!r})"

    def _handle_response(self, response: httpx.Response) -> APIResponse:
        """Process HTTP response and raise appropriate errors."""
        return _parse_api_response(response)

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> APIResponse:
        """Make an async HTTP request to the API with automatic retries."""
        log_request(method, path)

        async def make_request() -> httpx.Response:
            return await self._client.request(
                method=method,
                url=path,
                json=json,
                params=params,
                headers=headers,
            )

        try:
            if self.max_retries > 0:
                response = await self._retry_handler.execute(make_request)
            else:
                response = await make_request()

            result = self._handle_response(response)
            log_response(response.status_code, path, result.metadata.credits_used)
            return result
        except httpx.TimeoutException as e:
            raise APITimeoutError(f"Request timed out: {e}") from e
        except httpx.TransportError as e:
            raise APIConnectionError(f"Connection error: {e}") from e

    async def get(self, path: str, *, params: dict[str, Any] | None = None) -> APIResponse:
        """Make a GET request."""
        return await self.request("GET", path, params=params)

    async def post(
        self,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """Make a POST request."""
        extra_headers = {"Content-Type": "application/json"} if json is not None else None
        return await self.request("POST", path, json=json, params=params, headers=extra_headers)

    async def delete(self, path: str, *, params: dict[str, Any] | None = None) -> APIResponse:
        """Make a DELETE request."""
        return await self.request("DELETE", path, params=params)
