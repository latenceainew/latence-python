"""Exception hierarchy for the Latence SDK."""

from __future__ import annotations

from typing import Any


class LatenceError(Exception):
    """Base exception for all Latence SDK errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class APIError(LatenceError):
    """
    Base class for API errors with HTTP status codes.

    Attributes:
        status_code: HTTP status code
        error_code: API error code (e.g., "INVALID_KEY")
        request_id: Request ID for debugging
        body: Raw response body
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_code: str | None = None,
        request_id: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.request_id = request_id
        self.body = body
        super().__init__(message)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"status_code={self.status_code}, "
            f"error_code={self.error_code!r}, "
            f"request_id={self.request_id!r})"
        )


class AuthenticationError(APIError):
    """
    Raised when authentication fails (401).

    Causes:
    - Missing API key
    - Invalid or expired API key
    """

    def __init__(
        self,
        message: str = "Authentication failed",
        *,
        error_code: str | None = None,
        request_id: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=401,
            error_code=error_code,
            request_id=request_id,
            body=body,
        )


class InsufficientCreditsError(APIError):
    """
    Raised when the account has insufficient credits (402).

    The API key is valid but the account balance is zero.
    """

    def __init__(
        self,
        message: str = "Insufficient credits",
        *,
        error_code: str | None = None,
        request_id: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=402,
            error_code=error_code,
            request_id=request_id,
            body=body,
        )


class NotFoundError(APIError):
    """
    Raised when a resource is not found (404).

    Causes:
    - Invalid endpoint path
    - Job ID not found
    """

    def __init__(
        self,
        message: str = "Resource not found",
        *,
        error_code: str | None = None,
        request_id: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=404,
            error_code=error_code,
            request_id=request_id,
            body=body,
        )


class ValidationError(APIError):
    """
    Raised when request validation fails (400).

    Causes:
    - Invalid JSON
    - Missing required fields
    - Invalid field values
    """

    def __init__(
        self,
        message: str = "Validation error",
        *,
        error_code: str | None = None,
        request_id: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=400,
            error_code=error_code,
            request_id=request_id,
            body=body,
        )


class RateLimitError(APIError):
    """
    Raised when rate limit is exceeded (429).

    Attributes:
        retry_after: Seconds to wait before retrying
    """

    retry_after: float | None

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        error_code: str | None = None,
        request_id: str | None = None,
        body: dict[str, Any] | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            message,
            status_code=429,
            error_code=error_code,
            request_id=request_id,
            body=body,
        )


class ServerError(APIError):
    """
    Raised for server-side errors (5xx).

    Includes:
    - 500 Internal Server Error
    - 502 Bad Gateway
    - 503 Service Unavailable
    - 504 Timeout
    """

    def __init__(
        self,
        message: str = "Server error",
        *,
        status_code: int = 500,
        error_code: str | None = None,
        request_id: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            error_code=error_code,
            request_id=request_id,
            body=body,
        )


class TransportError(LatenceError):
    """Base for network-level errors (no HTTP status code).

    Use ``except TransportError`` to catch all connection and timeout errors.
    Use ``except LatenceError`` to catch everything (HTTP + transport + job).
    """

    def __init__(self, message: str = "Transport error") -> None:
        super().__init__(message)


class APIConnectionError(TransportError):
    """
    Raised when connection to the API fails.

    Causes:
    - Network connectivity issues
    - DNS resolution failures
    - Connection refused
    """

    def __init__(self, message: str = "Connection error") -> None:
        super().__init__(message)


class APITimeoutError(TransportError):
    """
    Raised when a request times out.

    Causes:
    - Request took longer than the configured timeout
    """

    def __init__(self, message: str = "Request timed out") -> None:
        super().__init__(message)


class JobError(LatenceError):
    """
    Raised when a background job fails.

    Attributes:
        job_id: The job ID that failed
        error_code: Error code from the job
        is_resumable: Whether the pipeline can be resumed from a checkpoint
    """

    def __init__(
        self,
        message: str,
        *,
        job_id: str,
        error_code: str | None = None,
        is_resumable: bool = False,
    ) -> None:
        self.job_id = job_id
        self.error_code = error_code
        self.is_resumable = is_resumable
        super().__init__(message)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"job_id={self.job_id!r}, "
            f"error_code={self.error_code!r}, "
            f"is_resumable={self.is_resumable})"
        )


class JobTimeoutError(JobError):
    """Raised when waiting for a job exceeds the timeout."""

    def __init__(self, message: str, *, job_id: str) -> None:
        super().__init__(message, job_id=job_id)

    def __repr__(self) -> str:
        return f"JobTimeoutError(message={self.message!r}, job_id={self.job_id!r})"


def _create_api_error(
    status_code: int,
    message: str,
    error_code: str | None = None,
    request_id: str | None = None,
    body: dict[str, Any] | None = None,
    retry_after: float | None = None,
) -> APIError:
    """Factory function to create the appropriate APIError subclass."""
    if status_code == 400:
        return ValidationError(message, error_code=error_code, request_id=request_id, body=body)
    elif status_code == 401:
        return AuthenticationError(message, error_code=error_code, request_id=request_id, body=body)
    elif status_code == 402:
        return InsufficientCreditsError(
            message, error_code=error_code, request_id=request_id, body=body
        )
    elif status_code == 404:
        return NotFoundError(message, error_code=error_code, request_id=request_id, body=body)
    elif status_code == 429:
        return RateLimitError(
            message,
            error_code=error_code,
            request_id=request_id,
            body=body,
            retry_after=retry_after,
        )
    elif status_code >= 500:
        return ServerError(
            message,
            status_code=status_code,
            error_code=error_code,
            request_id=request_id,
            body=body,
        )
    else:
        return APIError(
            message,
            status_code=status_code,
            error_code=error_code,
            request_id=request_id,
            body=body,
        )
