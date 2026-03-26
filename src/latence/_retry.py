"""Retry logic with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import random
import time
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar

import httpx

from ._constants import (
    DEFAULT_EXPONENTIAL_BASE,
    DEFAULT_INITIAL_RETRY_DELAY,
    DEFAULT_JITTER,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_RETRY_DELAY,
    RETRYABLE_STATUS_CODES,
)
from ._logging import log_retry

if TYPE_CHECKING:
    from ._base import APIResponse

T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior."""

    __slots__ = (
        "max_retries",
        "initial_delay",
        "max_delay",
        "exponential_base",
        "jitter",
        "retryable_status_codes",
    )

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_RETRY_DELAY,
        max_delay: float = DEFAULT_MAX_RETRY_DELAY,
        exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
        jitter: float = DEFAULT_JITTER,
        retryable_status_codes: frozenset[int] | None = None,
    ) -> None:
        """
        Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts. Set to 0 to disable retries.
            initial_delay: Initial delay between retries in seconds.
            max_delay: Maximum delay between retries in seconds.
            exponential_base: Base for exponential backoff calculation.
            jitter: Jitter factor (0.0-1.0) to randomize delays.
            retryable_status_codes: HTTP status codes that should trigger retries.
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_status_codes = retryable_status_codes or RETRYABLE_STATUS_CODES

    def calculate_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """
        Calculate delay before next retry attempt.

        Uses exponential backoff with jitter, but respects Retry-After header
        when provided by the server.

        Args:
            attempt: Current attempt number (0-indexed)
            retry_after: Value from Retry-After header, if present

        Returns:
            Delay in seconds before next retry
        """
        # If server specified Retry-After, respect it (but cap at max_delay)
        if retry_after is not None:
            return min(retry_after, self.max_delay)

        # Exponential backoff: initial_delay * (base ^ attempt)
        delay = self.initial_delay * (self.exponential_base ** attempt)

        # Apply jitter: delay * (1 - jitter + random * 2 * jitter)
        # This gives us delay ± jitter%
        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay = delay - jitter_range + (random.random() * 2 * jitter_range)

        # Cap at max_delay
        return min(delay, self.max_delay)

    def should_retry(
        self,
        attempt: int,
        status_code: int | None = None,
        exception: Exception | None = None,
    ) -> bool:
        """
        Determine if a request should be retried.

        Args:
            attempt: Current attempt number (0-indexed)
            status_code: HTTP status code from response, if any
            exception: Exception that was raised, if any

        Returns:
            True if the request should be retried
        """
        # Check if we've exceeded max retries
        if attempt >= self.max_retries:
            return False

        # Retry on retryable status codes
        if status_code is not None and status_code in self.retryable_status_codes:
            return True

        # Retry on connection/timeout errors
        if exception is not None:
            if isinstance(exception, (httpx.ConnectError, httpx.TimeoutException)):
                return True
            # Also retry on generic network errors
            if isinstance(exception, httpx.HTTPStatusError):
                return exception.response.status_code in self.retryable_status_codes

        return False


def is_retryable_exception(exc: Exception) -> bool:
    """Check if an exception is retryable."""
    return isinstance(exc, (httpx.ConnectError, httpx.TimeoutException))


def get_retry_after(response: httpx.Response) -> float | None:
    """Extract Retry-After value from response headers."""
    retry_after = response.headers.get("retry-after")
    if retry_after is None:
        return None

    try:
        # Retry-After can be seconds (integer) or HTTP-date
        # We only handle the seconds case for simplicity
        return float(retry_after)
    except (ValueError, TypeError):
        return None


class SyncRetryHandler:
    """Handles retry logic for synchronous requests."""

    def __init__(self, config: RetryConfig) -> None:
        self.config = config

    def execute(
        self,
        request_fn: Callable[[], httpx.Response],
    ) -> httpx.Response:
        """
        Execute a request with retry logic.

        Args:
            request_fn: Function that makes the HTTP request

        Returns:
            The successful response

        Raises:
            The last exception if all retries are exhausted
        """
        last_exception: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = request_fn()

                # Check if we should retry based on status code
                if self.config.should_retry(attempt, status_code=response.status_code):
                    retry_after = get_retry_after(response)
                    delay = self.config.calculate_delay(attempt, retry_after)
                    log_retry(attempt + 1, delay, f"status {response.status_code}")
                    time.sleep(delay)
                    continue

                return response

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exception = e

                if not self.config.should_retry(attempt, exception=e):
                    raise

                delay = self.config.calculate_delay(attempt)
                log_retry(attempt + 1, delay, str(type(e).__name__))
                time.sleep(delay)

        # If we get here, all retries were exhausted
        if last_exception is not None:
            raise last_exception

        # This shouldn't happen, but just in case
        raise RuntimeError("Retry loop exited unexpectedly")


class AsyncRetryHandler:
    """Handles retry logic for asynchronous requests."""

    def __init__(self, config: RetryConfig) -> None:
        self.config = config

    async def execute(
        self,
        request_fn: Callable[[], Awaitable[httpx.Response]],
    ) -> httpx.Response:
        """
        Execute an async request with retry logic.

        Args:
            request_fn: Async function that makes the HTTP request

        Returns:
            The successful response

        Raises:
            The last exception if all retries are exhausted
        """
        last_exception: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await request_fn()

                # Check if we should retry based on status code
                if self.config.should_retry(attempt, status_code=response.status_code):
                    retry_after = get_retry_after(response)
                    delay = self.config.calculate_delay(attempt, retry_after)
                    log_retry(attempt + 1, delay, f"status {response.status_code}")
                    await asyncio.sleep(delay)
                    continue

                return response

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exception = e

                if not self.config.should_retry(attempt, exception=e):
                    raise

                delay = self.config.calculate_delay(attempt)
                log_retry(attempt + 1, delay, str(type(e).__name__))
                await asyncio.sleep(delay)

        # If we get here, all retries were exhausted
        if last_exception is not None:
            raise last_exception

        # This shouldn't happen, but just in case
        raise RuntimeError("Retry loop exited unexpectedly")
