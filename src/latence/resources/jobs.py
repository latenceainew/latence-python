"""Jobs service resource for background job management."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator

import httpx

from .._constants import B2_FETCH_TIMEOUT, DEFAULT_POLL_INTERVAL, DEFAULT_POLL_TIMEOUT
from .._exceptions import APIError, JobError, JobTimeoutError
from .._models import (
    JobCancelResponse,
    JobListResponse,
    JobStatus,
    JobStatusResponse,
)
from ._base import AsyncResource, SyncResource

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient

logger = logging.getLogger("latence")

# Transient HTTP status codes that should be retried during polling
_TRANSIENT_STATUS_CODES = {500, 502, 503, 504}
_MAX_TRANSIENT_RETRIES = 5


class Jobs(SyncResource):
    """
    Jobs service - manage background jobs.

    Example:
        >>> job = client.document_intelligence.process(file_url="...", return_job=True)
        >>> result = client.jobs.wait(job.job_id)
        >>> print(result.output)
    """

    def __init__(self, client: BaseSyncClient) -> None:
        super().__init__(client)

    def list(
        self,
        *,
        status: JobStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> JobListResponse:
        """
        List background jobs.

        Args:
            status: Filter by status (QUEUED, IN_PROGRESS, COMPLETED, etc.)
            limit: Maximum results (max 500)
            offset: Pagination offset

        Returns:
            JobListResponse with list of jobs
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = self._client.get("/api/v1/jobs", params=params)
        result = JobListResponse.model_validate(response.data)
        return self._inject_metadata(result, response)

    def list_iter(
        self,
        *,
        status: JobStatus | None = None,
        page_size: int = 100,
    ) -> Iterator[JobStatusResponse]:
        """
        Iterate over all background jobs with automatic pagination.

        Args:
            status: Filter by status (QUEUED, IN_PROGRESS, COMPLETED, etc.)
            page_size: Number of jobs per page (max 500)

        Yields:
            JobStatusResponse for each job

        Example:
            >>> for job in client.jobs.list_iter(status="COMPLETED"):
            ...     print(f"{job.job_id}: {job.status}")
        """
        offset = 0
        while True:
            page = self.list(status=status, limit=page_size, offset=offset)
            for job in page.jobs:
                yield job

            if len(page.jobs) < page_size:
                break
            offset += page_size

    def get(self, job_id: str) -> JobStatusResponse:
        """
        Get status of a background job.

        Args:
            job_id: Job identifier

        Returns:
            JobStatusResponse with status and output (if complete)
        """
        response = self._client.get(f"/api/v1/jobs/{job_id}")
        result = JobStatusResponse.model_validate(response.data)
        return self._inject_metadata(result, response)

    def cancel(self, job_id: str) -> JobCancelResponse:
        """
        Cancel a background job.

        Args:
            job_id: Job identifier

        Returns:
            JobCancelResponse with cancellation result
        """
        response = self._client.delete(f"/api/v1/jobs/{job_id}")
        result = JobCancelResponse.model_validate(response.data)
        return self._inject_metadata(result, response)

    def wait(
        self,
        job_id: str,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        timeout: float = DEFAULT_POLL_TIMEOUT,
    ) -> JobStatusResponse:
        """
        Wait for a job to complete, polling periodically.

        Transient server errors (5xx) are retried automatically up to
        ``_MAX_TRANSIENT_RETRIES`` consecutive times before raising.

        Args:
            job_id: Job identifier
            poll_interval: Seconds between polls
            timeout: Maximum wait time in seconds

        Returns:
            JobStatusResponse with final status and output

        Raises:
            JobTimeoutError: If timeout exceeded
            JobError: If job failed
        """
        start_time = time.monotonic()
        consecutive_errors = 0

        while True:
            try:
                result = self.get(job_id)
                consecutive_errors = 0  # reset on success
            except APIError as exc:
                if (
                    getattr(exc, "status_code", None) in _TRANSIENT_STATUS_CODES
                    and consecutive_errors < _MAX_TRANSIENT_RETRIES
                ):
                    consecutive_errors += 1
                    logger.warning(
                        "Transient error polling job %s (attempt %d/%d): %s",
                        job_id,
                        consecutive_errors,
                        _MAX_TRANSIENT_RETRIES,
                        exc,
                    )
                    time.sleep(poll_interval)
                    continue
                raise

            if result.status in ("COMPLETED", "CACHED", "PULLED"):
                return result
            elif result.status == "FAILED":
                raise JobError(
                    result.error_message or "Job failed",
                    job_id=job_id,
                    error_code=result.error_code,
                )
            elif result.status == "CANCELLED":
                raise JobError("Job was cancelled", job_id=job_id)

            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                raise JobTimeoutError(
                    f"Job {job_id} did not complete within {timeout}s",
                    job_id=job_id,
                )

            time.sleep(poll_interval)

    def retrieve(self, job_id: str) -> dict[str, Any]:
        """
        Retrieve job output, fetching from B2 if necessary.

        If the job result is cached in B2 storage, this method
        automatically fetches it from the presigned URL.

        Args:
            job_id: Job identifier

        Returns:
            The job output data

        Raises:
            JobError: If job failed or output unavailable
        """
        result = self.get(job_id)

        if result.status == "FAILED":
            raise JobError(
                result.error_message or "Job failed",
                job_id=job_id,
                error_code=result.error_code,
            )

        if result.status not in ("COMPLETED", "CACHED", "PULLED"):
            raise JobError(
                f"Job not complete (status: {result.status})",
                job_id=job_id,
            )

        # If output is inline, return it directly
        if result.output is not None:
            return result.output

        # If output is in B2, fetch it
        if result.output_url:
            return self._fetch_output_url(result.output_url)

        raise JobError("No output available", job_id=job_id)

    def _fetch_output_url(self, url: str) -> dict[str, Any]:
        """Fetch job output from B2 presigned URL."""
        # Use a separate client for B2 (no auth headers)
        with httpx.Client(timeout=B2_FETCH_TIMEOUT) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()


class AsyncJobs(AsyncResource):
    """Async Jobs service."""

    def __init__(self, client: BaseAsyncClient) -> None:
        super().__init__(client)

    async def list(
        self,
        *,
        status: JobStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> JobListResponse:
        """
        List background jobs.

        Args:
            status: Filter by status (QUEUED, IN_PROGRESS, COMPLETED, etc.)
            limit: Maximum results (max 500)
            offset: Pagination offset

        Returns:
            JobListResponse with list of jobs
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = await self._client.get("/api/v1/jobs", params=params)
        result = JobListResponse.model_validate(response.data)
        return self._inject_metadata(result, response)

    async def list_iter(
        self,
        *,
        status: JobStatus | None = None,
        page_size: int = 100,
    ) -> AsyncIterator[JobStatusResponse]:
        """
        Iterate over all background jobs with automatic pagination.

        Args:
            status: Filter by status (QUEUED, IN_PROGRESS, COMPLETED, etc.)
            page_size: Number of jobs per page (max 500)

        Yields:
            JobStatusResponse for each job

        Example:
            >>> async for job in client.jobs.list_iter(status="COMPLETED"):
            ...     print(f"{job.job_id}: {job.status}")
        """
        offset = 0
        while True:
            page = await self.list(status=status, limit=page_size, offset=offset)
            for job in page.jobs:
                yield job

            if len(page.jobs) < page_size:
                break
            offset += page_size

    async def get(self, job_id: str) -> JobStatusResponse:
        """
        Get status of a background job.

        Args:
            job_id: Job identifier

        Returns:
            JobStatusResponse with status and output (if complete)
        """
        response = await self._client.get(f"/api/v1/jobs/{job_id}")
        result = JobStatusResponse.model_validate(response.data)
        return self._inject_metadata(result, response)

    async def cancel(self, job_id: str) -> JobCancelResponse:
        """
        Cancel a background job.

        Args:
            job_id: Job identifier

        Returns:
            JobCancelResponse with cancellation result
        """
        response = await self._client.delete(f"/api/v1/jobs/{job_id}")
        result = JobCancelResponse.model_validate(response.data)
        return self._inject_metadata(result, response)

    async def wait(
        self,
        job_id: str,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        timeout: float = DEFAULT_POLL_TIMEOUT,
    ) -> JobStatusResponse:
        """
        Wait for a job to complete, polling periodically.

        Transient server errors (5xx) are retried automatically up to
        ``_MAX_TRANSIENT_RETRIES`` consecutive times before raising.

        Args:
            job_id: Job identifier
            poll_interval: Seconds between polls
            timeout: Maximum wait time in seconds

        Returns:
            JobStatusResponse with final status and output

        Raises:
            JobTimeoutError: If timeout exceeded
            JobError: If job failed
        """
        start_time = time.monotonic()
        consecutive_errors = 0

        while True:
            try:
                result = await self.get(job_id)
                consecutive_errors = 0
            except APIError as exc:
                if (
                    getattr(exc, "status_code", None) in _TRANSIENT_STATUS_CODES
                    and consecutive_errors < _MAX_TRANSIENT_RETRIES
                ):
                    consecutive_errors += 1
                    logger.warning(
                        "Transient error polling job %s (attempt %d/%d): %s",
                        job_id,
                        consecutive_errors,
                        _MAX_TRANSIENT_RETRIES,
                        exc,
                    )
                    await asyncio.sleep(poll_interval)
                    continue
                raise

            if result.status in ("COMPLETED", "CACHED", "PULLED"):
                return result
            elif result.status == "FAILED":
                raise JobError(
                    result.error_message or "Job failed",
                    job_id=job_id,
                    error_code=result.error_code,
                )
            elif result.status == "CANCELLED":
                raise JobError("Job was cancelled", job_id=job_id)

            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                raise JobTimeoutError(
                    f"Job {job_id} did not complete within {timeout}s",
                    job_id=job_id,
                )

            await asyncio.sleep(poll_interval)

    async def retrieve(self, job_id: str) -> dict[str, Any]:
        """
        Retrieve job output, fetching from B2 if necessary.

        If the job result is cached in B2 storage, this method
        automatically fetches it from the presigned URL.

        Args:
            job_id: Job identifier

        Returns:
            The job output data

        Raises:
            JobError: If job failed or output unavailable
        """
        result = await self.get(job_id)

        if result.status == "FAILED":
            raise JobError(
                result.error_message or "Job failed",
                job_id=job_id,
                error_code=result.error_code,
            )

        if result.status not in ("COMPLETED", "CACHED", "PULLED"):
            raise JobError(
                f"Job not complete (status: {result.status})",
                job_id=job_id,
            )

        if result.output is not None:
            return result.output

        if result.output_url:
            return await self._fetch_output_url(result.output_url)

        raise JobError("No output available", job_id=job_id)

    async def _fetch_output_url(self, url: str) -> dict[str, Any]:
        """Fetch job output from B2 presigned URL."""
        async with httpx.AsyncClient(timeout=B2_FETCH_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
