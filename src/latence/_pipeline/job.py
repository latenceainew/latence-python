"""Job handle: thin wrapper around a pipeline job ID.

The Job object is the primary handle a user interacts with after
submitting a pipeline.  It is async/job-based by nature: ``submit()``
returns immediately, and the user can check status or wait at their
convenience.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .._exceptions import JobError, JobTimeoutError
from .._models.pipeline import (
    PipelineReport,
    PipelineStatusResponse,
    StageDownload,
)
from .data_package import DataPackage

if TYPE_CHECKING:
    from ..resources.pipeline import AsyncPipeline, Pipeline


# Defaults matching the pipeline resource
_DEFAULT_POLL_INTERVAL = 5.0
_DEFAULT_POLL_TIMEOUT = 1800.0  # 30 minutes


class Job:
    """Synchronous handle for a pipeline job.

    Returned by ``client.pipeline.run()`` and ``client.pipeline.submit()``.
    Provides methods to poll status, wait for completion, cancel, and
    retrieve results as a composed :class:`DataPackage`.

    Example::

        job = client.pipeline.run(files=["doc.pdf"])
        print(f"Submitted: {job.id}")

        # Check status
        status = job.status()
        print(f"Stage {status.stages_completed}/{status.total_stages}")

        # Wait for the composed data package
        pkg = job.wait_for_completion()
        print(pkg.document.markdown)
    """

    __slots__ = ("_id", "_name", "_services", "_pipeline", "_cached_package")

    def __init__(
        self,
        job_id: str,
        pipeline: Pipeline,
        *,
        name: str | None = None,
        services: list[str] | None = None,
    ) -> None:
        self._id = job_id
        self._pipeline = pipeline
        self._name = name
        self._services = services
        self._cached_package: DataPackage | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        """The pipeline job ID (e.g. ``pipe_xxx``)."""
        return self._id

    @property
    def name(self) -> str | None:
        """Human-readable pipeline name, if provided at submission."""
        return self._name

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> PipelineStatusResponse:
        """Poll the current status of this job.

        Returns per-stage status, the current stage being processed,
        and progress counts.

        Returns:
            PipelineStatusResponse with job status details.
        """
        return self._pipeline.status(self._id)

    # ------------------------------------------------------------------
    # Wait for completion
    # ------------------------------------------------------------------

    def wait_for_completion(
        self,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_POLL_TIMEOUT,
        save_to_disk: str | Path | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> DataPackage:
        """Block until the pipeline completes and return a composed DataPackage.

        Polls the job status at ``poll_interval`` and, once complete,
        fetches the full result and composes it into a structured
        :class:`DataPackage`.

        Args:
            poll_interval: Seconds between status polls (default 5.0).
            timeout: Maximum wait time in seconds (default 1800 = 30 min).
            save_to_disk: If provided, automatically save the result as a
                ZIP archive at this path. Equivalent to calling
                ``pkg.download_archive(save_to_disk)`` after completion.
            on_progress: Optional callback invoked on each poll with
                ``(status: str, elapsed_seconds: float)``.

        Returns:
            Composed DataPackage with organized sections and quality report.

        Raises:
            JobTimeoutError: If the timeout is exceeded.
            JobError: If the pipeline fails or is cancelled.
        """
        start = time.monotonic()

        while True:
            current = self.status()
            elapsed = time.monotonic() - start

            if on_progress is not None:
                on_progress(current.status, elapsed)

            if current.status in ("COMPLETED", "CACHED", "PULLED"):
                raw_result = self._pipeline.retrieve(self._id)
                pkg = DataPackage.from_pipeline_result(
                    raw_result,
                    name=self._name,
                    services=self._services,
                )
                self._cached_package = pkg
                if save_to_disk is not None:
                    pkg.download_archive(save_to_disk)
                return pkg

            if current.status == "FAILED":
                raise JobError(
                    current.error_message or "Pipeline failed",
                    job_id=self._id,
                    error_code=current.error_code,
                    is_resumable=current.is_resumable,
                )

            if current.status == "RESUMABLE":
                msg = current.error_message or "Pipeline failed"
                if current.failed_stage:
                    msg = f"{msg} at stage '{current.failed_stage}'"
                msg += ". Call job.resume() to continue from the last checkpoint."
                raise JobError(
                    msg,
                    job_id=self._id,
                    error_code=current.error_code,
                    is_resumable=True,
                )

            if current.status == "CANCELLED":
                raise JobError("Pipeline was cancelled", job_id=self._id)

            if elapsed >= timeout:
                raise JobTimeoutError(
                    f"Pipeline {self._id} did not complete within {timeout}s",
                    job_id=self._id,
                )

            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Cancel / Resume
    # ------------------------------------------------------------------

    def cancel(self) -> dict[str, Any]:
        """Cancel this pipeline job.

        Returns:
            Cancellation response from the API.
        """
        return self._pipeline.cancel(self._id)  # type: ignore[return-value]

    def resume(self) -> "Job":
        """Resume this pipeline from its last checkpoint.

        The pipeline must be in RESUMABLE or FAILED status with at least
        one completed stage.  Completed stages are skipped; only remaining
        stages are re-executed.

        Returns:
            ``self`` for chaining (e.g. ``job.resume().wait_for_completion()``).

        Raises:
            LatenceError: If the pipeline cannot be resumed.
        """
        self._pipeline.resume(self._id)
        return self

    # ------------------------------------------------------------------
    # Intermediate results / Report
    # ------------------------------------------------------------------

    def intermediate_results(self) -> list[StageDownload]:
        """Get per-stage download URLs for completed stages.

        Each :class:`StageDownload` contains a presigned URL to the
        stage's ``results.jsonl`` file in B2.

        Returns:
            List of stage downloads with presigned URLs.
        """
        return self._pipeline.stages(self._id)

    @property
    def report(self) -> PipelineReport | None:
        """Structured pipeline report with dataset facts, per-stage metrics.

        Returns ``None`` if the pipeline hasn't generated a report yet
        (still running or no stages completed).
        """
        st = self._pipeline.status(self._id)
        if st.pipeline_report:
            return PipelineReport.model_validate(st.pipeline_report)
        return None

    # ------------------------------------------------------------------
    # Data package (lazy)
    # ------------------------------------------------------------------

    @property
    def data_package(self) -> DataPackage:
        """Fetch and compose the DataPackage (cached after first access).

        Retrieves the pipeline result and composes it into a structured
        DataPackage.  The result is cached: subsequent accesses return
        the same object without an API call.

        Returns:
            Composed DataPackage.

        Raises:
            JobError: If the pipeline has not completed or failed.
        """
        if self._cached_package is not None:
            return self._cached_package

        raw_result = self._pipeline.retrieve(self._id)
        pkg = DataPackage.from_pipeline_result(
            raw_result,
            name=self._name,
            services=self._services,
        )
        self._cached_package = pkg
        return pkg

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        name_part = f", name={self._name!r}" if self._name else ""
        return f"Job(id={self._id!r}{name_part})"


class AsyncJob:
    """Asynchronous handle for a pipeline job.

    Returned by ``client.pipeline.run()`` and ``client.pipeline.submit()``
    on the :class:`AsyncLatence` client.

    Example::

        job = await client.pipeline.run(files=["doc.pdf"])
        print(f"Submitted: {job.id}")

        pkg = await job.wait_for_completion()
        print(pkg.entities.summary)
    """

    __slots__ = ("_id", "_name", "_services", "_pipeline", "_cached_package")

    def __init__(
        self,
        job_id: str,
        pipeline: AsyncPipeline,
        *,
        name: str | None = None,
        services: list[str] | None = None,
    ) -> None:
        self._id = job_id
        self._pipeline = pipeline
        self._name = name
        self._services = services
        self._cached_package: DataPackage | None = None

    @property
    def id(self) -> str:
        """The pipeline job ID."""
        return self._id

    @property
    def name(self) -> str | None:
        """Pipeline name, if provided."""
        return self._name

    async def status(self) -> PipelineStatusResponse:
        """Poll the current status of this job."""
        return await self._pipeline.status(self._id)

    async def wait_for_completion(
        self,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_POLL_TIMEOUT,
        save_to_disk: str | Path | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> DataPackage:
        """Wait until the pipeline completes and return a composed DataPackage.

        Args:
            poll_interval: Seconds between status polls (default 5.0).
            timeout: Maximum wait time in seconds (default 1800 = 30 min).
            save_to_disk: If provided, automatically save the result as a
                ZIP archive at this path.
            on_progress: Optional callback invoked on each poll with
                ``(status: str, elapsed_seconds: float)``.

        Returns:
            Composed DataPackage.

        Raises:
            JobTimeoutError: If the timeout is exceeded.
            JobError: If the pipeline fails or is cancelled.
        """
        start = time.monotonic()

        while True:
            current = await self.status()
            elapsed = time.monotonic() - start

            if on_progress is not None:
                on_progress(current.status, elapsed)

            if current.status in ("COMPLETED", "CACHED", "PULLED"):
                raw_result = await self._pipeline.retrieve(self._id)
                pkg = DataPackage.from_pipeline_result(
                    raw_result,
                    name=self._name,
                    services=self._services,
                )
                self._cached_package = pkg
                if save_to_disk is not None:
                    pkg.download_archive(save_to_disk)
                return pkg

            if current.status == "FAILED":
                raise JobError(
                    current.error_message or "Pipeline failed",
                    job_id=self._id,
                    error_code=current.error_code,
                    is_resumable=current.is_resumable,
                )

            if current.status == "RESUMABLE":
                msg = current.error_message or "Pipeline failed"
                if current.failed_stage:
                    msg = f"{msg} at stage '{current.failed_stage}'"
                msg += ". Call job.resume() to continue from the last checkpoint."
                raise JobError(
                    msg,
                    job_id=self._id,
                    error_code=current.error_code,
                    is_resumable=True,
                )

            if current.status == "CANCELLED":
                raise JobError("Pipeline was cancelled", job_id=self._id)

            if elapsed >= timeout:
                raise JobTimeoutError(
                    f"Pipeline {self._id} did not complete within {timeout}s",
                    job_id=self._id,
                )

            await asyncio.sleep(poll_interval)

    async def cancel(self) -> dict[str, Any]:
        """Cancel this pipeline job."""
        return await self._pipeline.cancel(self._id)  # type: ignore[return-value]

    async def resume(self) -> "AsyncJob":
        """Resume this pipeline from its last checkpoint.

        Returns ``self`` for chaining.

        Raises:
            LatenceError: If the pipeline cannot be resumed.
        """
        await self._pipeline.resume(self._id)
        return self

    async def intermediate_results(self) -> list[StageDownload]:
        """Get per-stage download URLs for completed stages."""
        return await self._pipeline.stages(self._id)

    async def get_report(self) -> PipelineReport | None:
        """Fetch the structured pipeline report with dataset facts and per-stage metrics.

        Returns ``None`` if the pipeline hasn't generated a report yet.
        """
        st = await self._pipeline.status(self._id)
        if st.pipeline_report:
            return PipelineReport.model_validate(st.pipeline_report)
        return None

    @property
    def data_package(self) -> DataPackage:
        """Return cached DataPackage (only available after wait_for_completion)."""
        if self._cached_package is not None:
            return self._cached_package
        raise JobError(
            "DataPackage not available. Call await job.wait_for_completion() first.",
            job_id=self._id,
        )

    def __repr__(self) -> str:
        name_part = f", name={self._name!r}" if self._name else ""
        return f"AsyncJob(id={self._id!r}{name_part})"
