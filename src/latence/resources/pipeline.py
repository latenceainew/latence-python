"""Pipeline service resource for multi-service pipeline execution.

This is the primary interface for the Latence AI Data Intelligence Pipeline.
Pipelines are async/job-based by default: ``run()`` and ``submit()`` return
a :class:`Job` handle immediately while data flows from service to service
on the backend.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Callable

import httpx

from .._constants import (
    B2_PIPELINE_FETCH_TIMEOUT,
    B2_UPLOAD_TIMEOUT,
    PRESIGNED_UPLOAD_THRESHOLD,
)
from .._exceptions import APIError, JobError, JobTimeoutError
from .._models.jobs import JobCancelResponse
from .._models.pipeline import (
    FileInput,
    PipelineConfig,
    PipelineInput,
    PipelineResultResponse,
    PipelineStatusResponse,
    PipelineSubmitResponse,
    PipelineValidationResult,
    StageDownload,
)
from .._pipeline.builder import PipelineBuilder
from .._pipeline.job import AsyncJob, Job
from .._pipeline.spec import build_pipeline_config, has_file_input, parse_input
from .._pipeline.validator import PipelineValidationError, validate_pipeline
from .._utils import file_to_base64
from ._base import AsyncResource, SyncResource, _guess_content_type

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient


# Pipeline-specific poll configuration (longer defaults for multi-stage pipelines)
PIPELINE_POLL_INTERVAL = 5.0  # 5 seconds between polls
PIPELINE_POLL_TIMEOUT = 1800.0  # 30 minutes max wait

_log = logging.getLogger(__name__)

_ZIP_MAGIC = b"PK\x03\x04"
_TRANSIENT_STATUS_CODES = frozenset({500, 502, 503, 504})
_MAX_TRANSIENT_RETRIES = 5


def _is_zip(response: httpx.Response, raw: bytes) -> bool:
    """Detect whether a B2 response contains a ZIP archive."""
    ct = response.headers.get("content-type", "")
    if "zip" in ct or "octet-stream" in ct:
        return True
    return raw[:4] == _ZIP_MAGIC


def _parse_pipeline_zip(raw: bytes) -> dict[str, Any]:
    """Parse a pipeline output ZIP into intermediate_results + final_output.

    ZIP layout produced by the pipeline worker::

        result.json                         -- manifest
        stage_01_document_intelligence/
            document_001.json
            _stage_meta.json
        stage_02_extraction/
            document_001.json
            _stage_meta.json
        ...

    Returns a dict shaped for ``PipelineResultResponse`` consumption::

        {
            "final_output": <last stage's aggregated output>,
            "intermediate_results": {
                "document_intelligence": {"service": ..., "output": ..., ...},
                "extraction": {"service": ..., "output": ..., ...},
                ...
            },
            "_zip_manifest": <result.json contents>,
            "_zip_documents": {
                "document_intelligence": [<per-file outputs>],
                ...
            },
        }
    """
    buf = io.BytesIO(raw)
    intermediate: dict[str, Any] = {}
    per_doc: dict[str, list[dict[str, Any]]] = {}
    manifest: dict[str, Any] = {}
    stage_meta: dict[str, dict[str, Any]] = {}

    with zipfile.ZipFile(buf, "r") as zf:
        names = zf.namelist()

        # Read manifest first
        for name in names:
            if name.endswith("result.json") and "/" not in name.rstrip("/"):
                manifest = json.loads(zf.read(name))
                break

        stage_folders: dict[str, str] = manifest.get("stage_folders", {})
        folder_to_service: dict[str, str] = {v: k for k, v in stage_folders.items()}

        for name in names:
            parts = name.rstrip("/").split("/")
            if len(parts) < 2:
                continue

            folder = parts[0]
            filename = parts[-1]

            if folder == "upload":
                continue

            service = folder_to_service.get(folder)
            if not service:
                for svc, sf in stage_folders.items():
                    if folder == sf or folder.endswith(sf):
                        service = svc
                        folder_to_service[folder] = svc
                        break
            if not service:
                continue

            if not (filename.endswith(".json") or filename.endswith(".jsonl")):
                continue

            raw_bytes = zf.read(name)

            # JSONL files: one JSON object per line (pipeline worker format)
            if filename.endswith(".jsonl"):
                for line in raw_bytes.decode("utf-8").strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        per_doc.setdefault(service, []).append(json.loads(line))
                    except json.JSONDecodeError:
                        _log.warning("Skipping malformed JSONL line in %s", name)
                continue

            try:
                data = json.loads(raw_bytes)
            except (json.JSONDecodeError, KeyError):
                _log.warning("Skipping malformed JSON in ZIP: %s", name)
                continue

            if filename == "_stage_meta.json":
                stage_meta[service] = data
                continue

            per_doc.setdefault(service, []).append(data)

    # Build intermediate_results in the format DataPackage expects
    ordered_services = manifest.get("stages", list(per_doc.keys()))
    last_service = ordered_services[-1] if ordered_services else None

    for service in ordered_services:
        docs = per_doc.get(service, [])
        meta = stage_meta.get(service, {})

        if len(docs) == 1:
            output = docs[0].get("output", docs[0])
        elif len(docs) > 1:
            outputs = [d.get("output", d) for d in docs]
            output = {"documents": outputs}
        else:
            output = {}

        intermediate[service] = {
            "service": service,
            "status": "completed",
            "output": output,
            "credits_used": meta.get("cost_usd"),
            "processing_time_ms": meta.get("duration_ms"),
        }

    final_output = None
    if last_service and last_service in intermediate:
        final_output = intermediate[last_service].get("output")

    result: dict[str, Any] = {
        "final_output": final_output,
        "intermediate_results": intermediate,
    }

    # Attach raw per-document data and manifest for merge() utility
    result["_zip_manifest"] = manifest
    result["_zip_documents"] = {
        svc: [d.get("output", d) for d in docs] for svc, docs in per_doc.items()
    }

    return result


class Pipeline(SyncResource):
    """
    Pipeline service -- the primary interface for Latence AI.

    All pipelines are async/job-based by default.  ``run()`` and
    ``submit()`` return a :class:`Job` handle immediately.  Data flows
    from service to service on the backend; the user never waits between
    stages.

    **Recommended entry points:**

    - ``run()``   -- dict-based config *or* smart defaults (simplest)
    - ``submit()`` -- fluent builder config

    **Legacy entry points** (still fully supported):

    - ``execute()`` / ``wait()`` / ``status()`` / ``retrieve()`` / ``cancel()``

    Example -- simplest usage (smart defaults)::

        job = client.pipeline.run(files=["contract.pdf"])
        pkg = job.wait_for_completion()
        print(pkg.document.markdown)
        print(pkg.entities.summary)

    Example -- explicit steps::

        job = client.pipeline.run(
            files=["contract.pdf"],
            steps={"ocr": {"mode": "performance"}, "extraction": {}, "knowledge_graph": {}},
            name="Legal Contracts",
        )
        pkg = job.wait_for_completion()

    Example -- fluent builder::

        from latence import PipelineBuilder
        config = PipelineBuilder().doc_intel().extraction().ontology().build()
        job = client.pipeline.submit(config, files=["contract.pdf"])
        pkg = job.wait_for_completion()
    """

    def __init__(self, client: BaseSyncClient) -> None:
        super().__init__(client)

    # =================================================================
    # Primary entry points (v0.2+)
    # =================================================================

    def run(
        self,
        *,
        files: list[str | Path | BinaryIO] | str | Path | None = None,
        file_urls: list[str] | None = None,
        text: str | None = None,
        entities: list[dict[str, Any]] | None = None,
        steps: dict[str, dict[str, Any]] | None = None,
        name: str | None = None,
        request_id: str | None = None,
    ) -> Job:
        """Submit a Data Intelligence Pipeline and return a Job handle.

        This is the primary, recommended entry point.  It returns
        immediately with a :class:`Job` handle.  Data flows from service
        to service on the backend asynchronously.

        If ``steps`` is ``None`` and the input includes files, the smart
        default pipeline is applied automatically:
        **OCR -> Entity Extraction -> Knowledge Graph**.

        Args:
            files: Local files (paths or binary streams) or a single path.
            file_urls: Remote file URLs to process.
            text: Raw text input.
            entities: Pre-extracted entities (for ontology-only pipelines).
            steps: Pipeline step configuration dict.  Keys are step names
                (e.g. ``"ocr"``, ``"extraction"``, ``"knowledge_graph"``).
                Values are config dicts for each step.  If ``None``,
                smart defaults apply.
            name: Human-readable pipeline name for tracking.
            request_id: Optional request tracking ID.

        Returns:
            :class:`Job` handle with ``.wait_for_completion()``,
            ``.status()``, and ``.cancel()`` methods.

        Raises:
            ValueError: If no input is provided.
            NotImplementedError: If a placeholder step is requested.

        Example::

            job = client.pipeline.run(files=["doc.pdf"])
            pkg = job.wait_for_completion()
        """
        # Build PipelineConfig from steps (or smart defaults)
        is_files = has_file_input(files=files, file_urls=file_urls)
        config = build_pipeline_config(steps=steps, name=name, has_files=is_files)

        # Build input
        pipeline_input = parse_input(files=files, file_urls=file_urls, text=text, entities=entities)
        if pipeline_input is None:
            raise ValueError(
                "At least one input must be provided: files, file_urls, text, or entities"
            )

        # Validate + execute via existing execute()
        result = self.execute(
            config,
            files=None,  # Already encoded in pipeline_input
            text=None,
            entities=None,
            request_id=request_id,
            _pipeline_input_override=pipeline_input,
        )

        return Job(
            job_id=result.job_id,
            pipeline=self,
            name=name,
            services=result.services,
        )

    def submit(
        self,
        pipeline: PipelineConfig | PipelineBuilder,
        *,
        files: list[str | Path | BinaryIO] | None = None,
        file_urls: list[str] | None = None,
        text: str | None = None,
        entities: list[dict[str, Any]] | None = None,
        name: str | None = None,
        request_id: str | None = None,
    ) -> Job:
        """Submit a pipeline from a builder or config and return a Job handle.

        Like ``run()``, this returns immediately.  Use this when you have
        a :class:`PipelineBuilder` or :class:`PipelineConfig` already
        constructed.

        Args:
            pipeline: PipelineConfig or PipelineBuilder to execute.
            files: Local files to process.
            file_urls: Remote file URLs.
            text: Raw text input.
            entities: Pre-extracted entities.
            name: Pipeline name for tracking.
            request_id: Optional tracking ID.

        Returns:
            :class:`Job` handle.

        Example::

            from latence import PipelineBuilder
            config = PipelineBuilder().doc_intel().extraction().ontology().build()
            job = client.pipeline.submit(config, files=["doc.pdf"])
            pkg = job.wait_for_completion()
        """
        # Inject name into config if not already set
        built_config = pipeline.build() if isinstance(pipeline, PipelineBuilder) else pipeline
        if name and not built_config.name:
            built_config.name = name

        result = self.execute(
            built_config,
            files=files,
            file_urls=file_urls,
            text=text,
            entities=entities,
            request_id=request_id,
        )

        return Job(
            job_id=result.job_id,
            pipeline=self,
            name=name or built_config.name,
            services=result.services,
        )

    # =================================================================
    # Legacy entry points (still fully supported)
    # =================================================================

    def validate(
        self,
        pipeline: PipelineConfig | PipelineBuilder,
        *,
        files: list[str | Path | BinaryIO] | None = None,
        file_urls: list[str] | None = None,
        text: str | None = None,
        entities: list[dict[str, Any]] | None = None,
    ) -> PipelineValidationResult:
        """
        Validate a pipeline configuration without executing it.

        Args:
            pipeline: PipelineConfig or PipelineBuilder to validate
            files: Local files to process (optional, for validation context)
            file_urls: URLs of files to process (optional, for validation context)
            text: Text input (optional, for validation context)
            entities: Pre-extracted entities (optional, for validation context)

        Returns:
            PipelineValidationResult with validation details

        Raises:
            PipelineValidationError: If strict_mode=True and validation fails
        """
        # Convert builder to config if needed
        config = pipeline.build() if isinstance(pipeline, PipelineBuilder) else pipeline

        # Build pipeline input for validation context
        pipeline_input = self._build_pipeline_input(
            files=files,
            file_urls=file_urls,
            text=text,
            entities=entities,
        )

        return validate_pipeline(config, pipeline_input)

    def execute(
        self,
        pipeline: PipelineConfig | PipelineBuilder,
        *,
        files: list[str | Path | BinaryIO] | None = None,
        file_urls: list[str] | None = None,
        text: str | None = None,
        entities: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        _pipeline_input_override: PipelineInput | None = None,
    ) -> PipelineSubmitResponse:
        """
        Submit a pipeline for async execution.

        Pipelines are always executed asynchronously. Use `wait()` or `status()`
        to poll for completion.

        Args:
            pipeline: PipelineConfig or PipelineBuilder to execute
            files: Local files to process (converted to base64)
            file_urls: URLs of files to process
            text: Text input
            entities: Pre-extracted entities (for ontology-only pipelines)
            request_id: Optional tracking ID

        Returns:
            PipelineSubmitResponse with job_id for polling

        Raises:
            PipelineValidationError: If strict_mode=True and validation fails
            ValueError: If no input is provided
        """
        # Convert builder to config if needed
        config = pipeline.build() if isinstance(pipeline, PipelineBuilder) else pipeline

        # Build pipeline input (or use override from run())
        if _pipeline_input_override is not None:
            pipeline_input = _pipeline_input_override
        else:
            pipeline_input = self._build_pipeline_input(  # type: ignore[assignment]
                files=files,
                file_urls=file_urls,
                text=text,
                entities=entities,
            )

        if pipeline_input is None:
            raise ValueError(
                "At least one input must be provided: files, file_urls, text, or entities"
            )

        # Validate pipeline (may auto-inject services or raise error)
        validation = validate_pipeline(config, pipeline_input)

        if not validation.valid and config.strict_mode:
            raise PipelineValidationError(
                message="Pipeline validation failed",
                errors=validation.errors,
                suggestion="Fix validation errors or disable strict_mode",
            )

        # Build request body -- map configs by service name, not index
        svc_configs = {sc.service: sc.config for sc in config.services}
        body: dict[str, Any] = {
            "services": [
                {"service": s, "config": svc_configs.get(s, {})}  # type: ignore[call-overload]
                for s in validation.services
            ],
            "store_intermediate": config.store_intermediate,
            "input": pipeline_input.model_dump(exclude_none=True),
        }

        if config.name:
            body["name"] = config.name

        if request_id:
            body["request_id"] = request_id

        response = self._client.post("/api/v1/pipeline/execute", json=body)

        result = PipelineSubmitResponse.model_validate(response.data)
        result.services = validation.services
        result.auto_injected = validation.auto_injected if validation.auto_injected else None
        result.name = config.name

        return self._inject_metadata(result, response)

    def status(self, job_id: str) -> PipelineStatusResponse:
        """
        Get status of a pipeline job.

        Args:
            job_id: Pipeline job identifier

        Returns:
            PipelineStatusResponse with current status
        """
        response = self._client.get(f"/api/v1/pipeline/{job_id}")
        result = PipelineStatusResponse.model_validate(response.data)
        return self._inject_metadata(result, response)

    def wait(
        self,
        job_id: str,
        *,
        poll_interval: float = PIPELINE_POLL_INTERVAL,
        timeout: float = PIPELINE_POLL_TIMEOUT,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> PipelineResultResponse:
        """
        Wait for a pipeline job to complete.

        Args:
            job_id: Pipeline job identifier
            poll_interval: Seconds between polls (default: 5.0)
            timeout: Maximum wait time in seconds (default: 1800.0 = 30 min)
            on_progress: Optional callback ``(status: str, elapsed_seconds: float) -> None``
                invoked on each poll.

        Returns:
            PipelineResultResponse with final results

        Raises:
            JobTimeoutError: If timeout exceeded
            JobError: If pipeline failed
        """
        start_time = time.monotonic()
        consecutive_errors = 0

        while True:
            try:
                status = self.status(job_id)
                consecutive_errors = 0
            except APIError as exc:
                if (
                    getattr(exc, "status_code", None) in _TRANSIENT_STATUS_CODES
                    and consecutive_errors < _MAX_TRANSIENT_RETRIES
                ):
                    consecutive_errors += 1
                    _log.warning(
                        "Transient error polling pipeline %s (attempt %d/%d): %s",
                        job_id,
                        consecutive_errors,
                        _MAX_TRANSIENT_RETRIES,
                        exc,
                    )
                    time.sleep(poll_interval)
                    continue
                raise

            if status.status in ("COMPLETED", "CACHED", "PULLED"):
                return self.retrieve(job_id)
            elif status.status == "FAILED":
                raise JobError(
                    status.error_message or "Pipeline failed",
                    job_id=job_id,
                    error_code=status.error_code,
                    is_resumable=status.is_resumable,
                )
            elif status.status == "RESUMABLE":
                msg = status.error_message or "Pipeline failed"
                if status.failed_stage:
                    msg = f"{msg} at stage '{status.failed_stage}'"
                msg += ". Call job.resume() to continue from the last checkpoint."
                raise JobError(
                    msg,
                    job_id=job_id,
                    error_code=status.error_code,
                    is_resumable=True,
                )
            elif status.status == "CANCELLED":
                raise JobError("Pipeline was cancelled", job_id=job_id)

            elapsed = time.monotonic() - start_time

            if on_progress is not None:
                on_progress(status.status, elapsed)

            if elapsed >= timeout:
                raise JobTimeoutError(
                    f"Pipeline {job_id} did not complete within {timeout}s",
                    job_id=job_id,
                )

            time.sleep(poll_interval)

    def retrieve(self, job_id: str) -> PipelineResultResponse:
        """
        Retrieve pipeline results, fetching from B2 if necessary.

        Args:
            job_id: Pipeline job identifier

        Returns:
            PipelineResultResponse with results

        Raises:
            JobError: If pipeline failed or results unavailable
        """
        response = self._client.get(f"/api/v1/pipeline/{job_id}/result")
        data = response.data

        # If result is in B2, fetch it (handles both ZIP and JSON)
        _zip_meta: dict[str, Any] = {}
        if data.get("output_url"):
            output_data = self._fetch_output_url(data["output_url"])
            data["final_output"] = output_data.get("final_output")
            data["intermediate_results"] = output_data.get("intermediate_results")
            if "_zip_manifest" in output_data:
                _zip_meta["_zip_manifest"] = output_data["_zip_manifest"]
                _zip_meta["_zip_documents"] = output_data.get("_zip_documents", {})

        result = PipelineResultResponse.model_validate(data)
        # Attach raw ZIP data for downstream merge() usage
        if _zip_meta:
            result._zip_manifest = _zip_meta.get("_zip_manifest", {})
            result._zip_documents = _zip_meta.get("_zip_documents", {})
        return self._inject_metadata(result, response)

    def cancel(self, job_id: str) -> JobCancelResponse:
        """
        Cancel a pipeline job.

        Args:
            job_id: Pipeline job identifier

        Returns:
            JobCancelResponse with cancellation status
        """
        response = self._client.delete(f"/api/v1/pipeline/{job_id}")
        result = JobCancelResponse.model_validate(response.data)
        return self._inject_metadata(result, response)

    def resume(self, job_id: str) -> PipelineStatusResponse:
        """Resume a failed/resumable pipeline from its last checkpoint.

        Args:
            job_id: Pipeline job identifier

        Returns:
            PipelineStatusResponse with updated status (QUEUED)

        Raises:
            LatenceError: If pipeline cannot be resumed
        """
        response = self._client.post(f"/api/v1/pipeline/{job_id}/resume")
        return PipelineStatusResponse.model_validate(response.data)

    def stages(self, job_id: str) -> list[StageDownload]:
        """Get per-stage download URLs for completed pipeline stages.

        Args:
            job_id: Pipeline job identifier

        Returns:
            List of StageDownload with presigned URLs for each completed stage
        """
        response = self._client.get(f"/api/v1/pipeline/{job_id}/stages")
        raw_stages = response.data.get("stages", [])
        return [StageDownload.model_validate(s) for s in raw_stages]

    def _build_pipeline_input(
        self,
        *,
        files: list[str | Path | BinaryIO] | None = None,
        file_urls: list[str] | None = None,
        text: str | None = None,
        entities: list[dict[str, Any]] | None = None,
    ) -> PipelineInput | None:
        """Build pipeline input from various sources.

        Large local files (>= PRESIGNED_UPLOAD_THRESHOLD) are uploaded
        directly to B2 via presigned URLs so the gateway never receives
        the raw bytes. When *any* file qualifies, the entire batch is
        uploaded and a ``batch_id`` is returned instead of inline base64.
        """
        file_inputs: list[FileInput] = []

        if file_urls:
            for url in file_urls:
                file_inputs.append(FileInput(url=url))

        if files:
            # Determine file sizes to decide upload strategy
            local_files: list[tuple[str | Path | BinaryIO, int | None]] = []
            for f in files:
                if isinstance(f, (str, Path)):
                    resolved = Path(f)
                    local_files.append(
                        (f, os.path.getsize(resolved) if resolved.is_file() else None)
                    )
                else:
                    local_files.append((f, None))

            total_size = sum(sz for _, sz in local_files if sz is not None)
            needs_batch = (
                any(sz is not None and sz >= PRESIGNED_UPLOAD_THRESHOLD for _, sz in local_files)
                or total_size >= PRESIGNED_UPLOAD_THRESHOLD
            )

            if needs_batch:
                batch_id = self._presigned_batch_upload(local_files)
                if not file_inputs and not text and not entities:
                    return PipelineInput(batch_id=batch_id)
                return PipelineInput(
                    files=file_inputs if file_inputs else None,
                    batch_id=batch_id,
                    text=text,
                    entities=entities,  # type: ignore[arg-type]
                )

            for f, _ in local_files:
                base64_data, filename = file_to_base64(f)
                file_inputs.append(FileInput(base64=base64_data, filename=filename))

        if not file_inputs and not text and not entities:
            return None

        return PipelineInput(
            files=file_inputs if file_inputs else None,
            text=text,
            entities=entities,  # type: ignore[arg-type]
        )

    def _presigned_batch_upload(
        self,
        local_files: list[tuple[str | Path | BinaryIO, int | None]],
    ) -> str:
        """Upload local files to B2 via batch presigned URLs. Returns batch_id."""
        file_meta = []
        for f, _ in local_files:
            if isinstance(f, (str, Path)):
                name = Path(f).name
            else:
                name = getattr(f, "name", "upload")
                if hasattr(f, "name"):
                    name = Path(f.name).name
            file_meta.append(
                {
                    "filename": name,
                    "content_type": _guess_content_type(name),
                }
            )

        resp = self._client.post(
            "/api/v1/pipeline/presign",
            json={"files": file_meta},
        )
        batch_id: str = resp.data["batch_id"]
        presigned_files: list[dict[str, str]] = resp.data["files"]

        for (f, _), presigned in zip(local_files, presigned_files):
            content_type = presigned.get("content_type", "application/octet-stream")
            upload_url = presigned["upload_url"]

            if isinstance(f, (str, Path)):
                with open(f, "rb") as fh:
                    put_resp = httpx.put(
                        upload_url,
                        content=fh,
                        headers={"Content-Type": content_type},
                        timeout=httpx.Timeout(B2_UPLOAD_TIMEOUT),
                    )
                    put_resp.raise_for_status()
            else:
                data = f.read()
                put_resp = httpx.put(
                    upload_url,
                    content=data,
                    headers={"Content-Type": content_type},
                    timeout=httpx.Timeout(B2_UPLOAD_TIMEOUT),
                )
                put_resp.raise_for_status()

        return batch_id

    def _fetch_output_url(self, url: str) -> dict[str, Any]:
        """Fetch pipeline output from B2 presigned URL.

        Handles both ZIP archives (new staged architecture) and raw JSON
        (legacy pipelines).  ZIP detection uses Content-Type header with
        a fallback to ZIP magic bytes.
        """
        with httpx.Client(timeout=B2_PIPELINE_FETCH_TIMEOUT) as client:
            response = client.get(url)
            response.raise_for_status()
            raw = response.content
            if _is_zip(response, raw):
                return _parse_pipeline_zip(raw)
            return json.loads(raw)  # type: ignore[no-any-return]


class AsyncPipeline(AsyncResource):
    """
    Async Pipeline service -- see :class:`Pipeline` for full documentation.

    All methods are async/await equivalents of their sync counterparts.
    """

    def __init__(self, client: BaseAsyncClient) -> None:
        super().__init__(client)

    # =================================================================
    # Primary entry points (v0.2+)
    # =================================================================

    async def run(
        self,
        *,
        files: list[str | Path | BinaryIO] | str | Path | None = None,
        file_urls: list[str] | None = None,
        text: str | None = None,
        entities: list[dict[str, Any]] | None = None,
        steps: dict[str, dict[str, Any]] | None = None,
        name: str | None = None,
        request_id: str | None = None,
    ) -> AsyncJob:
        """Submit a Data Intelligence Pipeline and return an AsyncJob handle.

        See :meth:`Pipeline.run` for full documentation.
        """
        is_files = has_file_input(files=files, file_urls=file_urls)
        config = build_pipeline_config(steps=steps, name=name, has_files=is_files)

        pipeline_input = parse_input(files=files, file_urls=file_urls, text=text, entities=entities)
        if pipeline_input is None:
            raise ValueError(
                "At least one input must be provided: files, file_urls, text, or entities"
            )

        result = await self.execute(
            config,
            files=None,
            text=None,
            entities=None,
            request_id=request_id,
            _pipeline_input_override=pipeline_input,
        )

        return AsyncJob(
            job_id=result.job_id,
            pipeline=self,
            name=name,
            services=result.services,
        )

    async def submit(
        self,
        pipeline: PipelineConfig | PipelineBuilder,
        *,
        files: list[str | Path | BinaryIO] | None = None,
        file_urls: list[str] | None = None,
        text: str | None = None,
        entities: list[dict[str, Any]] | None = None,
        name: str | None = None,
        request_id: str | None = None,
    ) -> AsyncJob:
        """Submit a pipeline from a builder or config and return an AsyncJob.

        See :meth:`Pipeline.submit` for full documentation.
        """
        built_config = pipeline.build() if isinstance(pipeline, PipelineBuilder) else pipeline
        if name and not built_config.name:
            built_config.name = name

        result = await self.execute(
            built_config,
            files=files,
            file_urls=file_urls,
            text=text,
            entities=entities,
            request_id=request_id,
        )

        return AsyncJob(
            job_id=result.job_id,
            pipeline=self,
            name=name or built_config.name,
            services=result.services,
        )

    # =================================================================
    # Legacy entry points (still fully supported)
    # =================================================================

    async def validate(
        self,
        pipeline: PipelineConfig | PipelineBuilder,
        *,
        files: list[str | Path | BinaryIO] | None = None,
        file_urls: list[str] | None = None,
        text: str | None = None,
        entities: list[dict[str, Any]] | None = None,
    ) -> PipelineValidationResult:
        """
        Validate a pipeline configuration without executing it.

        Args:
            pipeline: PipelineConfig or PipelineBuilder to validate
            files: Local files to process (optional, for validation context)
            file_urls: URLs of files to process (optional, for validation context)
            text: Text input (optional, for validation context)
            entities: Pre-extracted entities (optional, for validation context)

        Returns:
            PipelineValidationResult with validation details

        Raises:
            PipelineValidationError: If strict_mode=True and validation fails
        """
        config = pipeline.build() if isinstance(pipeline, PipelineBuilder) else pipeline

        pipeline_input, _ = self._build_pipeline_input(
            files=files,
            file_urls=file_urls,
            text=text,
            entities=entities,
        )

        return validate_pipeline(config, pipeline_input)

    async def execute(
        self,
        pipeline: PipelineConfig | PipelineBuilder,
        *,
        files: list[str | Path | BinaryIO] | None = None,
        file_urls: list[str] | None = None,
        text: str | None = None,
        entities: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        _pipeline_input_override: PipelineInput | None = None,
    ) -> PipelineSubmitResponse:
        """
        Submit a pipeline for async execution.

        Pipelines are always executed asynchronously. Use `wait()` or `status()`
        to poll for completion.

        Args:
            pipeline: PipelineConfig or PipelineBuilder to execute
            files: Local files to process (converted to base64)
            file_urls: URLs of files to process
            text: Text input
            entities: Pre-extracted entities (for ontology-only pipelines)
            request_id: Optional tracking ID

        Returns:
            PipelineSubmitResponse with job_id for polling

        Raises:
            PipelineValidationError: If strict_mode=True and validation fails
            ValueError: If no input is provided
        """
        config = pipeline.build() if isinstance(pipeline, PipelineBuilder) else pipeline

        pending_files: list[tuple[str | Path | BinaryIO, int | None]] | None = None

        if _pipeline_input_override is not None:
            pipeline_input = _pipeline_input_override
        else:
            pipeline_input, pending_files = self._build_pipeline_input(  # type: ignore[assignment]
                files=files,
                file_urls=file_urls,
                text=text,
                entities=entities,
            )

        if pipeline_input is None:
            raise ValueError(
                "At least one input must be provided: files, file_urls, text, or entities"
            )

        if pending_files is not None:
            real_batch_id = await self._presigned_batch_upload_async(pending_files)
            pipeline_input = pipeline_input.model_copy(update={"batch_id": real_batch_id})

        validation = validate_pipeline(config, pipeline_input)

        if not validation.valid and config.strict_mode:
            raise PipelineValidationError(
                message="Pipeline validation failed",
                errors=validation.errors,
                suggestion="Fix validation errors or disable strict_mode",
            )

        svc_configs = {sc.service: sc.config for sc in config.services}
        body: dict[str, Any] = {
            "services": [
                {"service": s, "config": svc_configs.get(s, {})}  # type: ignore[call-overload]
                for s in validation.services
            ],
            "store_intermediate": config.store_intermediate,
            "input": pipeline_input.model_dump(exclude_none=True),
        }

        if config.name:
            body["name"] = config.name

        if request_id:
            body["request_id"] = request_id

        response = await self._client.post("/api/v1/pipeline/execute", json=body)

        result = PipelineSubmitResponse.model_validate(response.data)
        result.services = validation.services
        result.auto_injected = validation.auto_injected if validation.auto_injected else None
        result.name = config.name

        return self._inject_metadata(result, response)

    async def status(self, job_id: str) -> PipelineStatusResponse:
        """
        Get status of a pipeline job.

        Args:
            job_id: Pipeline job identifier

        Returns:
            PipelineStatusResponse with current status
        """
        response = await self._client.get(f"/api/v1/pipeline/{job_id}")
        result = PipelineStatusResponse.model_validate(response.data)
        return self._inject_metadata(result, response)

    async def wait(
        self,
        job_id: str,
        *,
        poll_interval: float = PIPELINE_POLL_INTERVAL,
        timeout: float = PIPELINE_POLL_TIMEOUT,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> PipelineResultResponse:
        """
        Wait for a pipeline job to complete.

        Args:
            job_id: Pipeline job identifier
            poll_interval: Seconds between polls (default: 5.0)
            timeout: Maximum wait time in seconds (default: 1800.0 = 30 min)
            on_progress: Optional callback ``(status: str, elapsed_seconds: float) -> None``
                invoked on each poll.

        Returns:
            PipelineResultResponse with final results

        Raises:
            JobTimeoutError: If timeout exceeded
            JobError: If pipeline failed
        """
        start_time = time.monotonic()
        consecutive_errors = 0

        while True:
            try:
                status = await self.status(job_id)
                consecutive_errors = 0
            except APIError as exc:
                if (
                    getattr(exc, "status_code", None) in _TRANSIENT_STATUS_CODES
                    and consecutive_errors < _MAX_TRANSIENT_RETRIES
                ):
                    consecutive_errors += 1
                    _log.warning(
                        "Transient error polling pipeline %s (attempt %d/%d): %s",
                        job_id,
                        consecutive_errors,
                        _MAX_TRANSIENT_RETRIES,
                        exc,
                    )
                    await asyncio.sleep(poll_interval)
                    continue
                raise

            if status.status in ("COMPLETED", "CACHED", "PULLED"):
                return await self.retrieve(job_id)
            elif status.status == "FAILED":
                raise JobError(
                    status.error_message or "Pipeline failed",
                    job_id=job_id,
                    error_code=status.error_code,
                    is_resumable=status.is_resumable,
                )
            elif status.status == "RESUMABLE":
                msg = status.error_message or "Pipeline failed"
                if status.failed_stage:
                    msg = f"{msg} at stage '{status.failed_stage}'"
                msg += ". Call job.resume() to continue from the last checkpoint."
                raise JobError(
                    msg,
                    job_id=job_id,
                    error_code=status.error_code,
                    is_resumable=True,
                )
            elif status.status == "CANCELLED":
                raise JobError("Pipeline was cancelled", job_id=job_id)

            elapsed = time.monotonic() - start_time

            if on_progress is not None:
                on_progress(status.status, elapsed)

            if elapsed >= timeout:
                raise JobTimeoutError(
                    f"Pipeline {job_id} did not complete within {timeout}s",
                    job_id=job_id,
                )

            await asyncio.sleep(poll_interval)

    async def retrieve(self, job_id: str) -> PipelineResultResponse:
        """
        Retrieve pipeline results, fetching from B2 if necessary.

        Args:
            job_id: Pipeline job identifier

        Returns:
            PipelineResultResponse with results

        Raises:
            JobError: If pipeline failed or results unavailable
        """
        response = await self._client.get(f"/api/v1/pipeline/{job_id}/result")
        data = response.data

        _zip_meta: dict[str, Any] = {}
        if data.get("output_url"):
            output_data = await self._fetch_output_url(data["output_url"])
            data["final_output"] = output_data.get("final_output")
            data["intermediate_results"] = output_data.get("intermediate_results")
            if "_zip_manifest" in output_data:
                _zip_meta["_zip_manifest"] = output_data["_zip_manifest"]
                _zip_meta["_zip_documents"] = output_data.get("_zip_documents", {})

        result = PipelineResultResponse.model_validate(data)
        if _zip_meta:
            result._zip_manifest = _zip_meta.get("_zip_manifest", {})
            result._zip_documents = _zip_meta.get("_zip_documents", {})
        return self._inject_metadata(result, response)

    async def cancel(self, job_id: str) -> JobCancelResponse:
        """
        Cancel a pipeline job.

        Args:
            job_id: Pipeline job identifier

        Returns:
            JobCancelResponse with cancellation status
        """
        response = await self._client.delete(f"/api/v1/pipeline/{job_id}")
        result = JobCancelResponse.model_validate(response.data)
        return self._inject_metadata(result, response)

    async def resume(self, job_id: str) -> PipelineStatusResponse:
        """Resume a failed/resumable pipeline from its last checkpoint.

        Args:
            job_id: Pipeline job identifier

        Returns:
            PipelineStatusResponse with updated status (QUEUED)

        Raises:
            LatenceError: If pipeline cannot be resumed
        """
        response = await self._client.post(f"/api/v1/pipeline/{job_id}/resume")
        return PipelineStatusResponse.model_validate(response.data)

    async def stages(self, job_id: str) -> list[StageDownload]:
        """Get per-stage download URLs for completed pipeline stages.

        Args:
            job_id: Pipeline job identifier

        Returns:
            List of StageDownload with presigned URLs for each completed stage
        """
        response = await self._client.get(f"/api/v1/pipeline/{job_id}/stages")
        raw_stages = response.data.get("stages", [])
        return [StageDownload.model_validate(s) for s in raw_stages]

    def _build_pipeline_input(
        self,
        *,
        files: list[str | Path | BinaryIO] | None = None,
        file_urls: list[str] | None = None,
        text: str | None = None,
        entities: list[dict[str, Any]] | None = None,
    ) -> tuple[PipelineInput | None, list[tuple[str | Path | BinaryIO, int | None]] | None]:
        """Build pipeline input from various sources.

        Returns a tuple of (PipelineInput, pending_local_files). When
        ``pending_local_files`` is not None the caller must upload them
        via :meth:`_presigned_batch_upload_async` and patch the resulting
        ``batch_id`` into the input before submission.
        """
        file_inputs: list[FileInput] = []

        if file_urls:
            for url in file_urls:
                file_inputs.append(FileInput(url=url))

        if files:
            local_files: list[tuple[str | Path | BinaryIO, int | None]] = []
            for f in files:
                if isinstance(f, (str, Path)):
                    resolved = Path(f)
                    local_files.append(
                        (f, os.path.getsize(resolved) if resolved.is_file() else None)
                    )
                else:
                    local_files.append((f, None))

            total_size = sum(sz for _, sz in local_files if sz is not None)
            needs_batch = (
                any(sz is not None and sz >= PRESIGNED_UPLOAD_THRESHOLD for _, sz in local_files)
                or total_size >= PRESIGNED_UPLOAD_THRESHOLD
            )

            if needs_batch:
                pending_input = PipelineInput(
                    files=file_inputs if file_inputs else None,
                    batch_id="__pending__",
                    text=text,
                    entities=entities,  # type: ignore[arg-type]
                )
                return pending_input, local_files

            for f, _ in local_files:
                base64_data, filename = file_to_base64(f)
                file_inputs.append(FileInput(base64=base64_data, filename=filename))

        if not file_inputs and not text and not entities:
            return None, None

        return PipelineInput(
            files=file_inputs if file_inputs else None,
            text=text,
            entities=entities,  # type: ignore[arg-type]
        ), None

    async def _presigned_batch_upload_async(
        self,
        local_files: list[tuple[str | Path | BinaryIO, int | None]],
    ) -> str:
        """Upload local files to B2 via batch presigned URLs (async). Returns batch_id."""
        file_meta = []
        for f, _ in local_files:
            if isinstance(f, (str, Path)):
                name = Path(f).name
            else:
                name = getattr(f, "name", "upload")
                if hasattr(f, "name"):
                    name = Path(f.name).name
            file_meta.append(
                {
                    "filename": name,
                    "content_type": _guess_content_type(name),
                }
            )

        resp = await self._client.post(
            "/api/v1/pipeline/presign",
            json={"files": file_meta},
        )
        batch_id: str = resp.data["batch_id"]
        presigned_files: list[dict[str, str]] = resp.data["files"]

        sem = asyncio.Semaphore(6)

        async def _upload_one(f: str | Path | BinaryIO, presigned: dict[str, str]) -> None:
            ct = presigned.get("content_type", "application/octet-stream")
            upload_url = presigned["upload_url"]
            if isinstance(f, (str, Path)):
                data = await asyncio.to_thread(Path(f).read_bytes)
            else:
                data = await asyncio.to_thread(f.read)
            async with sem:
                async with httpx.AsyncClient(timeout=httpx.Timeout(B2_UPLOAD_TIMEOUT)) as client:
                    put_resp = await client.put(
                        upload_url,
                        content=data,
                        headers={"Content-Type": ct},
                    )
                    put_resp.raise_for_status()
            del data

        await asyncio.gather(
            *(_upload_one(f, p) for (f, _), p in zip(local_files, presigned_files))
        )

        return batch_id

    async def _fetch_output_url(self, url: str) -> dict[str, Any]:
        """Fetch pipeline output from B2 presigned URL.

        Handles both ZIP archives (new staged architecture) and raw JSON
        (legacy pipelines).
        """
        async with httpx.AsyncClient(timeout=B2_PIPELINE_FETCH_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status()
            raw = response.content
            if _is_zip(response, raw):
                return _parse_pipeline_zip(raw)
            return json.loads(raw)  # type: ignore[no-any-return]
