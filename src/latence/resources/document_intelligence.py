"""Document Intelligence service resource."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, Union, overload

try:
    from bs4 import BeautifulSoup  # type: ignore[import-untyped,import-not-found]

    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False

from .._constants import PRESIGNED_UPLOAD_THRESHOLD
from .._models import (
    JobSubmittedResponse,
    OutputOptions,
    PipelineOptions,
    PredictOptions,
    ProcessDocumentResponse,
    RefineOptions,
)
from .._utils import file_to_base64
from ._base import AsyncResource, SyncResource

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient


class DocumentIntelligence(SyncResource):
    """
    Document Intelligence service (V2) - extract text from documents using advanced OCR.

    Supports PDFs, images (PNG, JPG, HEIC, TIFF, BMP, WEBP), Office files, and more.
    Features layout detection, chart recognition, table extraction, and configurable options.

    Example:
        >>> # Basic usage - from URL
        >>> result = client.document_intelligence.process(
        ...     file_url="https://example.com/doc.pdf"
        ... )
        >>> print(result.content)

        >>> # From local file with performance mode (auto-refinement)
        >>> result = client.document_intelligence.process(
        ...     file_path="/path/to/document.pdf",
        ...     mode="performance"
        ... )

        >>> # With custom pipeline options
        >>> result = client.document_intelligence.process(
        ...     file_path="/path/to/document.pdf",
        ...     pipeline_options={"use_chart_recognition": True, "use_seal_recognition": True},
        ...     output_format="json"
        ... )
    """

    def __init__(self, client: BaseSyncClient) -> None:
        super().__init__(client)

    @overload
    def process(
        self,
        *,
        file_path: str | Path | BinaryIO | None = None,
        file_base64: str | None = None,
        file_url: str | None = None,
        filename: str | None = None,
        mode: Literal["default", "performance"] = "default",
        output_format: Literal["markdown", "json", "html", "xlsx"] = "markdown",
        max_pages: int | None = None,
        target_longest: int | None = None,
        pipeline_options: PipelineOptions | dict[str, Any] | None = None,
        predict_options: PredictOptions | dict[str, Any] | None = None,
        output_options: OutputOptions | dict[str, Any] | None = None,
        use_layout_detection: bool | None = None,
        use_chart_recognition: bool | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> ProcessDocumentResponse: ...

    @overload
    def process(
        self,
        *,
        file_path: str | Path | BinaryIO | None = None,
        file_base64: str | None = None,
        file_url: str | None = None,
        filename: str | None = None,
        mode: Literal["default", "performance"] = "default",
        output_format: Literal["markdown", "json", "html", "xlsx"] = "markdown",
        max_pages: int | None = None,
        target_longest: int | None = None,
        pipeline_options: PipelineOptions | dict[str, Any] | None = None,
        predict_options: PredictOptions | dict[str, Any] | None = None,
        output_options: OutputOptions | dict[str, Any] | None = None,
        use_layout_detection: bool | None = None,
        use_chart_recognition: bool | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def process(
        self,
        *,
        file_path: str | Path | BinaryIO | None = None,
        file_base64: str | None = None,
        file_url: str | None = None,
        filename: str | None = None,
        mode: Literal["default", "performance"] = "default",
        output_format: Literal["markdown", "json", "html", "xlsx"] = "markdown",
        max_pages: int | None = None,
        target_longest: int | None = None,
        pipeline_options: PipelineOptions | dict[str, Any] | None = None,
        predict_options: PredictOptions | dict[str, Any] | None = None,
        output_options: OutputOptions | dict[str, Any] | None = None,
        use_layout_detection: bool | None = None,
        use_chart_recognition: bool | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[ProcessDocumentResponse, JobSubmittedResponse]:
        """
        Process a document and extract text content using advanced OCR.

        Provide one of: file_path, file_base64, or file_url.

        Args:
            file_path: Local file path, Path object, or file-like object
            file_base64: Base64-encoded file data
            file_url: Public URL to file
            filename: Filename for type detection (auto-detected from file_path)
            mode: Processing mode:
                - "default": Standard processing
                - "performance": Processing with auto-refinement (table merge, title relevel)
            output_format: Output format - "markdown", "json", "html", or "xlsx"
            max_pages: Limit pages processed (None = all pages)
            target_longest: Target longest dimension for image preprocessing (None = native)
            pipeline_options: Pipeline configuration (layout detection, chart recognition,
                seal recognition, etc.). See PipelineOptions for details.
            predict_options: Per-request inference options (thresholds, VLM sampling params).
                See PredictOptions for details.
            output_options: Output formatting options (pretty markdown, include images, etc.).
                See OutputOptions for details.
            use_layout_detection: Simple top-level layout detection toggle.
            use_chart_recognition: Simple top-level chart recognition toggle.
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling (recommended for large files)

        Returns:
            ProcessDocumentResponse with extracted content,
            or JobSubmittedResponse if return_job=True
        """
        # Handle file_path: auto-detect large files for presigned upload
        if file_path is not None:
            resolved = Path(file_path) if isinstance(file_path, (str, Path)) else None
            if resolved is not None and resolved.is_file():
                file_size = os.path.getsize(resolved)
                detected_filename = resolved.name
                if filename is None:
                    filename = detected_filename
                if file_size > PRESIGNED_UPLOAD_THRESHOLD:
                    # Large file → upload directly to B2 via presigned URL
                    file_url = self._presigned_upload(resolved, filename)
                    file_base64 = None  # Skip base64 encoding
                else:
                    # Small file → inline base64 (existing path)
                    file_base64, _ = file_to_base64(file_path)
            else:
                # File-like object — always use base64 (can't stat for size)
                file_base64, detected_filename = file_to_base64(file_path)
                if filename is None:
                    filename = detected_filename

        if file_base64 is None and file_url is None:
            raise ValueError("One of file_path, file_base64, or file_url must be provided")

        # Default filename if still not set
        if filename is None:
            filename = "document.pdf"

        # Convert Pydantic models to dicts if needed
        pipeline_opts_dict = None
        if pipeline_options is not None:
            pipeline_opts_dict = (
                pipeline_options.model_dump(exclude_none=True)
                if isinstance(pipeline_options, PipelineOptions)
                else pipeline_options
            )

        predict_opts_dict = None
        if predict_options is not None:
            predict_opts_dict = (
                predict_options.model_dump(exclude_none=True)
                if isinstance(predict_options, PredictOptions)
                else predict_options
            )

        output_opts_dict = None
        if output_options is not None:
            output_opts_dict = (
                output_options.model_dump(exclude_none=True)
                if isinstance(output_options, OutputOptions)
                else output_options
            )

        body = self._build_request_body(
            file_base64=file_base64,
            file_url=file_url,
            filename=filename,
            mode=mode,
            output_format=output_format,
            max_pages=max_pages,
            target_longest=target_longest,
            pipeline_options=pipeline_opts_dict,
            predict_options=predict_opts_dict,
            output_options=output_opts_dict,
            use_layout_detection=use_layout_detection,
            use_chart_recognition=use_chart_recognition,
            request_id=request_id,
            return_job=return_job,
        )

        response = self._client.post("/api/v1/document_intelligence/process", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = ProcessDocumentResponse.model_validate(response.data)  # type: ignore[assignment]

        return self._inject_metadata(result, response)

    @overload
    def refine(
        self,
        *,
        pages_result: list[dict[str, Any]],
        refine_options: RefineOptions | dict[str, Any] | None = None,
        output_format: Literal["markdown", "json", "html", "xlsx"] = "markdown",
        output_options: OutputOptions | dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> ProcessDocumentResponse: ...

    @overload
    def refine(
        self,
        *,
        pages_result: list[dict[str, Any]],
        refine_options: RefineOptions | dict[str, Any] | None = None,
        output_format: Literal["markdown", "json", "html", "xlsx"] = "markdown",
        output_options: OutputOptions | dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def refine(
        self,
        *,
        pages_result: list[dict[str, Any]],
        refine_options: RefineOptions | dict[str, Any] | None = None,
        output_format: Literal["markdown", "json", "html", "xlsx"] = "markdown",
        output_options: OutputOptions | dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[ProcessDocumentResponse, JobSubmittedResponse]:
        """Run explicit multi-page refinement on prior process results."""
        refine_opts_dict = None
        if refine_options is not None:
            refine_opts_dict = (
                refine_options.model_dump(exclude_none=True)
                if isinstance(refine_options, RefineOptions)
                else refine_options
            )

        output_opts_dict = None
        if output_options is not None:
            output_opts_dict = (
                output_options.model_dump(exclude_none=True)
                if isinstance(output_options, OutputOptions)
                else output_options
            )

        body = self._build_request_body(
            pages_result=pages_result,
            refine_options=refine_opts_dict,
            output_format=output_format,
            output_options=output_opts_dict,
            request_id=request_id,
            return_job=return_job,
        )

        response = self._client.post("/api/v1/document_intelligence/refine", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = ProcessDocumentResponse.model_validate(response.data)  # type: ignore[assignment]

        return self._inject_metadata(result, response)


class AsyncDocumentIntelligence(AsyncResource):
    """Async Document Intelligence service (V2)."""

    def __init__(self, client: BaseAsyncClient) -> None:
        super().__init__(client)

    @overload
    async def process(
        self,
        *,
        file_path: str | Path | BinaryIO | None = None,
        file_base64: str | None = None,
        file_url: str | None = None,
        filename: str | None = None,
        mode: Literal["default", "performance"] = "default",
        output_format: Literal["markdown", "json", "html", "xlsx"] = "markdown",
        max_pages: int | None = None,
        target_longest: int | None = None,
        pipeline_options: PipelineOptions | dict[str, Any] | None = None,
        predict_options: PredictOptions | dict[str, Any] | None = None,
        output_options: OutputOptions | dict[str, Any] | None = None,
        use_layout_detection: bool | None = None,
        use_chart_recognition: bool | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> ProcessDocumentResponse: ...

    @overload
    async def process(
        self,
        *,
        file_path: str | Path | BinaryIO | None = None,
        file_base64: str | None = None,
        file_url: str | None = None,
        filename: str | None = None,
        mode: Literal["default", "performance"] = "default",
        output_format: Literal["markdown", "json", "html", "xlsx"] = "markdown",
        max_pages: int | None = None,
        target_longest: int | None = None,
        pipeline_options: PipelineOptions | dict[str, Any] | None = None,
        predict_options: PredictOptions | dict[str, Any] | None = None,
        output_options: OutputOptions | dict[str, Any] | None = None,
        use_layout_detection: bool | None = None,
        use_chart_recognition: bool | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def process(
        self,
        *,
        file_path: str | Path | BinaryIO | None = None,
        file_base64: str | None = None,
        file_url: str | None = None,
        filename: str | None = None,
        mode: Literal["default", "performance"] = "default",
        output_format: Literal["markdown", "json", "html", "xlsx"] = "markdown",
        max_pages: int | None = None,
        target_longest: int | None = None,
        pipeline_options: PipelineOptions | dict[str, Any] | None = None,
        predict_options: PredictOptions | dict[str, Any] | None = None,
        output_options: OutputOptions | dict[str, Any] | None = None,
        use_layout_detection: bool | None = None,
        use_chart_recognition: bool | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[ProcessDocumentResponse, JobSubmittedResponse]:
        """
        Process a document and extract text content using advanced OCR.

        Provide one of: file_path, file_base64, or file_url.

        Args:
            file_path: Local file path, Path object, or file-like object
            file_base64: Base64-encoded file data
            file_url: Public URL to file
            filename: Filename for type detection (auto-detected from file_path)
            mode: Processing mode:
                - "default": Standard processing
                - "performance": Processing with auto-refinement (table merge, title relevel)
            output_format: Output format - "markdown", "json", "html", or "xlsx"
            max_pages: Limit pages processed (None = all pages)
            target_longest: Target longest dimension for image preprocessing (None = native)
            pipeline_options: Pipeline configuration (layout detection, chart recognition,
                seal recognition, etc.). See PipelineOptions for details.
            predict_options: Per-request inference options (thresholds, VLM sampling params).
                See PredictOptions for details.
            output_options: Output formatting options (pretty markdown, include images, etc.).
                See OutputOptions for details.
            use_layout_detection: Simple top-level layout detection toggle.
            use_chart_recognition: Simple top-level chart recognition toggle.
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling (recommended for large files)

        Returns:
            ProcessDocumentResponse with extracted content,
            or JobSubmittedResponse if return_job=True
        """
        # Handle file_path: auto-detect large files for presigned upload
        if file_path is not None:
            resolved = Path(file_path) if isinstance(file_path, (str, Path)) else None
            if resolved is not None and resolved.is_file():
                file_size = os.path.getsize(resolved)
                detected_filename = resolved.name
                if filename is None:
                    filename = detected_filename
                if file_size > PRESIGNED_UPLOAD_THRESHOLD:
                    # Large file → upload directly to B2 via presigned URL
                    file_url = await self._presigned_upload(resolved, filename)
                    file_base64 = None  # Skip base64 encoding
                else:
                    # Small file → inline base64 (existing path)
                    file_base64, _ = file_to_base64(file_path)
            else:
                # File-like object — always use base64 (can't stat for size)
                file_base64, detected_filename = file_to_base64(file_path)
                if filename is None:
                    filename = detected_filename

        if file_base64 is None and file_url is None:
            raise ValueError("One of file_path, file_base64, or file_url must be provided")

        if filename is None:
            filename = "document.pdf"

        # Convert Pydantic models to dicts if needed
        pipeline_opts_dict = None
        if pipeline_options is not None:
            pipeline_opts_dict = (
                pipeline_options.model_dump(exclude_none=True)
                if isinstance(pipeline_options, PipelineOptions)
                else pipeline_options
            )

        predict_opts_dict = None
        if predict_options is not None:
            predict_opts_dict = (
                predict_options.model_dump(exclude_none=True)
                if isinstance(predict_options, PredictOptions)
                else predict_options
            )

        output_opts_dict = None
        if output_options is not None:
            output_opts_dict = (
                output_options.model_dump(exclude_none=True)
                if isinstance(output_options, OutputOptions)
                else output_options
            )

        body = self._build_request_body(
            file_base64=file_base64,
            file_url=file_url,
            filename=filename,
            mode=mode,
            output_format=output_format,
            max_pages=max_pages,
            target_longest=target_longest,
            pipeline_options=pipeline_opts_dict,
            predict_options=predict_opts_dict,
            output_options=output_opts_dict,
            use_layout_detection=use_layout_detection,
            use_chart_recognition=use_chart_recognition,
            request_id=request_id,
            return_job=return_job,
        )

        response = await self._client.post("/api/v1/document_intelligence/process", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = ProcessDocumentResponse.model_validate(response.data)  # type: ignore[assignment]

        return self._inject_metadata(result, response)

    @overload
    async def refine(
        self,
        *,
        pages_result: list[dict[str, Any]],
        refine_options: RefineOptions | dict[str, Any] | None = None,
        output_format: Literal["markdown", "json", "html", "xlsx"] = "markdown",
        output_options: OutputOptions | dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> ProcessDocumentResponse: ...

    @overload
    async def refine(
        self,
        *,
        pages_result: list[dict[str, Any]],
        refine_options: RefineOptions | dict[str, Any] | None = None,
        output_format: Literal["markdown", "json", "html", "xlsx"] = "markdown",
        output_options: OutputOptions | dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def refine(
        self,
        *,
        pages_result: list[dict[str, Any]],
        refine_options: RefineOptions | dict[str, Any] | None = None,
        output_format: Literal["markdown", "json", "html", "xlsx"] = "markdown",
        output_options: OutputOptions | dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[ProcessDocumentResponse, JobSubmittedResponse]:
        """Run explicit multi-page refinement on prior process results (async)."""
        refine_opts_dict = None
        if refine_options is not None:
            refine_opts_dict = (
                refine_options.model_dump(exclude_none=True)
                if isinstance(refine_options, RefineOptions)
                else refine_options
            )

        output_opts_dict = None
        if output_options is not None:
            output_opts_dict = (
                output_options.model_dump(exclude_none=True)
                if isinstance(output_options, OutputOptions)
                else output_options
            )

        body = self._build_request_body(
            pages_result=pages_result,
            refine_options=refine_opts_dict,
            output_format=output_format,
            output_options=output_opts_dict,
            request_id=request_id,
            return_job=return_job,
        )

        response = await self._client.post("/api/v1/document_intelligence/refine", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = ProcessDocumentResponse.model_validate(response.data)  # type: ignore[assignment]

        return self._inject_metadata(result, response)


def clean_markdown(text: str) -> str:
    """
    Clean markdown text by removing layout noise (images, empty divs) while preserving tables.

    This utility removes:
    - <img> tags
    - <div> tags (wrapping content) except when they contain tables
    - Excessive newlines

    Args:
        text (str): The raw markdown text to clean.

    Returns:
        str: The cleaned markdown text.
    """
    if not text:
        return ""

    if not HAS_BEAUTIFULSOUP:
        # Fallback to simple regex if BS4 is not available
        return re.sub(r"<img[^>]+>", "", text)

    try:
        soup = BeautifulSoup(text, "html.parser")

        # Remove all img tags
        for img in soup.find_all("img"):
            img.decompose()

        # Remove div tags but PRESERVE tables
        for div in soup.find_all("div"):
            if div.find("table"):
                div.unwrap()  # Remove div tag, keep table content
            else:
                div.decompose()  # Remove div and its content (noise)

        # Get text back
        text = str(soup)

        # Collapse multiple spaces/newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text
    except Exception:
        # If parsing fails, return original text
        return text
