"""Entity Extraction service resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Union, overload

from .._models import CustomLabel, ExtractionConfig, ExtractResponse, JobSubmittedResponse
from ._base import AsyncResource, SyncResource

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient


class Extraction(SyncResource):
    """
    Entity Extraction service - zero-shot entity extraction.

    Example:
        >>> result = client.extraction.extract(
        ...     text="Apple Inc. is in Cupertino.",
        ...     config={"label_mode": "user", "user_labels": ["organization", "location"]}
        ... )
        >>> for entity in result.entities:
        ...     print(f"{entity.text}: {entity.label}")
    """

    def __init__(self, client: BaseSyncClient) -> None:
        super().__init__(client)

    @overload
    def extract(
        self,
        text: str,
        *,
        config: dict[str, Any] | ExtractionConfig | None = None,
        custom_labels: list[dict[str, str]] | list[CustomLabel] | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> ExtractResponse: ...

    @overload
    def extract(
        self,
        text: str,
        *,
        config: dict[str, Any] | ExtractionConfig | None = None,
        custom_labels: list[dict[str, str]] | list[CustomLabel] | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def extract(
        self,
        text: str,
        *,
        config: dict[str, Any] | ExtractionConfig | None = None,
        custom_labels: list[dict[str, str]] | list[CustomLabel] | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[ExtractResponse, JobSubmittedResponse]:
        """
        Extract entities from text.

        Args:
            text: Input text
            config: Extraction configuration:
                - label_mode: "user", "hybrid", or "generated"
                - user_labels: Labels to extract (required for user mode)
                - threshold: Confidence threshold (0.0-1.0)
                - enable_refinement: Use LLM to refine low-confidence entities
                - enforce_refinement: Refine ALL entities with LLM
            custom_labels: Custom regex extractors
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling

        Returns:
            ExtractResponse with extracted entities
        """
        # Convert Pydantic models to dicts
        cfg: dict[str, Any] | None = None
        if config is not None:
            if isinstance(config, ExtractionConfig):
                cfg = config.model_dump(exclude_none=True)
            else:
                cfg = config

        labels: list[dict[str, str]] | None = None
        if custom_labels is not None:
            labels = []
            for cl in custom_labels:
                if isinstance(cl, CustomLabel):
                    labels.append({"label_name": cl.label_name, "extractor": cl.extractor})
                else:
                    labels.append(cl)

        body = self._build_request_body(
            text=text,
            config=cfg,
            custom_labels=labels,
            request_id=request_id,
            return_job=return_job,
        )

        response = self._client.post("/api/v1/extraction/extract", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = ExtractResponse.model_validate(response.data)

        return self._inject_metadata(result, response)


class AsyncExtraction(AsyncResource):
    """Async Entity Extraction service."""

    def __init__(self, client: BaseAsyncClient) -> None:
        super().__init__(client)

    @overload
    async def extract(
        self,
        text: str,
        *,
        config: dict[str, Any] | ExtractionConfig | None = None,
        custom_labels: list[dict[str, str]] | list[CustomLabel] | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> ExtractResponse: ...

    @overload
    async def extract(
        self,
        text: str,
        *,
        config: dict[str, Any] | ExtractionConfig | None = None,
        custom_labels: list[dict[str, str]] | list[CustomLabel] | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def extract(
        self,
        text: str,
        *,
        config: dict[str, Any] | ExtractionConfig | None = None,
        custom_labels: list[dict[str, str]] | list[CustomLabel] | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[ExtractResponse, JobSubmittedResponse]:
        """
        Extract entities from text.

        Args:
            text: Input text
            config: Extraction configuration:
                - label_mode: "user", "hybrid", or "generated"
                - user_labels: Labels to extract (required for user mode)
                - threshold: Confidence threshold (0.0-1.0)
                - enable_refinement: Use LLM to refine low-confidence entities
                - enforce_refinement: Refine ALL entities with LLM
            custom_labels: Custom regex extractors
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling

        Returns:
            ExtractResponse with extracted entities
        """
        cfg: dict[str, Any] | None = None
        if config is not None:
            if isinstance(config, ExtractionConfig):
                cfg = config.model_dump(exclude_none=True)
            else:
                cfg = config

        labels: list[dict[str, str]] | None = None
        if custom_labels is not None:
            labels = []
            for cl in custom_labels:
                if isinstance(cl, CustomLabel):
                    labels.append({"label_name": cl.label_name, "extractor": cl.extractor})
                else:
                    labels.append(cl)

        body = self._build_request_body(
            text=text,
            config=cfg,
            custom_labels=labels,
            request_id=request_id,
            return_job=return_job,
        )

        response = await self._client.post("/api/v1/extraction/extract", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = ExtractResponse.model_validate(response.data)

        return self._inject_metadata(result, response)
