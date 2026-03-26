"""Redaction service resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Union, overload

from .._models import CustomLabel, DetectPIIResponse, JobSubmittedResponse, RedactionConfig
from ._base import AsyncResource, SyncResource

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient


class Redaction(SyncResource):
    """
    Redaction service - detect and redact PII.

    Example:
        >>> result = client.redaction.detect_pii(
        ...     text="Contact john@email.com or call 555-1234",
        ...     config={"redact": True, "redaction_mode": "mask"}
        ... )
        >>> print(result.redacted_text)
        # "Contact [EMAIL] or call [PHONE_NUMBER]"
    """

    def __init__(self, client: BaseSyncClient) -> None:
        super().__init__(client)

    @overload
    def detect_pii(
        self,
        text: str,
        *,
        config: dict[str, Any] | RedactionConfig | None = None,
        custom_labels: list[dict[str, str]] | list[CustomLabel] | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> DetectPIIResponse: ...

    @overload
    def detect_pii(
        self,
        text: str,
        *,
        config: dict[str, Any] | RedactionConfig | None = None,
        custom_labels: list[dict[str, str]] | list[CustomLabel] | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def detect_pii(
        self,
        text: str,
        *,
        config: dict[str, Any] | RedactionConfig | None = None,
        custom_labels: list[dict[str, str]] | list[CustomLabel] | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[DetectPIIResponse, JobSubmittedResponse]:
        """
        Detect and optionally redact PII in text.

        Args:
            text: Input text
            config: Redaction configuration:
                - mode: "balanced" or "strict"
                - redact: Whether to redact (True) or just detect (False)
                - redaction_mode: "mask" or "replace"
                - enable_refinement: Use AI to refine detection
            custom_labels: Custom PII patterns (regex)
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling

        Returns:
            DetectPIIResponse with detected PII and optionally redacted text
        """
        cfg: dict[str, Any] | None = None
        if config is not None:
            if isinstance(config, RedactionConfig):
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

        response = self._client.post("/api/v1/redaction/redact", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = DetectPIIResponse.model_validate(response.data)  # type: ignore[assignment]

        return self._inject_metadata(result, response)


class AsyncRedaction(AsyncResource):
    """Async Redaction service."""

    def __init__(self, client: BaseAsyncClient) -> None:
        super().__init__(client)

    @overload
    async def detect_pii(
        self,
        text: str,
        *,
        config: dict[str, Any] | RedactionConfig | None = None,
        custom_labels: list[dict[str, str]] | list[CustomLabel] | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> DetectPIIResponse: ...

    @overload
    async def detect_pii(
        self,
        text: str,
        *,
        config: dict[str, Any] | RedactionConfig | None = None,
        custom_labels: list[dict[str, str]] | list[CustomLabel] | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def detect_pii(
        self,
        text: str,
        *,
        config: dict[str, Any] | RedactionConfig | None = None,
        custom_labels: list[dict[str, str]] | list[CustomLabel] | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[DetectPIIResponse, JobSubmittedResponse]:
        """
        Detect and optionally redact PII in text.

        Args:
            text: Input text
            config: Redaction configuration:
                - mode: "balanced" or "strict"
                - redact: Whether to redact (True) or just detect (False)
                - redaction_mode: "mask" or "replace"
                - enable_refinement: Use AI to refine detection
            custom_labels: Custom PII patterns (regex)
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling

        Returns:
            DetectPIIResponse with detected PII and optionally redacted text
        """
        cfg: dict[str, Any] | None = None
        if config is not None:
            if isinstance(config, RedactionConfig):
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

        response = await self._client.post("/api/v1/redaction/redact", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = DetectPIIResponse.model_validate(response.data)  # type: ignore[assignment]

        return self._inject_metadata(result, response)
