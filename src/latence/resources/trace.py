"""Trace service resource: groundedness and phantom-hallucination scoring.

The gateway exposes three endpoints under ``/api/v1/trace/*`` that share one
RunPod pod. The lane (``scoring_mode``) is pinned server-side by the URL:

- ``POST /api/v1/trace/rag``    -- RAG groundedness lane
- ``POST /api/v1/trace/code``   -- agentic-code phantom scoring lane
- ``POST /api/v1/trace/rollup`` -- stateless session-level aggregation

Example:
    >>> from latence import Latence
    >>> client = Latence(api_key="lat_xxx")
    >>>
    >>> # RAG groundedness
    >>> r = client.experimental.trace.rag(
    ...     response_text="Paris is the capital of France.",
    ...     raw_context="France's capital city is Paris.",
    ... )
    >>> print(r.score, r.band)
    >>>
    >>> # Agentic-code phantom scoring with session chaining
    >>> t1 = client.experimental.trace.code(
    ...     response_text="def add(a, b): return a + b",
    ...     raw_context="# utils.py\\ndef sub(a, b): return a - b",
    ...     response_language_hint="python",
    ... )
    >>> t2 = client.experimental.trace.code(
    ...     response_text="def mul(a, b): return a * b",
    ...     raw_context="# utils.py\\ndef sub(a, b): return a - b",
    ...     response_language_hint="python",
    ...     session_state=t1.next_session_state,
    ... )
    >>>
    >>> # Session-level rollup
    >>> rollup = client.experimental.trace.rollup(turns=[t1, t2])
    >>> print(rollup.noise_pct, rollup.recommendations)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Union, overload

from .._models import (
    AttributionMode,
    HeatmapFormat,
    JobSubmittedResponse,
    PrimaryMetric,
    ResponseLanguageHint,
    SegmentationMode,
    SessionState,
    SupportUnitInput,
    TraceCodeResponse,
    TraceRagResponse,
    TraceResponse,
    TraceRollupResponse,
)
from ._base import AsyncResource, SyncResource

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_support_units(
    support_units: list[SupportUnitInput | dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Coerce a list of ``SupportUnitInput`` / dict mixtures to plain dicts."""
    if support_units is None:
        return None
    out: list[dict[str, Any]] = []
    for unit in support_units:
        if isinstance(unit, SupportUnitInput):
            out.append(unit.model_dump(exclude_none=True))
        else:
            out.append(unit)
    return out


def _normalize_session_state(
    session_state: SessionState | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Coerce a ``SessionState`` or plain dict to the on-wire dict shape."""
    if session_state is None:
        return None
    if isinstance(session_state, SessionState):
        return session_state.model_dump(exclude_none=True)
    return session_state


def _normalize_turns(
    turns: list[TraceResponse | dict[str, Any]],
) -> list[dict[str, Any]]:
    """Coerce a list of scoring responses / dicts to the rollup turn shape."""
    out: list[dict[str, Any]] = []
    for turn in turns:
        if isinstance(turn, TraceResponse):
            out.append(turn.model_dump(exclude_none=True))
        else:
            out.append(turn)
    return out


def _require_premise_lane(
    *,
    raw_context: str | None,
    chunk_ids: list[str | int] | None,
    support_units: list[SupportUnitInput | dict[str, Any]] | None,
) -> None:
    """Mirror the gateway's ``required_fields_any`` precondition.

    Surfaces a clean ``ValueError`` before the HTTP round-trip instead of
    letting the caller discover it from a 400 response.
    """
    if not (raw_context or chunk_ids or support_units):
        raise ValueError(
            "Trace scoring requires at least one of: raw_context, chunk_ids, or support_units."
        )


def _require_response_text(response_text: str) -> None:
    if not response_text or not response_text.strip():
        raise ValueError("`response_text` must be a non-empty string.")


# ---------------------------------------------------------------------------
# Sync resource
# ---------------------------------------------------------------------------


class Trace(SyncResource):
    """
    Groundedness and phantom-hallucination scoring service.

    Three endpoints sharing one pod:

    - :meth:`rag`    -- score a response against retrieval context
    - :meth:`code`   -- score an agentic-coding turn (composite + AST + NLI)
    - :meth:`rollup` -- aggregate N per-turn outputs into a session scoreboard

    ``scoring_mode`` is pinned server-side by the URL; clients never send it.
    """

    def __init__(self, client: BaseSyncClient) -> None:
        super().__init__(client)

    # ---------------- RAG lane ----------------

    @overload
    def rag(
        self,
        *,
        response_text: str,
        query_text: str | None = None,
        raw_context: str | None = None,
        chunk_ids: list[str | int] | None = None,
        support_units: list[SupportUnitInput | dict[str, Any]] | None = None,
        primary_metric: PrimaryMetric | None = None,
        evidence_limit: int | None = None,
        coverage_threshold: float | None = None,
        segmentation_mode: SegmentationMode | None = None,
        raw_context_chunk_tokens: int | None = None,
        response_chunk_tokens: int | None = None,
        attribution_mode: AttributionMode | None = None,
        include_triangular_diagnostics: bool | None = None,
        heatmap_format: HeatmapFormat | None = None,
        verification_samples: list[str] | None = None,
        content_type: str | None = None,
        risk_band_stratum: str | None = None,
        model: str | None = None,
        query_prompt_name: str | None = None,
        document_prompt_name: str | None = None,
        debug_dense_matrices: bool | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        verbose: bool | None = None,
        return_job: Literal[False] = False,
    ) -> TraceRagResponse: ...

    @overload
    def rag(
        self,
        *,
        response_text: str,
        query_text: str | None = None,
        raw_context: str | None = None,
        chunk_ids: list[str | int] | None = None,
        support_units: list[SupportUnitInput | dict[str, Any]] | None = None,
        primary_metric: PrimaryMetric | None = None,
        evidence_limit: int | None = None,
        coverage_threshold: float | None = None,
        segmentation_mode: SegmentationMode | None = None,
        raw_context_chunk_tokens: int | None = None,
        response_chunk_tokens: int | None = None,
        attribution_mode: AttributionMode | None = None,
        include_triangular_diagnostics: bool | None = None,
        heatmap_format: HeatmapFormat | None = None,
        verification_samples: list[str] | None = None,
        content_type: str | None = None,
        risk_band_stratum: str | None = None,
        model: str | None = None,
        query_prompt_name: str | None = None,
        document_prompt_name: str | None = None,
        debug_dense_matrices: bool | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        verbose: bool | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def rag(
        self,
        *,
        response_text: str,
        query_text: str | None = None,
        raw_context: str | None = None,
        chunk_ids: list[str | int] | None = None,
        support_units: list[SupportUnitInput | dict[str, Any]] | None = None,
        primary_metric: PrimaryMetric | None = None,
        evidence_limit: int | None = None,
        coverage_threshold: float | None = None,
        segmentation_mode: SegmentationMode | None = None,
        raw_context_chunk_tokens: int | None = None,
        response_chunk_tokens: int | None = None,
        attribution_mode: AttributionMode | None = None,
        include_triangular_diagnostics: bool | None = None,
        heatmap_format: HeatmapFormat | None = None,
        verification_samples: list[str] | None = None,
        content_type: str | None = None,
        risk_band_stratum: str | None = None,
        model: str | None = None,
        query_prompt_name: str | None = None,
        document_prompt_name: str | None = None,
        debug_dense_matrices: bool | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        verbose: bool | None = None,
        return_job: bool = False,
    ) -> Union[TraceRagResponse, JobSubmittedResponse]:
        """Score a response for groundedness against retrieval context.

        At least one of ``raw_context``, ``chunk_ids``, or ``support_units``
        must be supplied (enforced client-side before the HTTP call).

        Pricing: $0.008 per request, quantized per 32k context tokens.
        """
        _require_response_text(response_text)
        _require_premise_lane(
            raw_context=raw_context,
            chunk_ids=chunk_ids,
            support_units=support_units,
        )

        body = self._build_request_body(
            response_text=response_text,
            query_text=query_text,
            raw_context=raw_context,
            chunk_ids=chunk_ids,
            support_units=_normalize_support_units(support_units),
            primary_metric=primary_metric,
            evidence_limit=evidence_limit,
            coverage_threshold=coverage_threshold,
            segmentation_mode=segmentation_mode,
            raw_context_chunk_tokens=raw_context_chunk_tokens,
            response_chunk_tokens=response_chunk_tokens,
            attribution_mode=attribution_mode,
            include_triangular_diagnostics=include_triangular_diagnostics,
            heatmap_format=heatmap_format,
            verification_samples=verification_samples,
            content_type=content_type,
            risk_band_stratum=risk_band_stratum,
            model=model,
            query_prompt_name=query_prompt_name,
            document_prompt_name=document_prompt_name,
            debug_dense_matrices=debug_dense_matrices,
            session_id=session_id,
            verbose=verbose,
            request_id=request_id,
            return_job=return_job,
        )

        response = self._client.post("/api/v1/trace/rag", json=body)

        result: TraceRagResponse | JobSubmittedResponse
        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = TraceRagResponse.model_validate(response.data)

        return self._inject_metadata(result, response)

    # ---------------- Code lane ----------------

    @overload
    def code(
        self,
        *,
        response_text: str,
        query_text: str | None = None,
        raw_context: str | None = None,
        chunk_ids: list[str | int] | None = None,
        support_units: list[SupportUnitInput | dict[str, Any]] | None = None,
        response_language_hint: ResponseLanguageHint | None = None,
        emit_chunk_ownership: bool | None = None,
        session_state: SessionState | dict[str, Any] | None = None,
        primary_metric: PrimaryMetric | None = None,
        evidence_limit: int | None = None,
        coverage_threshold: float | None = None,
        segmentation_mode: SegmentationMode | None = None,
        raw_context_chunk_tokens: int | None = None,
        response_chunk_tokens: int | None = None,
        attribution_mode: AttributionMode | None = None,
        include_triangular_diagnostics: bool | None = None,
        heatmap_format: HeatmapFormat | None = None,
        verification_samples: list[str] | None = None,
        content_type: str | None = None,
        risk_band_stratum: str | None = None,
        model: str | None = None,
        query_prompt_name: str | None = None,
        document_prompt_name: str | None = None,
        debug_dense_matrices: bool | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        verbose: bool | None = None,
        return_job: Literal[False] = False,
    ) -> TraceCodeResponse: ...

    @overload
    def code(
        self,
        *,
        response_text: str,
        query_text: str | None = None,
        raw_context: str | None = None,
        chunk_ids: list[str | int] | None = None,
        support_units: list[SupportUnitInput | dict[str, Any]] | None = None,
        response_language_hint: ResponseLanguageHint | None = None,
        emit_chunk_ownership: bool | None = None,
        session_state: SessionState | dict[str, Any] | None = None,
        primary_metric: PrimaryMetric | None = None,
        evidence_limit: int | None = None,
        coverage_threshold: float | None = None,
        segmentation_mode: SegmentationMode | None = None,
        raw_context_chunk_tokens: int | None = None,
        response_chunk_tokens: int | None = None,
        attribution_mode: AttributionMode | None = None,
        include_triangular_diagnostics: bool | None = None,
        heatmap_format: HeatmapFormat | None = None,
        verification_samples: list[str] | None = None,
        content_type: str | None = None,
        risk_band_stratum: str | None = None,
        model: str | None = None,
        query_prompt_name: str | None = None,
        document_prompt_name: str | None = None,
        debug_dense_matrices: bool | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        verbose: bool | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def code(
        self,
        *,
        response_text: str,
        query_text: str | None = None,
        raw_context: str | None = None,
        chunk_ids: list[str | int] | None = None,
        support_units: list[SupportUnitInput | dict[str, Any]] | None = None,
        response_language_hint: ResponseLanguageHint | None = None,
        emit_chunk_ownership: bool | None = None,
        session_state: SessionState | dict[str, Any] | None = None,
        primary_metric: PrimaryMetric | None = None,
        evidence_limit: int | None = None,
        coverage_threshold: float | None = None,
        segmentation_mode: SegmentationMode | None = None,
        raw_context_chunk_tokens: int | None = None,
        response_chunk_tokens: int | None = None,
        attribution_mode: AttributionMode | None = None,
        include_triangular_diagnostics: bool | None = None,
        heatmap_format: HeatmapFormat | None = None,
        verification_samples: list[str] | None = None,
        content_type: str | None = None,
        risk_band_stratum: str | None = None,
        model: str | None = None,
        query_prompt_name: str | None = None,
        document_prompt_name: str | None = None,
        debug_dense_matrices: bool | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        verbose: bool | None = None,
        return_job: bool = False,
    ) -> Union[TraceCodeResponse, JobSubmittedResponse]:
        """Score an agentic-coding turn for phantom hallucinations.

        Multi-turn usage: pass the previous response's ``next_session_state``
        into the next call via ``session_state=prev.next_session_state``.

        Pricing: $2.00 per 1M aggregate tokens.
        """
        _require_response_text(response_text)
        _require_premise_lane(
            raw_context=raw_context,
            chunk_ids=chunk_ids,
            support_units=support_units,
        )

        body = self._build_request_body(
            response_text=response_text,
            query_text=query_text,
            raw_context=raw_context,
            chunk_ids=chunk_ids,
            support_units=_normalize_support_units(support_units),
            response_language_hint=response_language_hint,
            emit_chunk_ownership=emit_chunk_ownership,
            session_state=_normalize_session_state(session_state),
            primary_metric=primary_metric,
            evidence_limit=evidence_limit,
            coverage_threshold=coverage_threshold,
            segmentation_mode=segmentation_mode,
            raw_context_chunk_tokens=raw_context_chunk_tokens,
            response_chunk_tokens=response_chunk_tokens,
            attribution_mode=attribution_mode,
            include_triangular_diagnostics=include_triangular_diagnostics,
            heatmap_format=heatmap_format,
            verification_samples=verification_samples,
            content_type=content_type,
            risk_band_stratum=risk_band_stratum,
            model=model,
            query_prompt_name=query_prompt_name,
            document_prompt_name=document_prompt_name,
            debug_dense_matrices=debug_dense_matrices,
            session_id=session_id,
            verbose=verbose,
            request_id=request_id,
            return_job=return_job,
        )

        response = self._client.post("/api/v1/trace/code", json=body)

        result: TraceCodeResponse | JobSubmittedResponse
        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = TraceCodeResponse.model_validate(response.data)

        return self._inject_metadata(result, response)

    # ---------------- Rollup ----------------

    def rollup(
        self,
        *,
        turns: list[TraceResponse | dict[str, Any]],
        session_id: str | None = None,
        heatmap_format: HeatmapFormat | None = None,
        request_id: str | None = None,
    ) -> TraceRollupResponse:
        """Aggregate N per-turn outputs into a session scoreboard.

        Stateless, CPU-only, sub-ms on the pod. Safe to call on every
        keystroke for live scoreboards.

        Pricing: $0.001 flat per request.
        """
        if not turns:
            raise ValueError("`turns` must be a non-empty list.")

        body = self._build_request_body(
            turns=_normalize_turns(turns),
            session_id=session_id,
            heatmap_format=heatmap_format,
            request_id=request_id,
        )

        response = self._client.post("/api/v1/trace/rollup", json=body)
        result = TraceRollupResponse.model_validate(response.data)
        return self._inject_metadata(result, response)


# ---------------------------------------------------------------------------
# Async resource
# ---------------------------------------------------------------------------


class AsyncTrace(AsyncResource):
    """Async Trace service — see :class:`Trace` for full documentation."""

    def __init__(self, client: BaseAsyncClient) -> None:
        super().__init__(client)

    # ---------------- RAG lane ----------------

    @overload
    async def rag(
        self,
        *,
        response_text: str,
        query_text: str | None = None,
        raw_context: str | None = None,
        chunk_ids: list[str | int] | None = None,
        support_units: list[SupportUnitInput | dict[str, Any]] | None = None,
        primary_metric: PrimaryMetric | None = None,
        evidence_limit: int | None = None,
        coverage_threshold: float | None = None,
        segmentation_mode: SegmentationMode | None = None,
        raw_context_chunk_tokens: int | None = None,
        response_chunk_tokens: int | None = None,
        attribution_mode: AttributionMode | None = None,
        include_triangular_diagnostics: bool | None = None,
        heatmap_format: HeatmapFormat | None = None,
        verification_samples: list[str] | None = None,
        content_type: str | None = None,
        risk_band_stratum: str | None = None,
        model: str | None = None,
        query_prompt_name: str | None = None,
        document_prompt_name: str | None = None,
        debug_dense_matrices: bool | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        verbose: bool | None = None,
        return_job: Literal[False] = False,
    ) -> TraceRagResponse: ...

    @overload
    async def rag(
        self,
        *,
        response_text: str,
        query_text: str | None = None,
        raw_context: str | None = None,
        chunk_ids: list[str | int] | None = None,
        support_units: list[SupportUnitInput | dict[str, Any]] | None = None,
        primary_metric: PrimaryMetric | None = None,
        evidence_limit: int | None = None,
        coverage_threshold: float | None = None,
        segmentation_mode: SegmentationMode | None = None,
        raw_context_chunk_tokens: int | None = None,
        response_chunk_tokens: int | None = None,
        attribution_mode: AttributionMode | None = None,
        include_triangular_diagnostics: bool | None = None,
        heatmap_format: HeatmapFormat | None = None,
        verification_samples: list[str] | None = None,
        content_type: str | None = None,
        risk_band_stratum: str | None = None,
        model: str | None = None,
        query_prompt_name: str | None = None,
        document_prompt_name: str | None = None,
        debug_dense_matrices: bool | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        verbose: bool | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def rag(
        self,
        *,
        response_text: str,
        query_text: str | None = None,
        raw_context: str | None = None,
        chunk_ids: list[str | int] | None = None,
        support_units: list[SupportUnitInput | dict[str, Any]] | None = None,
        primary_metric: PrimaryMetric | None = None,
        evidence_limit: int | None = None,
        coverage_threshold: float | None = None,
        segmentation_mode: SegmentationMode | None = None,
        raw_context_chunk_tokens: int | None = None,
        response_chunk_tokens: int | None = None,
        attribution_mode: AttributionMode | None = None,
        include_triangular_diagnostics: bool | None = None,
        heatmap_format: HeatmapFormat | None = None,
        verification_samples: list[str] | None = None,
        content_type: str | None = None,
        risk_band_stratum: str | None = None,
        model: str | None = None,
        query_prompt_name: str | None = None,
        document_prompt_name: str | None = None,
        debug_dense_matrices: bool | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        verbose: bool | None = None,
        return_job: bool = False,
    ) -> Union[TraceRagResponse, JobSubmittedResponse]:
        """Score a response for groundedness (async variant)."""
        _require_response_text(response_text)
        _require_premise_lane(
            raw_context=raw_context,
            chunk_ids=chunk_ids,
            support_units=support_units,
        )

        body = self._build_request_body(
            response_text=response_text,
            query_text=query_text,
            raw_context=raw_context,
            chunk_ids=chunk_ids,
            support_units=_normalize_support_units(support_units),
            primary_metric=primary_metric,
            evidence_limit=evidence_limit,
            coverage_threshold=coverage_threshold,
            segmentation_mode=segmentation_mode,
            raw_context_chunk_tokens=raw_context_chunk_tokens,
            response_chunk_tokens=response_chunk_tokens,
            attribution_mode=attribution_mode,
            include_triangular_diagnostics=include_triangular_diagnostics,
            heatmap_format=heatmap_format,
            verification_samples=verification_samples,
            content_type=content_type,
            risk_band_stratum=risk_band_stratum,
            model=model,
            query_prompt_name=query_prompt_name,
            document_prompt_name=document_prompt_name,
            debug_dense_matrices=debug_dense_matrices,
            session_id=session_id,
            verbose=verbose,
            request_id=request_id,
            return_job=return_job,
        )

        response = await self._client.post("/api/v1/trace/rag", json=body)

        result: TraceRagResponse | JobSubmittedResponse
        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = TraceRagResponse.model_validate(response.data)

        return self._inject_metadata(result, response)

    # ---------------- Code lane ----------------

    @overload
    async def code(
        self,
        *,
        response_text: str,
        query_text: str | None = None,
        raw_context: str | None = None,
        chunk_ids: list[str | int] | None = None,
        support_units: list[SupportUnitInput | dict[str, Any]] | None = None,
        response_language_hint: ResponseLanguageHint | None = None,
        emit_chunk_ownership: bool | None = None,
        session_state: SessionState | dict[str, Any] | None = None,
        primary_metric: PrimaryMetric | None = None,
        evidence_limit: int | None = None,
        coverage_threshold: float | None = None,
        segmentation_mode: SegmentationMode | None = None,
        raw_context_chunk_tokens: int | None = None,
        response_chunk_tokens: int | None = None,
        attribution_mode: AttributionMode | None = None,
        include_triangular_diagnostics: bool | None = None,
        heatmap_format: HeatmapFormat | None = None,
        verification_samples: list[str] | None = None,
        content_type: str | None = None,
        risk_band_stratum: str | None = None,
        model: str | None = None,
        query_prompt_name: str | None = None,
        document_prompt_name: str | None = None,
        debug_dense_matrices: bool | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        verbose: bool | None = None,
        return_job: Literal[False] = False,
    ) -> TraceCodeResponse: ...

    @overload
    async def code(
        self,
        *,
        response_text: str,
        query_text: str | None = None,
        raw_context: str | None = None,
        chunk_ids: list[str | int] | None = None,
        support_units: list[SupportUnitInput | dict[str, Any]] | None = None,
        response_language_hint: ResponseLanguageHint | None = None,
        emit_chunk_ownership: bool | None = None,
        session_state: SessionState | dict[str, Any] | None = None,
        primary_metric: PrimaryMetric | None = None,
        evidence_limit: int | None = None,
        coverage_threshold: float | None = None,
        segmentation_mode: SegmentationMode | None = None,
        raw_context_chunk_tokens: int | None = None,
        response_chunk_tokens: int | None = None,
        attribution_mode: AttributionMode | None = None,
        include_triangular_diagnostics: bool | None = None,
        heatmap_format: HeatmapFormat | None = None,
        verification_samples: list[str] | None = None,
        content_type: str | None = None,
        risk_band_stratum: str | None = None,
        model: str | None = None,
        query_prompt_name: str | None = None,
        document_prompt_name: str | None = None,
        debug_dense_matrices: bool | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        verbose: bool | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def code(
        self,
        *,
        response_text: str,
        query_text: str | None = None,
        raw_context: str | None = None,
        chunk_ids: list[str | int] | None = None,
        support_units: list[SupportUnitInput | dict[str, Any]] | None = None,
        response_language_hint: ResponseLanguageHint | None = None,
        emit_chunk_ownership: bool | None = None,
        session_state: SessionState | dict[str, Any] | None = None,
        primary_metric: PrimaryMetric | None = None,
        evidence_limit: int | None = None,
        coverage_threshold: float | None = None,
        segmentation_mode: SegmentationMode | None = None,
        raw_context_chunk_tokens: int | None = None,
        response_chunk_tokens: int | None = None,
        attribution_mode: AttributionMode | None = None,
        include_triangular_diagnostics: bool | None = None,
        heatmap_format: HeatmapFormat | None = None,
        verification_samples: list[str] | None = None,
        content_type: str | None = None,
        risk_band_stratum: str | None = None,
        model: str | None = None,
        query_prompt_name: str | None = None,
        document_prompt_name: str | None = None,
        debug_dense_matrices: bool | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        verbose: bool | None = None,
        return_job: bool = False,
    ) -> Union[TraceCodeResponse, JobSubmittedResponse]:
        """Score an agentic-coding turn for phantom hallucinations (async)."""
        _require_response_text(response_text)
        _require_premise_lane(
            raw_context=raw_context,
            chunk_ids=chunk_ids,
            support_units=support_units,
        )

        body = self._build_request_body(
            response_text=response_text,
            query_text=query_text,
            raw_context=raw_context,
            chunk_ids=chunk_ids,
            support_units=_normalize_support_units(support_units),
            response_language_hint=response_language_hint,
            emit_chunk_ownership=emit_chunk_ownership,
            session_state=_normalize_session_state(session_state),
            primary_metric=primary_metric,
            evidence_limit=evidence_limit,
            coverage_threshold=coverage_threshold,
            segmentation_mode=segmentation_mode,
            raw_context_chunk_tokens=raw_context_chunk_tokens,
            response_chunk_tokens=response_chunk_tokens,
            attribution_mode=attribution_mode,
            include_triangular_diagnostics=include_triangular_diagnostics,
            heatmap_format=heatmap_format,
            verification_samples=verification_samples,
            content_type=content_type,
            risk_band_stratum=risk_band_stratum,
            model=model,
            query_prompt_name=query_prompt_name,
            document_prompt_name=document_prompt_name,
            debug_dense_matrices=debug_dense_matrices,
            session_id=session_id,
            verbose=verbose,
            request_id=request_id,
            return_job=return_job,
        )

        response = await self._client.post("/api/v1/trace/code", json=body)

        result: TraceCodeResponse | JobSubmittedResponse
        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = TraceCodeResponse.model_validate(response.data)

        return self._inject_metadata(result, response)

    # ---------------- Rollup ----------------

    async def rollup(
        self,
        *,
        turns: list[TraceResponse | dict[str, Any]],
        session_id: str | None = None,
        heatmap_format: HeatmapFormat | None = None,
        request_id: str | None = None,
    ) -> TraceRollupResponse:
        """Aggregate N per-turn outputs into a session scoreboard (async)."""
        if not turns:
            raise ValueError("`turns` must be a non-empty list.")

        body = self._build_request_body(
            turns=_normalize_turns(turns),
            session_id=session_id,
            heatmap_format=heatmap_format,
            request_id=request_id,
        )

        response = await self._client.post("/api/v1/trace/rollup", json=body)
        result = TraceRollupResponse.model_validate(response.data)
        return self._inject_metadata(result, response)
