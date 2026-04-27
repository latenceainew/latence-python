"""Pydantic models for the Trace (groundedness + phantom scoring) service.

The gateway exposes three endpoints under ``/api/v1/trace/*``:

- ``/rag``    -- groundedness scoring against retrieval context
- ``/code``   -- phantom-hallucination scoring for agentic code turns
- ``/rollup`` -- stateless session-level aggregation over N per-turn outputs

Scoring mode is pinned server-side by the URL; clients must not send a
``scoring_mode`` field in the request body. Responses are returned flat
(the gateway's ``map_response`` drops ``None`` keys so code-lane-only
fields are absent on RAG responses and vice versa).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .common import BaseResponse

# ---------------------------------------------------------------------------
# Enum aliases (match gateway/src/services/schemas/trace.py exactly)
# ---------------------------------------------------------------------------

AttributionMode = Literal["closed_book", "open_domain"]
"""Whether the scorer may use world knowledge outside the provided context."""

SegmentationMode = Literal["sentence", "sentence_packed", "paragraph"]
"""How the response is split into premises for NLI."""

PrimaryMetric = Literal["reverse_context", "triangular"]
"""Headline metric selector accepted on the request (input-side enum).

The **response** ``primary_metric`` field is ``str`` because the pod may
echo back a derived name (e.g. ``"groundedness_v2"``) rather than the
requested mode. See ``latence-trace/runpod/handler.py`` for the mapping.
"""

HeatmapFormat = Literal["none", "data", "html"]
"""Heatmap emission mode. ``data`` = structured JSON, ``html`` = <div> fragment."""

TraceProfile = Literal["standard", "quality"]
"""Hosted Trace billing/runtime profile. ``quality`` bills at 2x standard."""

ResponseLanguageHint = Literal[
    "python",
    "py",
    "typescript",
    "ts",
    "tsx",
    "javascript",
    "js",
    "jsx",
    "go",
    "golang",
    "rust",
    "rs",
]
"""Source-language hint for the code lane's AST extractor."""

ScoringMode = Literal["rag", "code"]
"""Lane identifier echoed back in the response (never sent by the client)."""


# ---------------------------------------------------------------------------
# Input building blocks
# ---------------------------------------------------------------------------


class SupportUnitInput(BaseModel):
    """A single support unit the response is expected to cite.

    Support units are an alternative to ``raw_context`` / ``chunk_ids`` --
    they carry structured provenance (speaker, timestamp, source id) which
    the scorer will propagate into ``support_units_usage`` on the response.
    """

    model_config = ConfigDict(extra="allow")

    text: str = Field(description="The unit's content; what the response may cite.")
    source_id: str | None = Field(default=None, description="Stable identifier for the source.")
    speaker: str | None = Field(
        default=None, description="Speaker / author label (transcripts, chats)."
    )
    timestamp: str | None = Field(default=None, description="Free-form timestamp for the unit.")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Arbitrary passthrough metadata."
    )


class SessionState(BaseModel):
    """Opaque session-state payload carried between code-lane turns.

    Clients never construct this by hand -- they receive a ``SessionState``
    on ``TraceCodeResponse.next_session_state`` and pass it back into the
    next ``client.experimental.trace.code(...)`` call via ``session_state``.
    ``extra="allow"`` lets the schema evolve pod-side without breaking
    clients.
    """

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Response building blocks
# ---------------------------------------------------------------------------


class SupportUnitUsage(BaseModel):
    """Per-unit scoring verdict returned on the response."""

    model_config = ConfigDict(extra="allow")

    index: int | None = Field(default=None, description="Position of the unit in the request list.")
    source_id: str | None = Field(
        default=None, description="Echoed ``source_id`` from the request."
    )
    usage_state: str | None = Field(
        default=None, description="``used`` / ``unused`` / ``uncertain``."
    )
    coverage_score: float | None = Field(default=None, description="Per-unit coverage score.")


class SupportUnitsUsageSummary(BaseModel):
    """Aggregate counts of ``used`` / ``unused`` / ``uncertain`` support units."""

    model_config = ConfigDict(extra="allow")

    used: int | None = Field(default=None)
    unused: int | None = Field(default=None)
    uncertain: int | None = Field(default=None)


class CodeLaneDiagnostics(BaseModel):
    """Code-lane-only composite / AST / NLI diagnostics.

    Shape is intentionally permissive (``extra='allow'``): the pod emits
    whichever sub-fields are relevant for the current turn (composite
    score, phantom probability, AST drift, dead-weight flags, etc.).
    """

    model_config = ConfigDict(extra="allow")


class SessionSignals(BaseModel):
    """Session-derived signals emitted alongside a code-lane response."""

    model_config = ConfigDict(extra="allow")

    total_turns: int | None = Field(default=None)
    ema_groundedness: float | None = Field(default=None)
    groundedness_drift: float | None = Field(default=None)
    drift_z_score: float | None = Field(default=None)
    cascade_density: float | None = Field(default=None)
    phantom_rate: float | None = Field(default=None)
    red_streak: int | None = Field(default=None)
    dead_weight_streak: int | None = Field(default=None)
    dead_file_candidates: list[Any] | None = Field(default=None)
    recommendation: str | None = Field(default=None)


class FileAttribution(BaseModel):
    """Per-file owner share and reason-code histogram (lane-neutral GTM surface)."""

    model_config = ConfigDict(extra="allow")


class Heatmap(BaseModel):
    """Structured heatmap payload for UI renderers."""

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Scoring responses
# ---------------------------------------------------------------------------


class TraceResponse(BaseResponse):
    """Fields present on both RAG and code scoring responses.

    The gateway flattens the pod's ``_compact_response`` onto the top
    level and drops keys whose value is ``None``; optional fields below
    may therefore be absent on any given call.
    """

    score: float | None = Field(default=None, description="Top-level groundedness score (0-1).")
    # ``str`` (not the input Literal) on purpose -- the pod can emit
    # derived metric names such as ``"groundedness_v2"`` that are not in
    # the request-side ``PrimaryMetric`` enum.
    primary_metric: str | None = Field(
        default=None,
        description="Metric backing ``score`` (echoed by the pod; may be a derived name).",
    )
    band: str | None = Field(
        default=None, description="Risk band label (e.g. green/amber/red/unknown)."
    )
    structured_score: dict[str, Any] | float | None = Field(
        default=None, description="Per-component score breakdown or scalar structured score."
    )
    nli_aggregate: float | None = Field(default=None, description="Aggregate NLI entailment score.")
    context_coverage_ratio: float | None = Field(
        default=None, description="Fraction of the response grounded in context."
    )
    context_usage_ratio: float | None = Field(
        default=None, description="Fraction of context actually used."
    )
    context_unused_ratio: float | None = Field(
        default=None, description="Fraction of context left unused."
    )
    context_uncertain_ratio: float | None = Field(
        default=None, description="Fraction with uncertain grounding."
    )
    support_units_usage: SupportUnitsUsageSummary | None = Field(
        default=None, description="Aggregate per-state counts."
    )
    support_units: list[SupportUnitUsage] | None = Field(
        default=None, description="Per-unit verdicts."
    )
    latency_ms: float | None = Field(default=None, description="Pod-side scoring latency.")
    scoring_mode: ScoringMode | None = Field(
        default=None, description="Lane echoed back by the pod."
    )
    session_id: str | None = Field(
        default=None, description="Session id echoed back when supplied."
    )
    version: str | None = Field(default=None, description="Pod handler version tag.")
    reason: str | None = Field(default=None, description="Human-readable reason for the band.")
    warnings: list[str] | None = Field(default=None, description="Scorer warnings (non-fatal).")
    file_attribution: FileAttribution | None = Field(
        default=None, description="Per-file owner share + reason codes."
    )
    heatmap: Heatmap | None = Field(default=None, description="Structured heatmap payload.")
    heatmap_html: str | None = Field(
        default=None, description="Self-contained <div> fragment (when heatmap_format='html')."
    )


class TraceRagResponse(TraceResponse):
    """Response from ``client.experimental.trace.rag(...)``."""

    scoring_mode: Literal["rag"] | None = Field(default=None, description="Always ``'rag'``.")  # type: ignore[assignment]


class TraceCodeResponse(TraceResponse):
    """Response from ``client.experimental.trace.code(...)``.

    Adds code-lane-only diagnostics, session state, and session signals.
    """

    scoring_mode: Literal["code"] | None = Field(default=None, description="Always ``'code'``.")  # type: ignore[assignment]
    code_lane: CodeLaneDiagnostics | None = Field(
        default=None,
        description="Composite / AST / NLI diagnostics for the turn.",
    )
    next_session_state: SessionState | None = Field(
        default=None,
        description="Opaque session state to pass into the next ``code()`` call.",
    )
    session_signals: SessionSignals | None = Field(
        default=None,
        description="Session-derived signals (EMA, drift, phantom rate, recommendation).",
    )


# ---------------------------------------------------------------------------
# Rollup
# ---------------------------------------------------------------------------


class TraceRollupResponse(BaseResponse):
    """Response from ``client.experimental.trace.rollup(...)``.

    Mirrors the gateway's ``/rollup`` ``response_extract``.
    """

    scoring_mode: str | None = Field(
        default=None, description="Always ``'rollup'`` on this endpoint."
    )
    turns_processed: int | None = Field(
        default=None, description="Number of per-turn outputs aggregated."
    )
    noise_pct: float | None = Field(default=None, description="Fraction of turns flagged as noise.")
    model_drift_pct: float | None = Field(
        default=None, description="Fraction of turns exhibiting model drift."
    )
    retrieval_waste_pct: float | None = Field(
        default=None, description="Fraction of retrieved context left unused."
    )
    reason_code_histogram: dict[str, int] | None = Field(
        default=None, description="Count of each reason code over the window."
    )
    recommendations: list[str] | None = Field(
        default=None, description="Session-level recommendations."
    )
    risk_band_trail: list[str] | None = Field(
        default=None, description="Risk band per turn, chronological."
    )
    drift_trend: dict[str, Any] | None = Field(
        default=None,
        description="Drift trajectory summary over the window (``{last, max, mean, min, ...}``).",
    )
    top_dead_files: list[Any] | None = Field(
        default=None, description="Files consistently marked dead-weight."
    )
    heatmap: Heatmap | None = Field(default=None, description="Session-level heatmap payload.")
    heatmap_html: str | None = Field(
        default=None, description="Self-contained <div> fragment (when heatmap_format='html')."
    )
    session_id: str | None = Field(default=None, description="Echoed session id.")
    version: str | None = Field(default=None, description="Rollup handler version tag.")
