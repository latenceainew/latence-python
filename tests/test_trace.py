"""Tests for the Trace resource (groundedness + phantom scoring).

Covers:
- Pydantic model parsing for RAG / code / rollup responses
- Request-body shape per lane (no ``scoring_mode``, correct URL, key passthrough)
- Client-side validation (empty response_text, missing premise lane, empty turns)
- Session-state round-trip (turn N's ``next_session_state`` -> turn N+1)
- Default/None pruning (``_build_request_body`` drops ``None`` keys)
- Experimental namespace registration
"""

from __future__ import annotations

from typing import Any

import pytest

from latence._base import APIResponse, ResponseMetadata
from latence._models.trace import (
    SessionState,
    SupportUnitInput,
    TraceCodeResponse,
    TraceRagResponse,
    TraceRollupResponse,
)
from latence.resources.experimental import (
    AsyncExperimentalNamespace,
    ExperimentalNamespace,
)
from latence.resources.trace import AsyncTrace, Trace


# ---------------------------------------------------------------------------
# Fake clients (mirror tests/test_refinery.py)
# ---------------------------------------------------------------------------


class _FakeSyncClient:
    def __init__(self, data: dict | None = None) -> None:
        self._data = data or {}
        self.calls: list[tuple[str, str, dict | None]] = []

    def post(self, path: str, json: dict | None = None) -> APIResponse:
        self.calls.append(("POST", path, json))
        return APIResponse(
            data=self._data,
            metadata=ResponseMetadata(request_id="req_test"),
            status_code=200,
        )

    def get(self, path: str) -> APIResponse:
        self.calls.append(("GET", path, None))
        return APIResponse(
            data=self._data,
            metadata=ResponseMetadata(request_id="req_test"),
            status_code=200,
        )

    base_url = "https://api.latence.ai"
    max_retries = 2


class _FakeAsyncClient:
    def __init__(self, data: dict | None = None) -> None:
        self._data = data or {}
        self.calls: list[tuple[str, str, dict | None]] = []

    async def post(self, path: str, json: dict | None = None) -> APIResponse:
        self.calls.append(("POST", path, json))
        return APIResponse(
            data=self._data,
            metadata=ResponseMetadata(request_id="req_test"),
            status_code=200,
        )

    async def get(self, path: str) -> APIResponse:
        self.calls.append(("GET", path, None))
        return APIResponse(
            data=self._data,
            metadata=ResponseMetadata(request_id="req_test"),
            status_code=200,
        )

    base_url = "https://api.latence.ai"
    max_retries = 2


# ---------------------------------------------------------------------------
# Realistic canned responses (shape matches gateway response_extract)
# ---------------------------------------------------------------------------


def _rag_response_fixture() -> dict[str, Any]:
    return {
        "success": True,
        "score": 0.87,
        "primary_metric": "reverse_context",
        "band": "green",
        "nli_aggregate": 0.91,
        "context_coverage_ratio": 0.92,
        "context_usage_ratio": 0.75,
        "context_unused_ratio": 0.20,
        "context_uncertain_ratio": 0.05,
        "support_units_usage": {"used": 3, "unused": 1, "uncertain": 0},
        "support_units": [
            {"index": 0, "source_id": "doc-1", "usage_state": "used", "coverage_score": 0.95},
            {"index": 1, "source_id": "doc-2", "usage_state": "used", "coverage_score": 0.80},
        ],
        "latency_ms": 142.7,
        "scoring_mode": "rag",
        "version": "0.6.3",
        "reason": "strong coverage, no entity swaps",
    }


def _code_response_fixture() -> dict[str, Any]:
    return {
        "success": True,
        "score": 0.62,
        "primary_metric": "reverse_context",
        "band": "amber",
        "nli_aggregate": 0.71,
        "context_coverage_ratio": 0.68,
        "context_usage_ratio": 0.54,
        "context_unused_ratio": 0.40,
        "context_uncertain_ratio": 0.06,
        "latency_ms": 98.1,
        "scoring_mode": "code",
        "session_id": "sess_abc",
        "version": "0.6.3",
        "code_lane": {
            "composite_phantom_score": 0.19,
            "grounded_ast_false_positive": False,
            "dead_weight_ratio": 0.12,
        },
        "next_session_state": {
            "turn_count": 1,
            "ema_groundedness": 0.62,
            "history": [{"turn": 0, "score": 0.62, "band": "amber"}],
        },
        "session_signals": {
            "total_turns": 1,
            "ema_groundedness": 0.62,
            "groundedness_drift": 0.0,
            "phantom_rate": 0.0,
            "recommendation": "continue",
        },
    }


def _rollup_response_fixture() -> dict[str, Any]:
    return {
        "success": True,
        "scoring_mode": "rollup",
        "turns_processed": 4,
        "noise_pct": 0.25,
        "model_drift_pct": 0.10,
        "retrieval_waste_pct": 0.33,
        "reason_code_histogram": {"entity_swap": 2, "negation": 1, "ok": 1},
        "recommendations": ["re_anchor_on_turn_3"],
        "risk_band_trail": ["green", "green", "amber", "red"],
        "drift_trend": {"last": 0.14, "max": 0.14, "mean": 0.06, "min": 0.0},
        "top_dead_files": [{"file": "utils.py", "dead_weight_ratio": 0.72}],
        "session_id": "sess_abc",
        "version": "0.6.3",
    }


# ===========================================================================
# Model parsing
# ===========================================================================


class TestModelParsing:
    def test_rag_response_parses(self):
        r = TraceRagResponse.model_validate(_rag_response_fixture())
        assert r.success is True
        assert r.score == 0.87
        assert r.band == "green"
        assert r.primary_metric == "reverse_context"
        assert r.scoring_mode == "rag"
        assert r.support_units_usage is not None
        assert r.support_units_usage.used == 3
        assert r.support_units is not None and len(r.support_units) == 2
        assert r.support_units[0].source_id == "doc-1"

    def test_code_response_parses_with_session(self):
        r = TraceCodeResponse.model_validate(_code_response_fixture())
        assert r.scoring_mode == "code"
        assert r.session_id == "sess_abc"
        assert r.code_lane is not None
        # extra="allow" -> composite fields preserved as attributes
        assert getattr(r.code_lane, "composite_phantom_score") == 0.19
        assert r.next_session_state is not None
        assert getattr(r.next_session_state, "turn_count") == 1
        assert r.session_signals is not None
        assert r.session_signals.recommendation == "continue"

    def test_rollup_response_parses(self):
        r = TraceRollupResponse.model_validate(_rollup_response_fixture())
        assert r.turns_processed == 4
        assert r.noise_pct == 0.25
        assert r.risk_band_trail == ["green", "green", "amber", "red"]
        assert r.reason_code_histogram == {"entity_swap": 2, "negation": 1, "ok": 1}
        assert r.recommendations == ["re_anchor_on_turn_3"]
        assert r.drift_trend == {"last": 0.14, "max": 0.14, "mean": 0.06, "min": 0.0}

    def test_real_prod_rollup_response_parses(self):
        """Regression: production ``/rollup`` returns ``drift_trend`` as a dict.

        The pod emits ``drift_trend`` as a summary object
        (``{"last", "max", "mean", "min"}``), not a list of floats
        -- see ``latence-trace/scripts/bench_runpod_360.py`` and
        ``latence-trace/docs/rollup.md``.
        """
        data = {
            "success": True,
            "scoring_mode": "rollup",
            "turns_processed": 2,
            "noise_pct": 0,
            "model_drift_pct": 1,
            "retrieval_waste_pct": 0,
            "reason_code_histogram": {},
            "recommendations": ["continue", "re_anchor"],
            "risk_band_trail": ["green", "red"],
            "drift_trend": {"last": 0, "max": 0, "mean": 0, "min": 0},
            "top_dead_files": [],
            "heatmap": {"summary": {"risk_band": "red"}},
            "version": "0.1.0",
        }
        r = TraceRollupResponse.model_validate(data)
        assert r.drift_trend == {"last": 0, "max": 0, "mean": 0, "min": 0}
        assert r.turns_processed == 2
        assert r.model_drift_pct == 1
        assert r.risk_band_trail == ["green", "red"]

    def test_response_extra_fields_preserved(self):
        """BaseResponse uses extra='allow', so forward-compat fields survive."""
        payload = _rag_response_fixture() | {"future_field": {"x": 1}}
        r = TraceRagResponse.model_validate(payload)
        assert getattr(r, "future_field") == {"x": 1}

    def test_real_prod_rag_response_parses(self):
        """Regression: real production gateway returns primary_metric='groundedness_v2'.

        The pod (see ``latence-trace/runpod/handler.py``) echoes
        ``"groundedness_v2"`` whenever ``scores.groundedness_v2`` is
        populated, not the requested input mode. The SDK must accept
        this without raising -- we use ``str`` on the response field
        rather than the input-side ``Literal``.
        """
        data = {
            "success": True,
            "score": 0.9235278423332299,
            "primary_metric": "groundedness_v2",   # <- NOT in the request-side Literal
            "band": "red",
            "nli_aggregate": 0.9986921121308114,
            "context_coverage_ratio": 1,
            "context_usage_ratio": 1,
            "context_unused_ratio": 0,
            "context_uncertain_ratio": 0,
            "support_units_usage": {"used": 1, "unused": 0, "uncertain": 0},
            "support_units": [
                {
                    "coverage_score": 0.9863664507865906,
                    "support_id": "raw-0",
                    "unused_confidence": 0.25,
                    "usage_confidence": 0.9986921121308114,
                    "usage_state": "used",
                    "used": True,
                }
            ],
            "latency_ms": 59.0165127068758,
            "scoring_mode": "rag",
            "version": "0.1.0",
            "file_attribution": {
                "coverage_threshold": 0.5,
                "dead_weight_files": [],
                "dead_weight_ratio": 0,
                "per_file": [{"path": "raw-0", "owner_share": 1.0}],
            },
            "heatmap": {
                "summary": {"headline_score": 0.9235, "risk_band": "red"},
                "tokens": [{"index": 0, "token": "Paris", "score": 0.98}],
            },
        }
        r = TraceRagResponse.model_validate(data)
        assert r.primary_metric == "groundedness_v2"
        assert r.band == "red"
        assert r.context_coverage_ratio == 1
        # Extra fields on nested models round-trip via extra='allow'.
        assert r.support_units is not None and len(r.support_units) == 1
        assert getattr(r.support_units[0], "support_id") == "raw-0"
        assert getattr(r.support_units[0], "used") is True
        assert r.file_attribution is not None
        assert getattr(r.file_attribution, "dead_weight_ratio") == 0
        assert r.heatmap is not None
        assert getattr(r.heatmap, "summary")["headline_score"] == 0.9235


# ===========================================================================
# Request-body shape (sync)
# ===========================================================================


class TestRagRequestBody:
    def test_rag_hits_correct_url(self):
        fake = _FakeSyncClient(data=_rag_response_fixture())
        Trace(fake).rag(response_text="answer", raw_context="context")

        assert len(fake.calls) == 1
        method, path, _ = fake.calls[0]
        assert method == "POST"
        assert path == "/api/v1/trace/rag"

    def test_rag_never_sends_scoring_mode(self):
        fake = _FakeSyncClient(data=_rag_response_fixture())
        Trace(fake).rag(response_text="answer", raw_context="context")

        body = fake.calls[0][2]
        assert body is not None
        assert "scoring_mode" not in body
        assert "action" not in body

    def test_rag_drops_none_keys(self):
        fake = _FakeSyncClient(data=_rag_response_fixture())
        Trace(fake).rag(response_text="answer", raw_context="context")

        body = fake.calls[0][2]
        assert body is not None
        assert body["response_text"] == "answer"
        assert body["raw_context"] == "context"
        # Unset optional kwargs must NOT be present.
        for forbidden in (
            "query_text",
            "chunk_ids",
            "support_units",
            "evidence_limit",
            "coverage_threshold",
            "primary_metric",
            "segmentation_mode",
            "attribution_mode",
            "heatmap_format",
            "verification_samples",
            "content_type",
            "risk_band_stratum",
            "model",
            "session_id",
            "verbose",
            "include_triangular_diagnostics",
            "request_id",
            "raw_context_chunk_tokens",
            "response_chunk_tokens",
            "query_prompt_name",
            "document_prompt_name",
            "debug_dense_matrices",
        ):
            assert forbidden not in body, f"{forbidden} should not be in the body"

    def test_rag_all_advanced_kwargs_pass_through(self):
        """Every gateway schema field is wired end-to-end.

        Regression for the gap the 360 bench surfaced: the SDK was
        silently dropping ``raw_context_chunk_tokens`` (+ 4 sibling
        knobs) because they weren't in the overload signatures, so
        callers couldn't exercise the full ``/rag`` contract.
        """
        fake = _FakeSyncClient(data=_rag_response_fixture())
        Trace(fake).rag(
            response_text="answer",
            raw_context="context",
            raw_context_chunk_tokens=48,
            response_chunk_tokens=128,
            query_prompt_name="bge-query",
            document_prompt_name="bge-doc",
            debug_dense_matrices=True,
            segmentation_mode="sentence_packed",
        )
        body = fake.calls[0][2]
        assert body is not None
        assert body["raw_context_chunk_tokens"] == 48
        assert body["response_chunk_tokens"] == 128
        assert body["query_prompt_name"] == "bge-query"
        assert body["document_prompt_name"] == "bge-doc"
        assert body["debug_dense_matrices"] is True
        assert body["segmentation_mode"] == "sentence_packed"

    def test_rag_support_units_serialized(self):
        fake = _FakeSyncClient(data=_rag_response_fixture())
        units = [
            SupportUnitInput(text="first", source_id="a"),
            {"text": "second", "source_id": "b", "speaker": "alice"},
        ]
        Trace(fake).rag(
            response_text="answer",
            support_units=units,
            query_text="what?",
            evidence_limit=4,
            heatmap_format="html",
            primary_metric="triangular",
        )

        body = fake.calls[0][2]
        assert body is not None
        assert body["support_units"] == [
            {"text": "first", "source_id": "a"},
            {"text": "second", "source_id": "b", "speaker": "alice"},
        ]
        assert body["query_text"] == "what?"
        assert body["evidence_limit"] == 4
        assert body["heatmap_format"] == "html"
        assert body["primary_metric"] == "triangular"

    def test_rag_chunk_ids_pass_through(self):
        fake = _FakeSyncClient(data=_rag_response_fixture())
        Trace(fake).rag(response_text="answer", chunk_ids=["c-1", "c-2"])

        body = fake.calls[0][2]
        assert body is not None
        assert body["chunk_ids"] == ["c-1", "c-2"]

    def test_rag_return_job_sets_async_flag(self):
        fake = _FakeSyncClient(
            data={
                "job_id": "job_abc",
                "poll_url": "/api/v1/jobs/job_abc",
                "status": "QUEUED",
            }
        )
        result = Trace(fake).rag(
            response_text="answer",
            raw_context="context",
            return_job=True,
        )

        body = fake.calls[0][2]
        assert body is not None
        assert body.get("async") is True
        assert result.job_id == "job_abc"


class TestCodeRequestBody:
    def test_code_hits_correct_url(self):
        fake = _FakeSyncClient(data=_code_response_fixture())
        Trace(fake).code(
            response_text="def f(): pass",
            raw_context="# utils.py",
            response_language_hint="python",
        )
        assert fake.calls[0][1] == "/api/v1/trace/code"

    def test_code_never_sends_scoring_mode(self):
        fake = _FakeSyncClient(data=_code_response_fixture())
        Trace(fake).code(
            response_text="def f(): pass",
            raw_context="# utils.py",
            response_language_hint="python",
        )
        body = fake.calls[0][2]
        assert body is not None
        assert "scoring_mode" not in body

    def test_code_passes_language_hint_and_ownership(self):
        fake = _FakeSyncClient(data=_code_response_fixture())
        Trace(fake).code(
            response_text="def f(): pass",
            raw_context="# utils.py",
            response_language_hint="py",
            emit_chunk_ownership=True,
        )
        body = fake.calls[0][2]
        assert body is not None
        assert body["response_language_hint"] == "py"
        assert body["emit_chunk_ownership"] is True


class TestCodeSessionRoundTrip:
    def test_session_state_dict_passed_verbatim(self):
        """A raw dict session_state must hit the wire unchanged."""
        fake = _FakeSyncClient(data=_code_response_fixture())
        prior = {"turn_count": 1, "ema_groundedness": 0.62, "history": [{"turn": 0}]}
        Trace(fake).code(
            response_text="def g(): pass",
            raw_context="# utils.py",
            session_state=prior,
        )
        body = fake.calls[0][2]
        assert body is not None
        assert body["session_state"] == prior

    def test_session_state_round_trip_from_prior_response(self):
        """Turn N's ``next_session_state`` must be usable as turn N+1's input."""
        fake = _FakeSyncClient(data=_code_response_fixture())
        trace = Trace(fake)

        turn1 = trace.code(
            response_text="def f(): pass",
            raw_context="# utils.py",
            response_language_hint="python",
        )
        assert isinstance(turn1, TraceCodeResponse)
        assert turn1.next_session_state is not None

        trace.code(
            response_text="def g(): pass",
            raw_context="# utils.py",
            response_language_hint="python",
            session_state=turn1.next_session_state,
        )

        body = fake.calls[-1][2]
        assert body is not None
        assert "session_state" in body
        # The dict round-trips exactly (keys/values from the fixture).
        assert body["session_state"]["turn_count"] == 1
        assert body["session_state"]["ema_groundedness"] == 0.62
        assert body["session_state"]["history"][0]["turn"] == 0

    def test_session_state_bare_pydantic_round_trip(self):
        """A bare ``SessionState`` with extras must serialize to the same dict."""
        fake = _FakeSyncClient(data=_code_response_fixture())
        state = SessionState.model_validate(
            {"turn_count": 3, "ema_groundedness": 0.81, "drift": 0.02}
        )
        Trace(fake).code(
            response_text="def h(): pass",
            raw_context="# utils.py",
            session_state=state,
        )
        body = fake.calls[0][2]
        assert body is not None
        assert body["session_state"] == {
            "turn_count": 3,
            "ema_groundedness": 0.81,
            "drift": 0.02,
        }


class TestRollupRequestBody:
    def test_rollup_hits_correct_url(self):
        fake = _FakeSyncClient(data=_rollup_response_fixture())
        Trace(fake).rollup(turns=[{"scores": {"groundedness_v2": 0.8}}])
        assert fake.calls[0][1] == "/api/v1/trace/rollup"

    def test_rollup_accepts_response_objects(self):
        """Passing in live response objects must serialize to dicts."""
        fake = _FakeSyncClient(data=_rollup_response_fixture())
        turn1 = TraceCodeResponse.model_validate(_code_response_fixture())
        turn2 = TraceRagResponse.model_validate(_rag_response_fixture())

        Trace(fake).rollup(turns=[turn1, turn2], session_id="sess_abc")

        body = fake.calls[0][2]
        assert body is not None
        assert isinstance(body["turns"], list)
        assert len(body["turns"]) == 2
        assert all(isinstance(t, dict) for t in body["turns"])
        # Code-lane-specific fields propagate.
        assert body["turns"][0].get("code_lane") is not None
        assert body["session_id"] == "sess_abc"

    def test_rollup_default_heatmap_not_sent(self):
        """Server injects the default ``heatmap_format='data'``; client stays silent."""
        fake = _FakeSyncClient(data=_rollup_response_fixture())
        Trace(fake).rollup(turns=[{"scores": {}}])
        body = fake.calls[0][2]
        assert body is not None
        assert "heatmap_format" not in body


# ===========================================================================
# Client-side validation
# ===========================================================================


class TestValidation:
    def test_rag_empty_response_text_raises(self):
        with pytest.raises(ValueError, match="response_text"):
            Trace(_FakeSyncClient()).rag(response_text="", raw_context="ctx")

    def test_rag_whitespace_response_text_raises(self):
        with pytest.raises(ValueError, match="response_text"):
            Trace(_FakeSyncClient()).rag(response_text="   ", raw_context="ctx")

    def test_rag_missing_premise_lane_raises(self):
        with pytest.raises(ValueError, match="raw_context"):
            Trace(_FakeSyncClient()).rag(response_text="answer")

    def test_code_missing_premise_lane_raises(self):
        with pytest.raises(ValueError, match="raw_context"):
            Trace(_FakeSyncClient()).code(response_text="def f(): pass")

    def test_rollup_empty_turns_raises(self):
        with pytest.raises(ValueError, match="turns"):
            Trace(_FakeSyncClient()).rollup(turns=[])

    def test_rag_support_units_counts_as_premise(self):
        """``support_units`` alone satisfies ``required_fields_any``."""
        fake = _FakeSyncClient(data=_rag_response_fixture())
        Trace(fake).rag(
            response_text="answer",
            support_units=[{"text": "premise-1"}],
        )
        assert fake.calls[0][1] == "/api/v1/trace/rag"

    def test_rag_chunk_ids_counts_as_premise(self):
        fake = _FakeSyncClient(data=_rag_response_fixture())
        Trace(fake).rag(response_text="answer", chunk_ids=["c-1"])
        assert fake.calls[0][1] == "/api/v1/trace/rag"


# ===========================================================================
# Metadata injection
# ===========================================================================


class TestMetadataInjection:
    def test_request_id_injected_from_response_metadata(self):
        fake = _FakeSyncClient(data=_rag_response_fixture())
        r = Trace(fake).rag(response_text="answer", raw_context="context")
        assert r.request_id == "req_test"


# ===========================================================================
# Async parity
# ===========================================================================


class TestAsyncParity:
    @pytest.mark.asyncio
    async def test_async_rag_shape(self):
        fake = _FakeAsyncClient(data=_rag_response_fixture())
        r = await AsyncTrace(fake).rag(
            response_text="answer",
            raw_context="context",
        )
        assert fake.calls[0][1] == "/api/v1/trace/rag"
        body = fake.calls[0][2]
        assert body is not None
        assert "scoring_mode" not in body
        assert r.scoring_mode == "rag"

    @pytest.mark.asyncio
    async def test_async_code_session_round_trip(self):
        fake = _FakeAsyncClient(data=_code_response_fixture())
        trace = AsyncTrace(fake)

        turn1 = await trace.code(
            response_text="def f(): pass",
            raw_context="# utils.py",
            response_language_hint="python",
        )
        await trace.code(
            response_text="def g(): pass",
            raw_context="# utils.py",
            response_language_hint="python",
            session_state=turn1.next_session_state,
        )
        body = fake.calls[-1][2]
        assert body is not None
        assert body["session_state"]["turn_count"] == 1

    @pytest.mark.asyncio
    async def test_async_rollup(self):
        fake = _FakeAsyncClient(data=_rollup_response_fixture())
        r = await AsyncTrace(fake).rollup(
            turns=[{"scores": {"groundedness_v2": 0.8}}],
        )
        assert fake.calls[0][1] == "/api/v1/trace/rollup"
        assert r.turns_processed == 4


# ===========================================================================
# Experimental namespace
# ===========================================================================


class TestExperimentalNamespaceTrace:
    def test_sync_namespace_exposes_trace(self):
        ns = ExperimentalNamespace(_FakeSyncClient())
        assert isinstance(ns.trace, Trace)
        # Property returns the cached instance (same object).
        assert ns.trace is ns.trace

    def test_async_namespace_exposes_trace(self):
        ns = AsyncExperimentalNamespace(_FakeAsyncClient())
        assert isinstance(ns.trace, AsyncTrace)
        assert ns.trace is ns.trace
