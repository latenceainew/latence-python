"""Offline contract tests for the Dataset Intelligence SDK resource.

Validates request construction, response parsing, the presigned upload
path for large payloads, and sync/async parity — all without network
access.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from latence._base import APIResponse, ResponseMetadata
from latence._models.dataset_intelligence_service import (
    DatasetIntelligenceDeltaSummary,
    DatasetIntelligenceResponse,
    DatasetIntelligenceStageTiming,
    DatasetIntelligenceUsage,
)
from latence._models.jobs import JobSubmittedResponse
from latence.resources.dataset_intelligence_service import (
    _DI_PAYLOAD_THRESHOLD,
    _DI_PRESIGN_ENDPOINT,
    _ENDPOINT,
    AsyncDatasetIntelligenceService,
    DatasetIntelligenceService,
)

_METADATA = ResponseMetadata(request_id="req_test_di")

_DI_RESPONSE_DATA: dict[str, Any] = {
    "success": True,
    "endpoint_id": "dataset_intelligence",
    "tier": "full",
    "dataset_id": "ds_unit_test",
    "mode": "create",
    "data": {
        "total_entities": 10,
        "total_relations": 5,
        "total_graph_nodes": 8,
    },
    "usage": {"credits": 1.5, "calculation": "1k pages", "details": {"num_pages": 3}},
    "stage_timings": [
        {"stage": "enrichment", "elapsed_ms": 100.0, "status": "completed"},
        {"stage": "graph", "elapsed_ms": 200.0, "status": "completed"},
    ],
    "processing_time_ms": 300.0,
    "version": "1.0.0",
}

_JOB_RESPONSE_DATA: dict[str, Any] = {
    "success": True,
    "job_id": "di_abc123",
    "status": "QUEUED",
}

_PRESIGN_RESPONSE_DATA: dict[str, Any] = {
    "upload_url": "https://b2.example.com/upload?token=xyz",
    "download_url": "https://b2.example.com/download/payload.json",
}


class _FakeSyncClient:
    """Minimal stand-in for BaseSyncClient used in unit tests."""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self._responses = list(responses or [_DI_RESPONSE_DATA])
        self._call_idx = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, path: str, json: dict[str, Any]) -> APIResponse:
        self.calls.append((path, json))
        data = self._responses[min(self._call_idx, len(self._responses) - 1)]
        self._call_idx += 1
        return APIResponse(data=data, metadata=_METADATA, status_code=200)


class _FakeAsyncClient:
    """Minimal stand-in for BaseAsyncClient used in async unit tests."""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self._responses = list(responses or [_DI_RESPONSE_DATA])
        self._call_idx = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def post(self, path: str, json: dict[str, Any]) -> APIResponse:
        self.calls.append((path, json))
        data = self._responses[min(self._call_idx, len(self._responses) - 1)]
        self._call_idx += 1
        return APIResponse(data=data, metadata=_METADATA, status_code=200)


# ---------------------------------------------------------------------------
# Request body construction
# ---------------------------------------------------------------------------


class TestRequestBody:
    """Verify the JSON body sent to the gateway for each method."""

    def test_run_full_tier(self) -> None:
        client = _FakeSyncClient()
        svc = DatasetIntelligenceService(client)  # type: ignore[arg-type]

        svc.run(input_data={"stage_01": [{"file_id": "f1"}]})

        path, body = client.calls[0]
        assert path == _ENDPOINT
        assert body["endpoint_id"] == "dataset_intelligence"
        assert body["tier"] == "full"
        assert body["input_data"] == {"stage_01": [{"file_id": "f1"}]}
        assert "async" not in body

    def test_run_with_return_job(self) -> None:
        client = _FakeSyncClient([_JOB_RESPONSE_DATA])
        svc = DatasetIntelligenceService(client)  # type: ignore[arg-type]

        result = svc.run(
            input_data={"stage_01": []},
            return_job=True,
        )

        _, body = client.calls[0]
        assert body["async"] is True
        assert isinstance(result, JobSubmittedResponse)
        assert result.job_id == "di_abc123"

    def test_run_with_dataset_id(self) -> None:
        client = _FakeSyncClient()
        svc = DatasetIntelligenceService(client)  # type: ignore[arg-type]

        svc.run(input_data={"s": []}, dataset_id="ds_existing")

        _, body = client.calls[0]
        assert body["dataset_id"] == "ds_existing"

    def test_enrich_tier1(self) -> None:
        client = _FakeSyncClient()
        svc = DatasetIntelligenceService(client)  # type: ignore[arg-type]

        svc.enrich(input_data={"s": []})

        _, body = client.calls[0]
        assert body["tier"] == "tier1"
        assert "async" not in body

    def test_build_graph_tier2(self) -> None:
        client = _FakeSyncClient([_JOB_RESPONSE_DATA])
        svc = DatasetIntelligenceService(client)  # type: ignore[arg-type]

        result = svc.build_graph(input_data={"s": []}, return_job=True)

        _, body = client.calls[0]
        assert body["tier"] == "tier2"
        assert body["async"] is True
        assert isinstance(result, JobSubmittedResponse)

    def test_build_ontology_tier3(self) -> None:
        client = _FakeSyncClient()
        svc = DatasetIntelligenceService(client)  # type: ignore[arg-type]

        svc.build_ontology(input_data={"s": []})

        _, body = client.calls[0]
        assert body["tier"] == "tier3"

    def test_config_overrides_forwarded(self) -> None:
        client = _FakeSyncClient()
        svc = DatasetIntelligenceService(client)  # type: ignore[arg-type]

        svc.run(
            input_data={"s": []},
            config_overrides={"enrichment": {"chunk_size": 512}},
        )

        _, body = client.calls[0]
        assert body["config_overrides"] == {"enrichment": {"chunk_size": 512}}

    def test_request_id_forwarded(self) -> None:
        client = _FakeSyncClient()
        svc = DatasetIntelligenceService(client)  # type: ignore[arg-type]

        svc.run(input_data={"s": []}, request_id="rid_42")

        _, body = client.calls[0]
        assert body["request_id"] == "rid_42"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    """Verify that canned API responses parse into the correct models."""

    def test_di_response_fields(self) -> None:
        client = _FakeSyncClient()
        svc = DatasetIntelligenceService(client)  # type: ignore[arg-type]

        result = svc.run(input_data={"s": []})

        assert isinstance(result, DatasetIntelligenceResponse)
        assert result.dataset_id == "ds_unit_test"
        assert result.tier == "full"
        assert result.mode == "create"
        assert result.data["total_entities"] == 10
        assert isinstance(result.usage, DatasetIntelligenceUsage)
        assert result.usage.credits == 1.5
        assert len(result.stage_timings) == 2
        assert isinstance(result.stage_timings[0], DatasetIntelligenceStageTiming)

    def test_job_submitted_response(self) -> None:
        client = _FakeSyncClient([_JOB_RESPONSE_DATA])
        svc = DatasetIntelligenceService(client)  # type: ignore[arg-type]

        result = svc.run(input_data={"s": []}, return_job=True)

        assert isinstance(result, JobSubmittedResponse)
        assert result.job_id == "di_abc123"
        assert result.status == "QUEUED"

    def test_delta_summary_parsed(self) -> None:
        data = {
            **_DI_RESPONSE_DATA,
            "mode": "append",
            "delta_summary": {
                "files_added": 3,
                "new_entities": 12,
                "merged_entities": 2,
                "new_edges": 8,
                "removed_edges": 1,
                "delta_ratio": 0.35,
            },
        }
        client = _FakeSyncClient([data])
        svc = DatasetIntelligenceService(client)  # type: ignore[arg-type]

        result = svc.run(input_data={"s": []})

        assert isinstance(result, DatasetIntelligenceResponse)
        assert result.mode == "append"
        assert result.delta_summary is not None
        assert isinstance(result.delta_summary, DatasetIntelligenceDeltaSummary)
        assert result.delta_summary.files_added == 3
        assert result.delta_summary.new_entities == 12
        assert result.delta_summary.delta_ratio == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# Large-payload presigned upload path
# ---------------------------------------------------------------------------


class TestPresignedUpload:
    """Verify that payloads > 8 MB trigger the presign+upload flow."""

    def _make_large_payload(self) -> dict[str, Any]:
        """Return a dict whose JSON encoding exceeds _DI_PAYLOAD_THRESHOLD."""
        filler = "x" * (_DI_PAYLOAD_THRESHOLD + 1024)
        return {"big_field": filler}

    @patch("latence.resources.dataset_intelligence_service.httpx.put")
    def test_large_payload_triggers_presign(self, mock_put: MagicMock) -> None:
        mock_put.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())

        client = _FakeSyncClient([_PRESIGN_RESPONSE_DATA, _DI_RESPONSE_DATA])
        svc = DatasetIntelligenceService(client)  # type: ignore[arg-type]

        payload = self._make_large_payload()
        svc.run(input_data=payload)

        assert len(client.calls) == 2
        presign_path, presign_body = client.calls[0]
        assert presign_path == _DI_PRESIGN_ENDPOINT
        assert presign_body["content_type"] == "application/json"

        process_path, process_body = client.calls[1]
        assert process_path == _ENDPOINT
        assert process_body["input_url"] == _PRESIGN_RESPONSE_DATA["download_url"]
        assert "input_data" not in process_body

        mock_put.assert_called_once()
        call_kwargs = mock_put.call_args
        assert call_kwargs[0][0] == _PRESIGN_RESPONSE_DATA["upload_url"]

    @patch("latence.resources.dataset_intelligence_service.httpx.put")
    def test_large_payload_with_dataset_id(self, mock_put: MagicMock) -> None:
        mock_put.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())

        client = _FakeSyncClient([_PRESIGN_RESPONSE_DATA, _DI_RESPONSE_DATA])
        svc = DatasetIntelligenceService(client)  # type: ignore[arg-type]

        payload = self._make_large_payload()
        svc.run(input_data=payload, dataset_id="ds_append")

        _, presign_body = client.calls[0]
        assert presign_body["dataset_id"] == "ds_append"

        _, process_body = client.calls[1]
        assert process_body["dataset_id"] == "ds_append"

    def test_small_payload_stays_inline(self) -> None:
        client = _FakeSyncClient()
        svc = DatasetIntelligenceService(client)  # type: ignore[arg-type]

        svc.run(input_data={"small": True})

        assert len(client.calls) == 1
        _, body = client.calls[0]
        assert body["input_data"] == {"small": True}
        assert "input_url" not in body


# ---------------------------------------------------------------------------
# Async parity
# ---------------------------------------------------------------------------


class TestAsyncParity:
    """Verify the async resource produces identical request bodies."""

    @pytest.mark.asyncio
    async def test_async_run_body(self) -> None:
        client = _FakeAsyncClient()
        svc = AsyncDatasetIntelligenceService(client)  # type: ignore[arg-type]

        result = await svc.run(input_data={"stage_01": [{"file_id": "f1"}]})

        path, body = client.calls[0]
        assert path == _ENDPOINT
        assert body["endpoint_id"] == "dataset_intelligence"
        assert body["tier"] == "full"
        assert isinstance(result, DatasetIntelligenceResponse)

    @pytest.mark.asyncio
    async def test_async_run_return_job(self) -> None:
        client = _FakeAsyncClient([_JOB_RESPONSE_DATA])
        svc = AsyncDatasetIntelligenceService(client)  # type: ignore[arg-type]

        result = await svc.run(input_data={"s": []}, return_job=True)

        _, body = client.calls[0]
        assert body["async"] is True
        assert isinstance(result, JobSubmittedResponse)

    @pytest.mark.asyncio
    async def test_async_enrich_tier(self) -> None:
        client = _FakeAsyncClient()
        svc = AsyncDatasetIntelligenceService(client)  # type: ignore[arg-type]

        await svc.enrich(input_data={"s": []})

        _, body = client.calls[0]
        assert body["tier"] == "tier1"

    @pytest.mark.asyncio
    async def test_async_build_graph_tier(self) -> None:
        client = _FakeAsyncClient([_JOB_RESPONSE_DATA])
        svc = AsyncDatasetIntelligenceService(client)  # type: ignore[arg-type]

        await svc.build_graph(input_data={"s": []}, return_job=True)

        _, body = client.calls[0]
        assert body["tier"] == "tier2"

    @pytest.mark.asyncio
    async def test_async_build_ontology_tier(self) -> None:
        client = _FakeAsyncClient()
        svc = AsyncDatasetIntelligenceService(client)  # type: ignore[arg-type]

        await svc.build_ontology(input_data={"s": []})

        _, body = client.calls[0]
        assert body["tier"] == "tier3"

    @pytest.mark.asyncio
    async def test_async_large_payload_presign(self) -> None:
        filler = "x" * (_DI_PAYLOAD_THRESHOLD + 1024)
        payload = {"big_field": filler}

        client = _FakeAsyncClient([_PRESIGN_RESPONSE_DATA, _DI_RESPONSE_DATA])
        svc = AsyncDatasetIntelligenceService(client)  # type: ignore[arg-type]

        mock_response = MagicMock(status_code=200)
        mock_response.raise_for_status = MagicMock()

        mock_put = AsyncMock(return_value=mock_response)
        mock_async_client_instance = AsyncMock()
        mock_async_client_instance.put = mock_put
        mock_async_client_instance.__aenter__ = AsyncMock(return_value=mock_async_client_instance)
        mock_async_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "latence.resources.dataset_intelligence_service.httpx.AsyncClient",
            return_value=mock_async_client_instance,
        ):
            await svc.run(input_data=payload)

        assert len(client.calls) == 2
        presign_path, _ = client.calls[0]
        assert presign_path == _DI_PRESIGN_ENDPOINT

        process_path, process_body = client.calls[1]
        assert process_path == _ENDPOINT
        assert process_body["input_url"] == _PRESIGN_RESPONSE_DATA["download_url"]
        assert "input_data" not in process_body
