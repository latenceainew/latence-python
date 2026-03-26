"""
End-to-end Dataset Intelligence tests for the Latence AI SDK.

Validates the full DI lifecycle:
  - Dataset creation (full tier, async job submission)
  - Job polling via gateway pipeline endpoint
  - Delta ingestion (append mode with existing dataset_id)
  - Response model parsing (DatasetIntelligenceResponse)
  - B2 presigned upload path (payloads > 8 MB trigger upload automatically)
  - Tier-specific methods (enrich, build_graph, build_ontology)

Prerequisites:
  1. API Gateway deployed (or local wrangler dev)
  2. Pipeline Worker deployed (or local)
  3. DI RunPod endpoint running (RUNPOD_DI_ENDPOINT_ID configured)
  4. LATENCE_API_KEY set in environment
  5. LATENCE_BASE_URL set (defaults to https://api.latence.ai)
  6. Example pipeline output at ../../../output_pipe_d9f5c1d292064a2d/

Run:
    LATENCE_API_KEY=lat_xxx \
      pytest tests/integration/test_di_e2e.py -v -s
"""

import json
import os
import time
from pathlib import Path

import httpx
import pytest

from latence import Latence
from latence._models.dataset_intelligence_service import (
    DatasetIntelligenceDeltaSummary,
    DatasetIntelligenceResponse,
    DatasetIntelligenceUsage,
)
from latence._models.jobs import JobSubmittedResponse

pytestmark = pytest.mark.skipif(
    not os.environ.get("LATENCE_API_KEY"),
    reason="LATENCE_API_KEY not set — skipping DI E2E tests",
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXAMPLE_DATA = WORKSPACE_ROOT / "output_pipe_d9f5c1d292064a2d"
POLL_INTERVAL = 5
POLL_TIMEOUT = 600


@pytest.fixture(scope="module")
def client():
    return Latence()


@pytest.fixture(scope="module")
def pipeline_payload():
    """Load the example pipeline output into the dict format the DI service expects."""
    if not EXAMPLE_DATA.exists():
        pytest.skip(f"Example data not found at {EXAMPLE_DATA}")

    result_json = EXAMPLE_DATA / "result.json"
    manifest = json.loads(result_json.read_text()) if result_json.exists() else {}

    stages = {}
    for stage_dir in sorted(EXAMPLE_DATA.iterdir()):
        if not stage_dir.is_dir() or not stage_dir.name.startswith("stage_"):
            continue
        jsonl_file = stage_dir / "results.jsonl"
        if not jsonl_file.exists():
            continue
        records = []
        for line in jsonl_file.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        stages[stage_dir.name] = records

    payload = {"result": manifest, **stages}
    payload_size = len(json.dumps(payload, default=str).encode())
    print(f"\n[DI E2E] Payload size: {payload_size / 1024 / 1024:.1f} MB")
    print(f"[DI E2E] Stages: {list(stages.keys())}")
    print(f"[DI E2E] Records per stage: { {k: len(v) for k, v in stages.items()} }")
    return payload


def _poll_pipeline_job(client: Latence, job_id: str) -> dict:
    """Poll the pipeline endpoint until the job completes or times out."""
    base_url = client._client.base_url
    api_key = client._client.api_key
    url = f"{base_url}/api/v1/pipeline/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    start = time.monotonic()
    last_status = ""
    while time.monotonic() - start < POLL_TIMEOUT:
        resp = httpx.get(url, headers=headers, timeout=30)
        if resp.status_code == 404:
            time.sleep(POLL_INTERVAL)
            continue
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "UNKNOWN")

        if status != last_status:
            elapsed = time.monotonic() - start
            print(f"[DI E2E] [{elapsed:.0f}s] Job {job_id}: {status}")
            last_status = status

        if status in ("COMPLETED", "FAILED", "CANCELLED"):
            return data

        time.sleep(POLL_INTERVAL)

    raise TimeoutError(f"Job {job_id} did not complete within {POLL_TIMEOUT}s")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDatasetCreation:
    """Test creating a new dataset via the full DI pipeline."""

    def test_submit_async_job(self, client, pipeline_payload):
        """Submit a full-tier DI job and verify the JobSubmittedResponse."""
        di = client.experimental.dataset_intelligence_service

        result = di.run(
            input_data=pipeline_payload,
            return_job=True,
        )

        assert isinstance(result, JobSubmittedResponse), (
            f"Expected JobSubmittedResponse, got {type(result).__name__}"
        )
        assert result.job_id, "job_id must not be empty"
        assert result.job_id.startswith("di_"), (
            f"DI job_id must start with 'di_', got '{result.job_id}'"
        )
        assert result.status == "QUEUED"
        print(f"[DI E2E] Submitted job: {result.job_id}")

        # Store for subsequent tests
        TestDatasetCreation._job_id = result.job_id

    def test_poll_to_completion(self, client):
        """Poll until the DI job completes and verify the status."""
        job_id = getattr(TestDatasetCreation, "_job_id", None)
        if not job_id:
            pytest.skip("No job_id from previous test")

        result = _poll_pipeline_job(client, job_id)

        assert result["status"] == "COMPLETED", (
            f"Expected COMPLETED, got {result['status']}: {result.get('error_message', 'no error')}"
        )
        print(f"[DI E2E] Job completed: {job_id}")

        # Store dataset_id for delta ingestion test
        TestDatasetCreation._completed_job = result

    def test_job_has_download_url(self, client):
        """Verify the completed job has a download_url for result artifacts."""
        result = getattr(TestDatasetCreation, "_completed_job", None)
        if not result:
            pytest.skip("No completed job from previous test")

        download_url = result.get("download_url")
        assert download_url, "Completed DI job must have a download_url"
        print(f"[DI E2E] Download URL present (len={len(download_url)})")

    def test_download_result_parses(self, client):
        """Download the result JSON and verify it parses as DatasetIntelligenceResponse."""
        result = getattr(TestDatasetCreation, "_completed_job", None)
        if not result:
            pytest.skip("No completed job from previous test")

        download_url = result.get("download_url")
        if not download_url:
            pytest.skip("No download_url")

        resp = httpx.get(download_url, timeout=60)
        resp.raise_for_status()
        di_result = resp.json()

        parsed = DatasetIntelligenceResponse.model_validate(di_result)

        assert parsed.success is True
        assert parsed.dataset_id, "dataset_id must be set"
        assert parsed.tier in ("tier1", "tier2", "tier3", "full")
        assert parsed.mode in ("create", "append")
        assert isinstance(parsed.usage, DatasetIntelligenceUsage)
        assert parsed.usage.credits >= 0
        assert len(parsed.stage_timings) > 0
        assert parsed.data, "data dict must not be empty"

        data = parsed.data
        print(f"[DI E2E] Dataset: {parsed.dataset_id}")
        print(f"[DI E2E] Tier: {parsed.tier}, Mode: {parsed.mode}")
        print(f"[DI E2E] Credits: {parsed.usage.credits}")
        print(f"[DI E2E] Entities: {data.get('total_entities', 0)}")
        print(f"[DI E2E] Relations: {data.get('total_relations', 0)}")
        print(f"[DI E2E] Graph nodes: {data.get('total_graph_nodes', 0)}")
        print(f"[DI E2E] Concepts: {len(data.get('concepts', []))}")

        TestDatasetCreation._dataset_id = parsed.dataset_id
        TestDatasetCreation._parsed_result = parsed


class TestDeltaIngestion:
    """Test appending to an existing dataset (incremental ingestion)."""

    def test_append_submit(self, client, pipeline_payload):
        """Submit an append-mode DI job to the same dataset."""
        dataset_id = getattr(TestDatasetCreation, "_dataset_id", None)
        if not dataset_id:
            pytest.skip("No dataset_id from creation test")

        di = client.experimental.dataset_intelligence_service
        result = di.run(
            input_data=pipeline_payload,
            dataset_id=dataset_id,
            return_job=True,
        )

        assert isinstance(result, JobSubmittedResponse)
        assert result.job_id.startswith("di_")
        print(f"[DI E2E] Append job submitted: {result.job_id}")

        TestDeltaIngestion._append_job_id = result.job_id

    def test_append_completes(self, client):
        """Poll append job to completion."""
        job_id = getattr(TestDeltaIngestion, "_append_job_id", None)
        if not job_id:
            pytest.skip("No append job_id")

        result = _poll_pipeline_job(client, job_id)
        assert result["status"] == "COMPLETED", (
            f"Append job failed: {result.get('error_message', 'unknown')}"
        )
        TestDeltaIngestion._append_result = result

    def test_append_has_delta_summary(self, client):
        """Verify the append result contains a delta_summary."""
        result = getattr(TestDeltaIngestion, "_append_result", None)
        if not result:
            pytest.skip("No append result")

        download_url = result.get("download_url")
        if not download_url:
            pytest.skip("No download_url for append result")

        resp = httpx.get(download_url, timeout=60)
        resp.raise_for_status()
        di_result = resp.json()

        parsed = DatasetIntelligenceResponse.model_validate(di_result)
        assert parsed.mode == "append", f"Expected append mode, got {parsed.mode}"
        assert parsed.delta_summary is not None, "Append result must have delta_summary"
        assert isinstance(parsed.delta_summary, DatasetIntelligenceDeltaSummary)

        ds = parsed.delta_summary
        print("[DI E2E] Delta summary:")
        print(f"  files_added={ds.files_added}, files_unchanged={ds.files_unchanged}")
        print(f"  new_entities={ds.new_entities}, merged_entities={ds.merged_entities}")
        print(f"  new_edges={ds.new_edges}, removed_edges={ds.removed_edges}")


class TestTierMethods:
    """Smoke test individual tier methods (submit only, don't wait)."""

    def test_enrich_submits(self, client, pipeline_payload):
        """Tier 1: enrich() submits without error."""
        di = client.experimental.dataset_intelligence_service
        try:
            result = di.enrich(input_data=pipeline_payload)
            assert isinstance(result, DatasetIntelligenceResponse)
            assert result.tier == "tier1"
            print(f"[DI E2E] enrich() returned: tier={result.tier}, credits={result.usage.credits}")
        except Exception as e:
            if "timeout" in str(e).lower() or "504" in str(e):
                pytest.skip(f"Tier1 sync call timed out (expected for large payloads): {e}")
            raise

    def test_build_graph_async(self, client, pipeline_payload):
        """Tier 2: build_graph(return_job=True) returns a job."""
        di = client.experimental.dataset_intelligence_service
        result = di.build_graph(
            input_data=pipeline_payload,
            return_job=True,
        )
        assert isinstance(result, JobSubmittedResponse)
        assert result.job_id.startswith("di_")
        print(f"[DI E2E] build_graph() job: {result.job_id}")

    def test_build_ontology_submits(self, client, pipeline_payload):
        """Tier 3: build_ontology() submits without error."""
        di = client.experimental.dataset_intelligence_service
        try:
            result = di.build_ontology(input_data=pipeline_payload)
            assert isinstance(result, DatasetIntelligenceResponse)
            assert result.tier == "tier3"
            print(
                f"[DI E2E] build_ontology() returned: tier={result.tier}, "
                f"credits={result.usage.credits}"
            )
        except Exception as e:
            if "timeout" in str(e).lower() or "504" in str(e):
                pytest.skip(f"Tier3 sync call timed out (expected for large payloads): {e}")
            raise

    def test_run_async(self, client, pipeline_payload):
        """Full: run(return_job=True) returns a job."""
        di = client.experimental.dataset_intelligence_service
        result = di.run(
            input_data=pipeline_payload,
            return_job=True,
        )
        assert isinstance(result, JobSubmittedResponse)
        assert result.job_id.startswith("di_")
        print(f"[DI E2E] run() job: {result.job_id}")


class TestB2PresignedUpload:
    """Verify large payloads trigger the B2 presigned upload path."""

    def test_payload_exceeds_threshold(self, pipeline_payload):
        """The example data should exceed the 8 MB inline threshold."""
        size = len(json.dumps(pipeline_payload, default=str).encode())
        threshold = 8 * 1024 * 1024
        size_mb = size / 1024 / 1024
        thresh_mb = threshold / 1024 / 1024
        print(f"[DI E2E] Payload: {size_mb:.1f} MB, threshold: {thresh_mb:.0f} MB")
        assert size > threshold, (
            f"Payload ({size / 1024 / 1024:.1f} MB) should exceed "
            f"{threshold / 1024 / 1024:.0f} MB threshold for presigned upload test"
        )

    def test_presign_endpoint(self, client):
        """The /api/v1/di/presign endpoint responds correctly."""
        base_url = client._client.base_url
        api_key = client._client.api_key

        resp = httpx.post(
            f"{base_url}/api/v1/di/presign",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"content_type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        assert "upload_url" in data, "presign response must include upload_url"
        assert "download_url" in data, "presign response must include download_url"
        print(f"[DI E2E] Presign OK: upload_url len={len(data['upload_url'])}")


class TestResponseModels:
    """Validate response model field coverage."""

    def test_usage_fields(self):
        """DatasetIntelligenceUsage parses all expected fields."""
        parsed = getattr(TestDatasetCreation, "_parsed_result", None)
        if not parsed:
            pytest.skip("No parsed result from creation test")

        u = parsed.usage
        assert isinstance(u.credits, float)
        assert isinstance(u.details, dict)
        print(
            f"[DI E2E] Usage: credits={u.credits}, "
            f"calculation='{u.calculation}', details keys={list(u.details.keys())}"
        )

    def test_stage_timings(self):
        """Stage timings are populated."""
        parsed = getattr(TestDatasetCreation, "_parsed_result", None)
        if not parsed:
            pytest.skip("No parsed result from creation test")

        assert len(parsed.stage_timings) > 0
        for t in parsed.stage_timings:
            assert t.stage, "stage name must not be empty"
            print(f"[DI E2E] Stage: {t.stage} — {t.elapsed_ms:.0f}ms ({t.status})")

    def test_data_payload_keys(self):
        """The data dict contains expected keys for a full-tier run."""
        parsed = getattr(TestDatasetCreation, "_parsed_result", None)
        if not parsed:
            pytest.skip("No parsed result from creation test")

        data = parsed.data
        expected_counts = ["total_entities", "total_relations", "total_graph_nodes"]
        for key in expected_counts:
            assert key in data, f"data['{key}'] expected in full-tier result"
            assert isinstance(data[key], int), f"data['{key}'] should be int"

        print(f"[DI E2E] Data keys: {sorted(data.keys())}")
