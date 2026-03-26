"""
SDK vs Portal Parity Check

Verifies that the SDK and Portal produce equivalent results when given
the same input and pipeline configuration. Both paths go through the same
API Gateway -> Worker pipeline, so results should match modulo timing/job_id.

Prerequisites:
  1. API Gateway running
  2. Pipeline Worker running (MOCK_RUNPOD=true OK for this test)
  3. LATENCE_API_KEY and LATENCE_BASE_URL set
  4. PORTAL_PIPELINE_URL set (e.g., http://localhost:3000/api/pipeline)

Run:
    LATENCE_API_KEY=lat_xxx \
    LATENCE_BASE_URL=https://your-gateway.workers.dev \
    PORTAL_PIPELINE_URL=http://localhost:3000/api/pipeline \
    SUPABASE_URL=https://xxx.supabase.co \
    SUPABASE_KEY=xxx \
        pytest tests/integration/test_parity_check.py -v -s

This test submits the same file via both SDK and Portal API routes,
waits for both to complete, then compares the Supabase job records
to verify structural parity.
"""

import base64
import json
import os
import time
from pathlib import Path

import httpx
import pytest

from latence import Latence


# ---------------------------------------------------------------------------
# Skip if required env vars are not set
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not all([
        os.environ.get("LATENCE_API_KEY"),
        os.environ.get("PORTAL_PIPELINE_URL"),
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_KEY"),
    ]),
    reason="Requires LATENCE_API_KEY, PORTAL_PIPELINE_URL, SUPABASE_URL, SUPABASE_KEY",
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"

PORTAL_PIPELINE_URL = os.environ.get("PORTAL_PIPELINE_URL", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
LATENCE_API_KEY = os.environ.get("LATENCE_API_KEY", "")


def _query_supabase_job(job_id: str) -> dict | None:
    """Query pipeline_jobs directly from Supabase."""
    resp = httpx.get(
        f"{SUPABASE_URL}/rest/v1/pipeline_jobs?job_id=eq.{job_id}",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        },
    )
    if resp.status_code == 200:
        rows = resp.json()
        return rows[0] if rows else None
    return None


def _submit_via_portal(pdf_path: Path) -> str:
    """
    Submit a pipeline via the Portal's API route (mimicking the Portal UI).

    Returns the job_id.
    """
    file_data = pdf_path.read_bytes()
    b64 = base64.b64encode(file_data).decode("ascii")

    payload = {
        "input": {
            "files": [
                {
                    "name": pdf_path.name,
                    "data": b64,
                    "type": "application/pdf",
                }
            ]
        },
        "services": [
            {"service": "document_intelligence", "config": {}},
            {"service": "extraction", "config": {}},
            {"service": "ontology", "config": {}},
        ],
        "store_intermediate": False,
        "name": "Parity Check - Portal",
        "submitted_via": "portal",
    }

    # Submit through the portal's API route (which proxies to the gateway)
    resp = httpx.post(
        PORTAL_PIPELINE_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {LATENCE_API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )

    data = resp.json()
    assert resp.status_code in (200, 201, 202), f"Portal submission failed: {data}"
    job_id = data.get("job_id")
    assert job_id, f"No job_id in portal response: {data}"
    return job_id


def _wait_for_job(job_id: str, timeout: int = 180) -> dict:
    """Poll Supabase until job reaches a terminal state."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        job = _query_supabase_job(job_id)
        if job and job["status"] in ("COMPLETED", "CACHED", "PULLED", "FAILED"):
            return job
        time.sleep(3)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


class TestSDKPortalParity:
    """Verify that SDK and Portal produce equivalent results."""

    def test_same_input_same_pipeline(self):
        """Submit same PDF + same services via SDK and Portal, compare results."""
        assert SAMPLE_PDF.exists(), f"Test fixture missing: {SAMPLE_PDF}"

        # 1. Submit via SDK
        client = Latence()
        sdk_job = client.pipeline.run(
            files=[str(SAMPLE_PDF)],
            steps={
                "ocr": {},
                "extraction": {},
                "knowledge_graph": {},
            },
            name="Parity Check - SDK",
        )
        print(f"\n  SDK job: {sdk_job.id}")

        # 2. Submit via Portal
        portal_job_id = _submit_via_portal(SAMPLE_PDF)
        print(f"  Portal job: {portal_job_id}")

        # 3. Wait for both to complete
        sdk_db = _wait_for_job(sdk_job.id)
        portal_db = _wait_for_job(portal_job_id)

        # 4. Compare structural properties
        print(f"\n  SDK status: {sdk_db['status']}, Portal status: {portal_db['status']}")

        # Both should succeed
        assert sdk_db["status"] in ("COMPLETED", "CACHED"), \
            f"SDK job failed: {sdk_db.get('error_message')}"
        assert portal_db["status"] in ("COMPLETED", "CACHED"), \
            f"Portal job failed: {portal_db.get('error_message')}"

        # Same number of stages
        assert sdk_db["total_stages"] == portal_db["total_stages"], \
            f"Stage count mismatch: SDK={sdk_db['total_stages']}, Portal={portal_db['total_stages']}"
        assert sdk_db["stages_completed"] == portal_db["stages_completed"]

        # Same services executed
        sdk_services = json.loads(sdk_db["services"]) if isinstance(sdk_db["services"], str) else sdk_db["services"]
        portal_services = json.loads(portal_db["services"]) if isinstance(portal_db["services"], str) else portal_db["services"]

        sdk_service_names = [s["service"] for s in sdk_services]
        portal_service_names = [s["service"] for s in portal_services]
        assert sdk_service_names == portal_service_names, \
            f"Service order mismatch: SDK={sdk_service_names}, Portal={portal_service_names}"

        # Correct submitted_via
        assert sdk_db["submitted_via"] == "sdk"
        assert portal_db["submitted_via"] == "portal"

        # Stage timings populated for both
        sdk_timings = json.loads(sdk_db["stage_timings"]) if isinstance(sdk_db.get("stage_timings"), str) else sdk_db.get("stage_timings", [])
        portal_timings = json.loads(portal_db["stage_timings"]) if isinstance(portal_db.get("stage_timings"), str) else portal_db.get("stage_timings", [])
        assert len(sdk_timings) == len(portal_timings), "Timing count mismatch"

        # B2 keys populated for both
        assert sdk_db.get("final_b2_key"), "SDK job missing B2 key"
        assert portal_db.get("final_b2_key"), "Portal job missing B2 key"

        print(f"\n  PARITY CHECK PASSED")
        print(f"  Services: {sdk_service_names}")
        print(f"  Stages: {sdk_db['stages_completed']}/{sdk_db['total_stages']}")
        print(f"  SDK cost: ${sdk_db.get('total_cost_usd', 0)}")
        print(f"  Portal cost: ${portal_db.get('total_cost_usd', 0)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
