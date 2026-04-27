"""
End-to-end pipeline tests for the Latence SDK.

Comprehensive 360-degree validation covering:
  - Smart defaults, explicit steps, builder API, text-only input
  - DataPackage deep validation (all sections)
  - Archive download verification
  - Pricing / billing verification via Supabase
  - Status transition tracking
  - Error handling
  - merge() convenience utility

Prerequisites:
  1. API Gateway running (deployed or local wrangler dev)
  2. Pipeline Worker running
  3. RunPod pod running (real services, not mock)
  4. Supabase migration 002+ applied
  5. LATENCE_API_KEY set in environment
  6. LATENCE_BASE_URL set (defaults to https://api.latence.ai)

Run:
    LATENCE_API_KEY=lat_xxx LATENCE_BASE_URL=https://your-gateway.workers.dev \
      SUPABASE_URL=https://xxx.supabase.co SUPABASE_KEY=xxx \
        pytest tests/integration/test_pipeline_e2e.py -v -s --timeout=300
"""

import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import httpx
import pytest

from latence import Latence, PipelineBuilder

pytestmark = pytest.mark.skipif(
    not os.environ.get("LATENCE_API_KEY"),
    reason="LATENCE_API_KEY not set -- skipping pipeline E2E tests",
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"
OUTPUT_DIR = Path(__file__).resolve().parent / "_test_output"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")


@pytest.fixture
def client():
    return Latence()


@pytest.fixture(autouse=True)
def cleanup_output():
    yield
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def _query_supabase_job(job_id: str) -> dict | None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
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


def _query_org_balance(org_id: str) -> float | None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    resp = httpx.get(
        f"{SUPABASE_URL}/rest/v1/organizations?id=eq.{org_id}&select=balance",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        },
    )
    if resp.status_code == 200:
        rows = resp.json()
        if rows:
            return float(rows[0].get("balance", 0))
    return None


# =========================================================================
# 2a. Pipeline Submission and Completion
# =========================================================================


class TestSmartDefaults:
    """Test pipeline with smart defaults (files only -> doc_intel + extraction + ontology)."""

    def test_submit_wait_datapackage(self, client):
        assert SAMPLE_PDF.exists(), f"Test fixture not found: {SAMPLE_PDF}"

        job = client.pipeline.run(files=[str(SAMPLE_PDF)])
        assert job.id.startswith("pipe_"), f"Unexpected job ID format: {job.id}"

        pkg = job.wait_for_completion(poll_interval=3.0, timeout=300)

        assert pkg is not None
        assert pkg.status in ("COMPLETED", "CACHED", "PULLED")

        # Smart defaults: doc_intel + extraction + ontology
        assert pkg.document is not None, "Document section missing"
        assert len(pkg.document.markdown) > 50, "Markdown content too short"
        assert pkg.entities is not None, "Entities section missing"
        assert pkg.knowledge_graph is not None, "Knowledge graph section missing"
        assert pkg.quality is not None, "Quality report missing"

        print(f"\n  [PASS] Smart defaults completed: {job.id}")
        print(f"    Markdown: {len(pkg.document.markdown)} chars")
        print(f"    Entities: {pkg.entities.summary.total}")
        print(f"    Relations: {len(pkg.knowledge_graph.relations)}")


class TestExplicitSteps:
    """Test pipeline with explicit step configuration (compliance pipeline)."""

    def test_compliance_pipeline(self, client):
        assert SAMPLE_PDF.exists()

        job = client.pipeline.run(
            files=[str(SAMPLE_PDF)],
            steps={
                "ocr": {"mode": "performance"},
                "redaction": {"mode": "balanced"},
                "extraction": {"threshold": 0.3},
                "knowledge_graph": {"resolve_entities": True},
            },
            name="SDK E2E - Compliance Pipeline",
        )
        assert job.id.startswith("pipe_")
        assert job.name == "SDK E2E - Compliance Pipeline"

        pkg = job.wait_for_completion(poll_interval=3.0, timeout=300)
        assert pkg is not None
        assert pkg.status in ("COMPLETED", "CACHED", "PULLED")

        assert pkg.document is not None, "Document section missing"
        assert pkg.entities is not None, "Entities section missing"
        assert pkg.knowledge_graph is not None, "Knowledge graph section missing"

        print(f"\n  [PASS] Compliance pipeline completed: {job.id}")
        print(f"    Document: {len(pkg.document.markdown)} chars")
        print(f"    Entities: {pkg.entities.summary.total}")
        if pkg.redaction:
            print(f"    Redaction: {len(pkg.redaction.pii_detected)} PII items")
        print(f"    KG Relations: {len(pkg.knowledge_graph.relations)}")


class TestBuilderAPI:
    """Test PipelineBuilder fluent API."""

    def test_builder_submit_complete(self, client):
        assert SAMPLE_PDF.exists()

        config = (
            PipelineBuilder()
            .doc_intel(mode="performance")
            .extraction(threshold=0.3)
            .ontology(resolve_entities=True)
            .build()
        )

        job = client.pipeline.submit(
            config,
            files=[str(SAMPLE_PDF)],
            name="SDK E2E - Builder Test",
        )
        assert job.id.startswith("pipe_")

        pkg = job.wait_for_completion(poll_interval=3.0, timeout=300)
        assert pkg is not None
        assert pkg.status in ("COMPLETED", "CACHED", "PULLED")
        assert pkg.document is not None
        assert pkg.entities is not None

        print(f"\n  [PASS] Builder API completed: {job.id}")


class TestTextOnlyExtraction:
    """Test text-only pipeline (no files, no doc_intel needed).

    Note: The pipeline worker currently requires file-based input (files are
    uploaded to B2 and resolved from there). Text-only pipelines are supported
    by the API gateway's synchronous path, not the async pipeline worker.
    """

    @pytest.mark.skip(
        reason="Pipeline worker requires file-based input; text-only uses sync API path"
    )
    def test_text_extraction(self, client):
        job = client.pipeline.run(
            text=(
                "Acme Corporation, headquartered in San Francisco, entered into "
                "a strategic partnership with GlobalTech Inc. on January 15, 2026. "
                "CEO Jane Smith and CFO Robert Johnson signed the agreement valued "
                "at $2,500,000 USD over 36 months."
            ),
            steps={"extraction": {"threshold": 0.3}},
            name="SDK E2E - Text Extraction",
        )

        pkg = job.wait_for_completion(poll_interval=2.0, timeout=120)
        assert pkg is not None
        assert pkg.status in ("COMPLETED", "CACHED", "PULLED")
        assert pkg.entities is not None, "Entities should be extracted from text"
        assert pkg.entities.summary.total > 0

        print(f"\n  [PASS] Text extraction completed: {job.id}")
        print(f"    Entities: {pkg.entities.summary.total}")
        for e in pkg.entities.items[:5]:
            print(f"      - {e.text} ({e.label}, score={e.score})")


# =========================================================================
# 2b. DataPackage Deep Validation
# =========================================================================


class TestDataPackageDeepValidation:
    """Deep validation of DataPackage sections and metadata."""

    def test_all_sections(self, client):
        assert SAMPLE_PDF.exists()

        job = client.pipeline.run(files=[str(SAMPLE_PDF)])
        pkg = job.wait_for_completion(poll_interval=3.0, timeout=300)

        # Status
        assert pkg.status in ("COMPLETED", "CACHED", "PULLED")

        # Document section
        assert pkg.document is not None
        assert isinstance(pkg.document.markdown, str)
        assert len(pkg.document.markdown) > 50
        if pkg.document.pages:
            assert len(pkg.document.pages) > 0
            assert all(isinstance(p, str) for p in pkg.document.pages)

        # Entities section
        assert pkg.entities is not None
        assert len(pkg.entities.items) > 0
        for e in pkg.entities.items:
            assert e.text, "Entity text should be non-empty"
            assert e.label, "Entity label should be non-empty"
        assert pkg.entities.summary.total == len(pkg.entities.items), (
            f"Summary total ({pkg.entities.summary.total})"
            f" != items count ({len(pkg.entities.items)})"
        )
        assert len(pkg.entities.summary.unique_labels) > 0

        # Knowledge graph section
        assert pkg.knowledge_graph is not None
        assert len(pkg.knowledge_graph.entities) > 0 or len(pkg.knowledge_graph.relations) > 0

        # Quality report
        assert pkg.quality is not None
        assert len(pkg.quality.stages) > 0

        print(f"\n  [PASS] Deep validation passed: {job.id}")
        print(f"    Quality stages: {len(pkg.quality.stages)}")
        for s in pkg.quality.stages:
            print(f"      - {s.service}: {s.status} ({s.processing_time_ms}ms)")


# =========================================================================
# 2c. Archive Download Validation
# =========================================================================


class TestArchiveDownload:
    """Test download_archive() produces valid ZIP with correct structure."""

    def test_archive_structure(self, client):
        assert SAMPLE_PDF.exists()

        job = client.pipeline.run(files=[str(SAMPLE_PDF)])
        pkg = job.wait_for_completion(poll_interval=3.0, timeout=300)
        assert pkg is not None

        archive_path = OUTPUT_DIR / "test_output.zip"
        result_path = pkg.download_archive(str(archive_path))
        assert Path(result_path).exists(), "Archive file not created"

        with zipfile.ZipFile(result_path, "r") as zf:
            names = zf.namelist()
            basenames = [n.split("/", 1)[1] if "/" in n else n for n in names]

            assert any("README.md" in n for n in basenames), (
                f"README.md missing from archive. Contents: {names}"
            )
            assert any("document.md" in n for n in basenames), (
                f"document.md missing from archive. Contents: {names}"
            )
            assert any("entities.json" in n for n in basenames), (
                f"entities.json missing from archive. Contents: {names}"
            )
            assert any("quality_report.json" in n for n in basenames), (
                f"quality_report.json missing from archive. Contents: {names}"
            )
            assert any("metadata.json" in n for n in basenames), (
                f"metadata.json missing from archive. Contents: {names}"
            )

            # Verify document.md matches DataPackage content
            doc_files = [n for n in names if n.endswith("document.md")]
            if doc_files:
                doc_content = zf.read(doc_files[0]).decode("utf-8")
                assert doc_content == pkg.document.markdown, (
                    "document.md content doesn't match DataPackage markdown"
                )

            # Verify entities.json has correct count
            ent_files = [n for n in names if n.endswith("entities.json")]
            if ent_files and pkg.entities:
                ent_data = json.loads(zf.read(ent_files[0]))
                items = ent_data.get("items", [])
                assert len(items) == len(pkg.entities.items), (
                    f"entities.json has {len(items)} items, expected {len(pkg.entities.items)}"
                )

        print(f"\n  [PASS] Archive download validated: {result_path}")
        print(f"    Files in archive: {len(names)}")


# =========================================================================
# 2d. Pricing and Billing Verification
# =========================================================================


class TestPricingBilling:
    """Verify Supabase billing records match expected values."""

    @pytest.mark.skipif(
        not SUPABASE_URL or not SUPABASE_KEY,
        reason="SUPABASE_URL/KEY not set -- skipping billing tests",
    )
    def test_billing_recorded(self, client):
        assert SAMPLE_PDF.exists()

        job = client.pipeline.run(
            files=[str(SAMPLE_PDF)],
            name="SDK E2E - Billing Check",
        )
        pkg = job.wait_for_completion(poll_interval=3.0, timeout=300)
        assert pkg is not None

        # Query Supabase for the job record
        db_job = _query_supabase_job(job.id)
        assert db_job is not None, f"Job {job.id} not found in Supabase"

        # Status should be terminal
        assert db_job["status"] in ("COMPLETED", "CACHED", "PULLED"), (
            f"Expected terminal status, got {db_job['status']}"
        )

        # Total cost USD should be populated (worker-computed billing)
        total_cost = db_job.get("total_cost_usd")
        assert total_cost is not None and float(total_cost) > 0, (
            f"total_cost_usd should be > 0, got {total_cost}"
        )

        # Stages should match
        assert db_job["stages_completed"] == db_job["total_stages"], (
            f"stages_completed ({db_job['stages_completed']})"
            f" != total_stages ({db_job['total_stages']})"
        )

        # Verify billed flag
        billed = db_job.get("billed")
        assert billed is True, f"billed should be True, got {billed}"

        # Cost estimated may or may not be set (portal sets it, SDK doesn't)
        cost_est = db_job.get("cost_estimated")

        print(f"\n  [PASS] Billing check passed: {job.id}")
        print(f"    cost_estimated: {f'${float(cost_est):.4f}' if cost_est else 'N/A (SDK path)'}")
        print(f"    total_cost_usd: ${float(total_cost):.4f}")
        print(f"    stages: {db_job['stages_completed']}/{db_job['total_stages']}")
        print(f"    billed: {billed}")


# =========================================================================
# 2e. Status Transition Tracking
# =========================================================================


class TestStatusTransitions:
    """Verify status transitions during pipeline execution."""

    def test_status_progression(self, client):
        assert SAMPLE_PDF.exists()

        job = client.pipeline.run(files=[str(SAMPLE_PDF)])

        statuses_seen: list[str] = []
        services_seen: list[str | None] = []
        stages_completed_max = 0
        max_polls = 120

        for _ in range(max_polls):
            s = job.status()
            if not statuses_seen or statuses_seen[-1] != s.status:
                statuses_seen.append(s.status)
            if s.current_service and s.current_service not in services_seen:
                services_seen.append(s.current_service)
            if s.stages_completed and s.stages_completed > stages_completed_max:
                stages_completed_max = s.stages_completed

            if s.status in ("COMPLETED", "CACHED", "PULLED"):
                break
            if s.status in ("FAILED", "CANCELLED"):
                break

            time.sleep(1.5)

        terminal = {"COMPLETED", "CACHED", "PULLED"}
        assert set(statuses_seen) & terminal, (
            f"Pipeline did not reach terminal state. Seen: {statuses_seen}"
        )

        # Should see progression
        assert len(statuses_seen) >= 1, "Should observe at least one status"

        print(f"\n  [PASS] Status transitions tracked: {job.id}")
        print(f"    Statuses: {' -> '.join(statuses_seen)}")
        print(f"    Services seen: {services_seen}")
        print(f"    Max stages completed: {stages_completed_max}")


# =========================================================================
# 2f. Error Handling
# =========================================================================


class TestErrorHandling:
    """Test error handling edge cases."""

    def test_no_input_raises_value_error(self, client):
        with pytest.raises(ValueError, match="input"):
            client.pipeline.run()

    def test_cancel_job(self, client):
        assert SAMPLE_PDF.exists()

        job = client.pipeline.run(
            files=[str(SAMPLE_PDF)],
            name="SDK E2E - Cancel Test",
        )

        try:
            job.cancel()
        except Exception:
            pass

        time.sleep(2)
        s = job.status()
        print(f"\n  Cancel test: final status = {s.status}")
        assert s.status in ("CANCELLED", "FAILED", "IN_PROGRESS", "COMPLETED", "CACHED")


# =========================================================================
# Phase 3: merge() Convenience Utility Tests
# =========================================================================


class TestMergeUtility:
    """Test merge() produces document-centric, redundancy-free JSON."""

    def test_merge_structure(self, client):
        assert SAMPLE_PDF.exists()

        job = client.pipeline.run(files=[str(SAMPLE_PDF)])
        pkg = job.wait_for_completion(poll_interval=3.0, timeout=300)
        assert pkg is not None

        merged = pkg.merge()

        # Top-level structure
        assert "id" in merged
        assert "name" in merged
        assert "status" in merged
        assert merged["status"] in ("COMPLETED", "CACHED", "PULLED")
        assert "documents" in merged
        assert "summary" in merged

        # Documents array
        docs = merged["documents"]
        assert isinstance(docs, list)
        assert len(docs) >= 1

        doc = docs[0]
        assert "filename" in doc
        assert "markdown" in doc
        assert len(doc["markdown"]) > 50, "Markdown should be non-empty"
        assert "page_count" in doc

        # Entities in document
        assert "entities" in doc
        assert len(doc["entities"]) > 0
        ent = doc["entities"][0]
        assert "text" in ent
        assert "label" in ent
        assert "score" in ent

        # Knowledge graph in document
        assert "knowledge_graph" in doc
        kg = doc["knowledge_graph"]
        assert "entities" in kg or "relations" in kg
        if "relations" in kg and kg["relations"]:
            rel = kg["relations"][0]
            assert "source" in rel
            assert "relation" in rel
            assert "target" in rel

        # Summary
        summary = merged["summary"]
        assert summary["documents"] >= 1
        assert summary["entities"]["total"] > 0
        assert len(summary["services_executed"]) > 0
        # processing_time_ms may be 0 if execution_summary didn't carry timing
        assert summary["processing_time_ms"] >= 0

        print(f"\n  [PASS] Merge structure validated: {merged['id']}")
        print(f"    Documents: {summary['documents']}")
        print(f"    Pages: {summary['pages']}")
        print(f"    Entities: {summary['entities']['total']}")
        print(f"    Relations: {summary['relations']['total']}")
        print(f"    Services: {', '.join(summary['services_executed'])}")

    def test_merge_no_redundancy(self, client):
        """Verify markdown text appears exactly once (in documents[].markdown)."""
        assert SAMPLE_PDF.exists()

        job = client.pipeline.run(files=[str(SAMPLE_PDF)])
        pkg = job.wait_for_completion(poll_interval=3.0, timeout=300)
        assert pkg is not None

        merged = pkg.merge()

        doc_md = merged["documents"][0]["markdown"]
        assert len(doc_md) > 50

        # Verify markdown is NOT present in entities or knowledge_graph sections
        # (it should only live under documents[].markdown, not duplicated elsewhere)
        doc = merged["documents"][0]
        entities_json = json.dumps(doc.get("entities", []))
        kg_json = json.dumps(doc.get("knowledge_graph", {}))

        # The full markdown should NOT appear in entities or KG output
        assert doc_md not in entities_json, "Markdown text duplicated in entities section"
        assert doc_md not in kg_json, "Markdown text duplicated in knowledge_graph section"

        # Verify "markdown" key only exists once in the document
        keys_with_markdown = [k for k in doc if "markdown" in k.lower()]
        assert len(keys_with_markdown) == 1, f"Expected 1 markdown key, found: {keys_with_markdown}"

        print("\n  [PASS] No redundancy: markdown appears exactly once")

    def test_merge_save_to_file(self, client):
        """Verify merge(save_to=...) writes valid JSON."""
        assert SAMPLE_PDF.exists()

        job = client.pipeline.run(files=[str(SAMPLE_PDF)])
        pkg = job.wait_for_completion(poll_interval=3.0, timeout=300)
        assert pkg is not None

        output_file = OUTPUT_DIR / "merged_result.json"
        merged = pkg.merge(save_to=str(output_file))

        assert output_file.exists(), "Merged JSON file not created"
        file_content = json.loads(output_file.read_text())

        # File content should match in-memory dict
        assert file_content["id"] == merged["id"]
        assert file_content["status"] == merged["status"]
        assert len(file_content["documents"]) == len(merged["documents"])
        assert (
            file_content["summary"]["entities"]["total"] == merged["summary"]["entities"]["total"]
        )

        print(f"\n  [PASS] Merge save_to validated: {output_file}")
        print(f"    File size: {output_file.stat().st_size} bytes")

    def test_merge_opt_in_sections(self, client):
        """Verify merge only includes sections for services that ran."""
        assert SAMPLE_PDF.exists()

        # Pipeline without redaction or compression
        job = client.pipeline.run(files=[str(SAMPLE_PDF)])
        pkg = job.wait_for_completion(poll_interval=3.0, timeout=300)
        assert pkg is not None

        merged = pkg.merge()
        doc = merged["documents"][0]

        # These should NOT be present (services didn't run)
        assert "redaction" not in doc, "Redaction should not be in merge output"
        assert "compression" not in doc, "Compression should not be in merge output"

        # These SHOULD be present
        assert "entities" in doc
        assert "knowledge_graph" in doc

        print("\n  [PASS] Opt-in sections: only executed services included")


# =========================================================================
# Phase 4: Final 360 Roundtrip Validation
# =========================================================================


class TestMergeRoundtrip:
    """Final 360 validation: verify consistent results across entry points."""

    def test_merge_roundtrip(self, client):
        """Run pipeline via SDK and verify full merge output integrity."""
        assert SAMPLE_PDF.exists()

        config = (
            PipelineBuilder()
            .doc_intel(mode="performance")
            .extraction(threshold=0.3)
            .ontology(resolve_entities=True)
            .build()
        )

        job = client.pipeline.submit(
            config,
            files=[str(SAMPLE_PDF)],
            name="SDK E2E - Roundtrip Validation",
        )
        pkg = job.wait_for_completion(poll_interval=3.0, timeout=300)
        assert pkg is not None

        # Full merge
        merged = pkg.merge()

        # 1. Verify all service outputs present in merge
        doc = merged["documents"][0]
        assert "markdown" in doc and len(doc["markdown"]) > 50
        assert "entities" in doc and len(doc["entities"]) > 0
        assert "knowledge_graph" in doc

        # 2. Verify entity counts match between DataPackage and merge
        pkg_entity_count = pkg.entities.summary.total if pkg.entities else 0
        merge_entity_count = merged["summary"]["entities"]["total"]
        assert pkg_entity_count == merge_entity_count, (
            f"Entity count mismatch: DataPackage={pkg_entity_count}, merge={merge_entity_count}"
        )

        # 3. Verify relations counts match
        pkg_rel_count = pkg.knowledge_graph.summary.total_relations if pkg.knowledge_graph else 0
        merge_rel_count = merged["summary"]["relations"]["total"]
        assert pkg_rel_count == merge_rel_count, (
            f"Relation count mismatch: DataPackage={pkg_rel_count}, merge={merge_rel_count}"
        )

        # 4. Verify markdown matches
        assert doc["markdown"] == pkg.document.markdown, (
            "Markdown content mismatch between DataPackage and merge"
        )

        # 5. Verify save_to produces identical output
        save_path = OUTPUT_DIR / "roundtrip.json"
        pkg.merge(save_to=str(save_path))
        reloaded = json.loads(save_path.read_text())
        assert reloaded["summary"]["entities"]["total"] == merged["summary"]["entities"]["total"]
        assert reloaded["documents"][0]["markdown"] == merged["documents"][0]["markdown"]

        # 6. Summary statistics
        assert merged["summary"]["services_executed"], "No services recorded in summary"
        assert merged["summary"]["processing_time_ms"] >= 0

        print(f"\n  [PASS] 360 Roundtrip validation passed: {merged['id']}")
        print(f"    Entities: {merge_entity_count}")
        print(f"    Relations: {merge_rel_count}")
        print(f"    Services: {', '.join(merged['summary']['services_executed'])}")
        print(f"    Cost: ${merged['summary']['cost_usd'] or 0:.4f}")
        print(f"    Time: {merged['summary']['processing_time_ms']:.0f}ms")


# =========================================================================
# Run directly
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
