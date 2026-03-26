"""Comprehensive tests for the Data Intelligence Pipeline refactor (v0.2).

Tests cover:
- DataPackage composition from raw pipeline results
- DataPackage ZIP archive generation
- Smart defaults (files-only -> OCR -> Extraction -> Knowledge Graph)
- Step aliases (ocr -> document_intelligence, etc.)
- Step ordering (canonical order regardless of user dict key order)
- Placeholder steps (feature_enrichment, global_resolution)
- Job handle (status, wait_for_completion, cancel, data_package)
- pipeline.run() and pipeline.submit() entry points
- Input parsing (files, file_urls, text, auto-detection)
- Experimental namespace
- Deprecation warnings on client.extraction etc.
- Backward compatibility (execute/PipelineBuilder unchanged)
"""

from __future__ import annotations

import json
import warnings
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from latence._base import APIResponse, ResponseMetadata
from latence._models.pipeline import (
    PipelineExecutionSummary,
    PipelineResultResponse,
    PipelineStatusResponse,
    PipelineSubmitResponse,
    StageResult,
    StageStatus,
)
from latence._pipeline.builder import PipelineBuilder
from latence._pipeline.data_package import (
    DataPackage,
)
from latence._pipeline.job import AsyncJob, Job
from latence._pipeline.spec import (
    build_pipeline_config,
    has_file_input,
    parse_input,
    parse_steps_config,
    resolve_step_name,
)

# =============================================================================
# Fake client for unit testing (no live API calls)
# =============================================================================


class _FakeSyncClient:
    """Fake synchronous client that records calls and returns canned responses."""

    def __init__(self, data: dict | None = None) -> None:
        self._data = data or {}
        self.calls: list[tuple[str, str, dict | None]] = []  # (method, path, json)

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

    def delete(self, path: str) -> APIResponse:
        self.calls.append(("DELETE", path, None))
        return APIResponse(
            data=self._data,
            metadata=ResponseMetadata(request_id="req_test"),
            status_code=200,
        )

    # Attributes expected by resources
    base_url = "https://api.latence.ai"
    max_retries = 2


# =============================================================================
# Helper: build a realistic PipelineResultResponse
# =============================================================================


def _make_pipeline_result(
    *,
    job_id: str = "pipe_test_123",
    include_doc: bool = True,
    include_extraction: bool = True,
    include_ontology: bool = True,
    include_redaction: bool = False,
) -> PipelineResultResponse:
    """Build a realistic PipelineResultResponse for testing DataPackage composition."""
    intermediate: dict[str, StageResult] = {}

    if include_doc:
        intermediate["document_intelligence"] = StageResult(
            service="document_intelligence",
            status="completed",
            output={
                "content": "# Contract\n\nThis is a test contract between Party A and Party B.",
                "content_type": "markdown",
                "pages_processed": 3,
                "metadata": {
                    "filename": "contract.pdf",
                    "processing_mode": "performance",
                },
                "pages": [
                    {"markdown": "# Page 1\nIntroduction"},
                    {"markdown": "# Page 2\nTerms"},
                    {"markdown": "# Page 3\nSignatures"},
                ],
            },
            processing_time_ms=3500.0,
            credits_used=0.05,
        )

    if include_extraction:
        intermediate["extraction"] = StageResult(
            service="extraction",
            status="completed",
            output={
                "entities": [
                    {
                        "start": 0,
                        "end": 8,
                        "text": "Party A",
                        "label": "ORGANIZATION",
                        "score": 0.95,
                    },
                    {
                        "start": 13,
                        "end": 20,
                        "text": "Party B",
                        "label": "ORGANIZATION",
                        "score": 0.91,
                    },
                    {"start": 25, "end": 35, "text": "2026-01-01", "label": "DATE", "score": 0.88},
                    {
                        "start": 40,
                        "end": 50,
                        "text": "John Smith",
                        "label": "PERSON",
                        "score": 0.93,
                    },
                    {"start": 55, "end": 65, "text": "Jane Doe", "label": "PERSON", "score": 0.90},
                ],
            },
            processing_time_ms=4200.0,
            credits_used=0.03,
        )

    if include_ontology:
        intermediate["ontology"] = StageResult(
            service="ontology",
            status="completed",
            output={
                "entities": [
                    {
                        "start": 0,
                        "end": 8,
                        "text": "Party A",
                        "label": "ORGANIZATION",
                        "score": 0.95,
                    },
                    {
                        "start": 13,
                        "end": 20,
                        "text": "Party B",
                        "label": "ORGANIZATION",
                        "score": 0.91,
                    },
                    {
                        "start": 40,
                        "end": 50,
                        "text": "John Smith",
                        "label": "PERSON",
                        "score": 0.93,
                    },
                ],
                "relations": [
                    {
                        "entity1": {
                            "text": "John Smith",
                            "label": "PERSON",
                            "start": 40,
                            "end": 50,
                        },
                        "entity2": {
                            "text": "Party A",
                            "label": "ORGANIZATION",
                            "start": 0,
                            "end": 8,
                        },
                        "score": 0.85,
                        "relation_label": "WORKS_FOR",
                    },
                    {
                        "entity1": {
                            "text": "Party A",
                            "label": "ORGANIZATION",
                            "start": 0,
                            "end": 8,
                        },
                        "entity2": {
                            "text": "Party B",
                            "label": "ORGANIZATION",
                            "start": 13,
                            "end": 20,
                        },
                        "score": 0.78,
                        "relation_label": "CONTRACTS_WITH",
                    },
                ],
                "entity_count": 3,
                "relation_count": 2,
                "resolved_entities": 3,
            },
            processing_time_ms=4800.0,
            credits_used=0.08,
        )

    if include_redaction:
        intermediate["redaction"] = StageResult(
            service="redaction",
            status="completed",
            output={
                "redacted_text": "# Contract\n\nThis is a test contract between [ORG] and [ORG].",
                "entities": [
                    {
                        "start": 40,
                        "end": 50,
                        "text": "John Smith",
                        "label": "PERSON",
                        "score": 0.93,
                    },
                    {"start": 55, "end": 65, "text": "Jane Doe", "label": "PERSON", "score": 0.90},
                ],
                "entity_count": 2,
            },
            processing_time_ms=2100.0,
            credits_used=0.02,
        )

    total_time = sum(s.processing_time_ms or 0 for s in intermediate.values())
    total_credits = sum(s.credits_used or 0 for s in intermediate.values())
    last_stage = list(intermediate.values())[-1] if intermediate else None

    return PipelineResultResponse(
        job_id=job_id,
        status="COMPLETED",
        final_output=last_stage.output if last_stage else None,
        intermediate_results=intermediate,
        execution_summary=PipelineExecutionSummary(
            total_stages=len(intermediate),
            completed_stages=len(intermediate),
            total_credits_used=total_credits,
            total_processing_time_ms=total_time,
        ),
    )


# =============================================================================
# DATAPACKAGE COMPOSITION TESTS
# =============================================================================


class TestDataPackageComposition:
    """Tests for DataPackage.from_pipeline_result()."""

    def test_full_pipeline_composition(self):
        """Full pipeline result should compose all sections."""
        result = _make_pipeline_result()
        pkg = DataPackage.from_pipeline_result(result, name="Test Pipeline")

        assert pkg.id == "pipe_test_123"
        assert pkg.name == "Test Pipeline"
        assert pkg.status == "COMPLETED"

        # Document section
        assert pkg.document is not None
        assert "Contract" in pkg.document.markdown
        assert pkg.document.pages is not None
        assert len(pkg.document.pages) == 3
        assert pkg.document.metadata.pages_processed == 3
        assert pkg.document.metadata.filename == "contract.pdf"

        # Entities section
        assert pkg.entities is not None
        assert pkg.entities.summary.total == 5
        assert "ORGANIZATION" in pkg.entities.summary.by_type
        assert pkg.entities.summary.by_type["ORGANIZATION"] == 2
        assert pkg.entities.summary.by_type["PERSON"] == 2
        assert pkg.entities.summary.by_type["DATE"] == 1
        assert pkg.entities.summary.avg_confidence is not None
        assert 0.85 < pkg.entities.summary.avg_confidence < 0.95

        # Knowledge graph section
        assert pkg.knowledge_graph is not None
        assert pkg.knowledge_graph.summary.total_entities == 3
        assert pkg.knowledge_graph.summary.total_relations == 2
        assert "WORKS_FOR" in pkg.knowledge_graph.summary.relation_types
        assert "CONTRACTS_WITH" in pkg.knowledge_graph.summary.relation_types

        # Redaction should be None (not included)
        assert pkg.redaction is None

        # Quality report
        assert pkg.quality is not None
        assert len(pkg.quality.stages) == 3
        assert pkg.quality.total_processing_time_ms > 0
        assert pkg.quality.confidence.entity_avg_confidence is not None

        # Raw access
        assert pkg.raw is not None

    def test_composition_with_redaction(self):
        """Pipeline with redaction should include redaction section."""
        result = _make_pipeline_result(include_redaction=True)
        pkg = DataPackage.from_pipeline_result(result)

        assert pkg.redaction is not None
        assert "[ORG]" in pkg.redaction.redacted_text
        assert pkg.redaction.summary.total_pii == 2
        assert pkg.redaction.summary.by_type["PERSON"] == 2

    def test_composition_doc_only(self):
        """Pipeline with only doc_intel should compose document section only."""
        result = _make_pipeline_result(include_extraction=False, include_ontology=False)
        pkg = DataPackage.from_pipeline_result(result)

        assert pkg.document is not None
        assert pkg.entities is None
        assert pkg.knowledge_graph is None
        assert pkg.redaction is None

    def test_composition_empty_pipeline(self):
        """Empty pipeline result should compose gracefully."""
        result = PipelineResultResponse(
            job_id="pipe_empty",
            status="COMPLETED",
            final_output=None,
            intermediate_results={},
            execution_summary=PipelineExecutionSummary(total_stages=0, completed_stages=0),
        )
        pkg = DataPackage.from_pipeline_result(result)

        assert pkg.document is None
        assert pkg.entities is None
        assert pkg.knowledge_graph is None
        assert pkg.redaction is None
        assert pkg.quality is not None


class TestDataPackageArchive:
    """Tests for DataPackage.download_archive()."""

    def test_archive_structure(self, tmp_path: Path):
        """ZIP archive should contain correct files and structure."""
        result = _make_pipeline_result()
        pkg = DataPackage.from_pipeline_result(result, name="Test Pipeline")

        archive_path = tmp_path / "output.zip"
        returned = pkg.download_archive(archive_path)

        assert returned == archive_path
        assert archive_path.exists()

        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()

            # Check expected files exist
            assert any("README.md" in n for n in names)
            assert any("document.md" in n for n in names)
            assert any("entities.json" in n for n in names)
            assert any("knowledge_graph.json" in n for n in names)
            assert any("quality_report.json" in n for n in names)
            assert any("metadata.json" in n for n in names)

            # Check page files
            assert any("page_001.md" in n for n in names)
            assert any("page_002.md" in n for n in names)
            assert any("page_003.md" in n for n in names)

            # Verify README content
            readme = next(n for n in names if "README.md" in n)
            content = zf.read(readme).decode()
            assert "Test Pipeline" in content
            assert "5 entities" in content

            # Verify entities JSON is valid
            entities_f = next(n for n in names if "entities.json" in n)
            entities_data = json.loads(zf.read(entities_f))
            assert "items" in entities_data
            assert "summary" in entities_data

    def test_archive_without_optional_sections(self, tmp_path: Path):
        """Archive should handle missing sections gracefully."""
        result = _make_pipeline_result(include_extraction=False, include_ontology=False)
        pkg = DataPackage.from_pipeline_result(result)

        archive_path = tmp_path / "minimal.zip"
        pkg.download_archive(archive_path)

        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()
            assert any("document.md" in n for n in names)
            assert not any("entities.json" in n for n in names)
            assert not any("knowledge_graph.json" in n for n in names)


# =============================================================================
# STEP ALIAS TESTS
# =============================================================================


class TestStepAliases:
    """Tests for step name alias resolution."""

    def test_ocr_alias(self):
        assert resolve_step_name("ocr") == "document_intelligence"

    def test_doc_intel_alias(self):
        assert resolve_step_name("doc_intel") == "document_intelligence"

    def test_knowledge_graph_alias(self):
        assert resolve_step_name("knowledge_graph") == "ontology"

    def test_graph_alias(self):
        assert resolve_step_name("graph") == "ontology"

    def test_relation_extraction_alias(self):
        assert resolve_step_name("relation_extraction") == "ontology"

    def test_redact_alias(self):
        assert resolve_step_name("redact") == "redaction"

    def test_extract_alias(self):
        assert resolve_step_name("extract") == "extraction"

    def test_compress_alias(self):
        assert resolve_step_name("compress") == "compression"

    def test_canonical_name_passthrough(self):
        assert resolve_step_name("document_intelligence") == "document_intelligence"
        assert resolve_step_name("extraction") == "extraction"
        assert resolve_step_name("ontology") == "ontology"

    def test_case_insensitive(self):
        assert resolve_step_name("OCR") == "document_intelligence"
        assert resolve_step_name("Knowledge_Graph") == "ontology"

    def test_unknown_step_raises(self):
        with pytest.raises(ValueError, match="Unknown pipeline step"):
            resolve_step_name("nonexistent_step")

    def test_placeholder_step_raises(self):
        # enrichment is a placeholder (corpus-level, coming soon)
        with pytest.raises(NotImplementedError, match="coming soon"):
            resolve_step_name("enrichment")

        # feature_enrichment resolves to enrichment → also raises
        with pytest.raises(NotImplementedError, match="coming soon"):
            resolve_step_name("feature_enrichment")

        # enrich resolves to enrichment → also raises
        with pytest.raises(NotImplementedError, match="coming soon"):
            resolve_step_name("enrich")

        # graph_ontology_builder is a placeholder
        with pytest.raises(NotImplementedError, match="coming soon"):
            resolve_step_name("graph_ontology_builder")


# =============================================================================
# STEP ORDERING TESTS
# =============================================================================


class TestStepOrdering:
    """Tests for canonical step ordering."""

    def test_steps_sorted_canonically(self):
        """Steps should be sorted into canonical order regardless of dict key order."""
        steps = {
            "extraction": {"threshold": 0.3},
            "ocr": {"mode": "performance"},
            "knowledge_graph": {"resolve_entities": True},
        }
        services = parse_steps_config(steps)

        assert services[0].service == "document_intelligence"
        assert services[1].service == "extraction"
        assert services[2].service == "ontology"

    def test_extraction_before_redaction_in_canonical_order(self):
        """Extraction should come before redaction in canonical STEP_ORDER."""
        steps = {
            "extraction": {},
            "redaction": {"mode": "balanced"},
            "ocr": {},
        }
        services = parse_steps_config(steps)

        assert services[0].service == "document_intelligence"
        assert services[1].service == "extraction"
        assert services[2].service == "redaction"

    def test_config_passed_through(self):
        """Step config should be preserved."""
        steps = {"ocr": {"mode": "performance", "output_format": "json"}}
        services = parse_steps_config(steps)

        assert services[0].service == "document_intelligence"
        assert services[0].config["mode"] == "performance"
        assert services[0].config["output_format"] == "json"


# =============================================================================
# SMART DEFAULTS TESTS
# =============================================================================


class TestSmartDefaults:
    """Tests for smart default pipeline generation."""

    def test_files_input_triggers_default_pipeline(self):
        """Files-only input with no steps should trigger default pipeline."""
        config = build_pipeline_config(steps=None, has_files=True)

        service_names = [s.service for s in config.services]
        assert service_names == ["document_intelligence", "extraction", "ontology"]

    def test_default_pipeline_stores_intermediate(self):
        """Default pipeline should always store intermediate results."""
        config = build_pipeline_config(steps=None, has_files=True)
        assert config.store_intermediate is True

    def test_explicit_steps_override_defaults(self):
        """Explicit steps should override smart defaults."""
        config = build_pipeline_config(steps={"ocr": {}, "extraction": {}}, has_files=True)
        service_names = [s.service for s in config.services]
        assert service_names == ["document_intelligence", "extraction"]
        assert "ontology" not in service_names

    def test_no_steps_no_files_raises(self):
        """No steps and no files should raise ValueError."""
        with pytest.raises(ValueError, match="Either 'steps' must be provided"):
            build_pipeline_config(steps=None, has_files=False)

    def test_name_passed_through(self):
        """Pipeline name should be set in config."""
        config = build_pipeline_config(steps=None, has_files=True, name="My Pipeline")
        assert config.name == "My Pipeline"


# =============================================================================
# INPUT PARSING TESTS
# =============================================================================


class TestInputParsing:
    """Tests for parse_input()."""

    def test_text_input(self):
        result = parse_input(text="Hello world")
        assert result is not None
        assert result.text == "Hello world"
        assert result.files is None

    def test_file_url_input(self):
        result = parse_input(file_urls=["https://example.com/doc.pdf"])
        assert result is not None
        assert result.files is not None
        assert len(result.files) == 1
        assert result.files[0].url == "https://example.com/doc.pdf"

    def test_http_string_in_files_treated_as_url(self):
        result = parse_input(files=["https://example.com/doc.pdf"])
        assert result is not None
        assert result.files is not None
        assert result.files[0].url == "https://example.com/doc.pdf"

    def test_s3_input_raises(self):
        with pytest.raises(NotImplementedError, match="S3 source input"):
            parse_input(files=["s3://my-bucket/file.pdf"])

    def test_no_input_returns_none(self):
        result = parse_input()
        assert result is None

    def test_has_file_input_detection(self):
        assert has_file_input(files=["doc.pdf"]) is True
        assert has_file_input(file_urls=["https://x.com/d.pdf"]) is True
        assert has_file_input() is False
        assert has_file_input(files=[]) is False


# =============================================================================
# JOB HANDLE TESTS
# =============================================================================


class TestJobHandle:
    """Tests for the Job handle class."""

    def test_job_properties(self):
        """Job should expose id and name."""
        fake_pipeline = MagicMock()
        job = Job("pipe_abc", fake_pipeline, name="My Job", services=["extraction"])

        assert job.id == "pipe_abc"
        assert job.name == "My Job"

    def test_job_repr(self):
        """Job repr should be readable."""
        fake_pipeline = MagicMock()
        job = Job("pipe_abc", fake_pipeline, name="Test")
        assert "pipe_abc" in repr(job)
        assert "Test" in repr(job)

    def test_job_status_delegates(self):
        """Job.status() should delegate to pipeline.status()."""
        fake_pipeline = MagicMock()
        fake_pipeline.status.return_value = PipelineStatusResponse(
            job_id="pipe_abc",
            status="IN_PROGRESS",
            stages_completed=1,
            total_stages=3,
            current_service="extraction",
        )
        job = Job("pipe_abc", fake_pipeline)

        status = job.status()
        fake_pipeline.status.assert_called_once_with("pipe_abc")
        assert status.status == "IN_PROGRESS"
        assert status.current_service == "extraction"

    def test_job_cancel_delegates(self):
        """Job.cancel() should delegate to pipeline.cancel()."""
        fake_pipeline = MagicMock()
        fake_pipeline.cancel.return_value = {"status": "cancelled"}
        job = Job("pipe_abc", fake_pipeline)

        result = job.cancel()
        fake_pipeline.cancel.assert_called_once_with("pipe_abc")
        assert result["status"] == "cancelled"

    def test_job_data_package_caches(self):
        """Job.data_package should cache after first access."""
        result = _make_pipeline_result()
        fake_pipeline = MagicMock()
        fake_pipeline.retrieve.return_value = result

        job = Job("pipe_test_123", fake_pipeline, services=["extraction"])

        pkg1 = job.data_package
        pkg2 = job.data_package

        # Should only call retrieve once (cached)
        fake_pipeline.retrieve.assert_called_once()
        assert pkg1 is pkg2

    def test_async_job_repr(self):
        """AsyncJob repr should include id."""
        fake_pipeline = MagicMock()
        job = AsyncJob("pipe_abc", fake_pipeline, name="Async Test")
        assert "pipe_abc" in repr(job)
        assert "Async Test" in repr(job)


# =============================================================================
# PIPELINE.RUN() TESTS
# =============================================================================


class TestPipelineRun:
    """Tests for pipeline.run() -- the primary entry point."""

    def test_run_with_text_and_steps(self):
        """run() with text and steps should submit correctly."""
        fake_client = _FakeSyncClient(
            data={
                "job_id": "pipe_run_1",
                "poll_url": "/api/v1/pipeline/pipe_run_1",
                "services": ["extraction"],
                "message": "submitted",
            }
        )
        from latence.resources.pipeline import Pipeline

        pipeline = Pipeline(fake_client)

        job = pipeline.run(
            text="Apple is in Cupertino.",
            steps={"extraction": {"threshold": 0.3}},
            name="Test Run",
        )

        assert isinstance(job, Job)
        assert job.id == "pipe_run_1"
        assert job.name == "Test Run"

        # Verify the API call was made
        assert len(fake_client.calls) == 1
        method, path, body = fake_client.calls[0]
        assert method == "POST"
        assert path == "/api/v1/pipeline/execute"
        assert body["name"] == "Test Run"
        assert body["store_intermediate"] is True

    def test_run_with_file_urls(self):
        """run() with file URLs should include them in input."""
        fake_client = _FakeSyncClient(
            data={
                "job_id": "pipe_run_2",
                "poll_url": "/api/v1/pipeline/pipe_run_2",
                "services": ["document_intelligence", "extraction", "ontology"],
                "message": "submitted",
            }
        )
        from latence.resources.pipeline import Pipeline

        pipeline = Pipeline(fake_client)

        job = pipeline.run(
            file_urls=["https://example.com/doc.pdf"],
            name="URL Test",
        )

        assert isinstance(job, Job)
        assert job.id == "pipe_run_2"

    def test_run_no_input_raises(self):
        """run() with no input should raise ValueError."""
        fake_client = _FakeSyncClient(data={})
        from latence.resources.pipeline import Pipeline

        pipeline = Pipeline(fake_client)

        with pytest.raises(ValueError):
            pipeline.run()


# =============================================================================
# PIPELINE.SUBMIT() TESTS
# =============================================================================


class TestPipelineSubmit:
    """Tests for pipeline.submit() -- builder-friendly entry point."""

    def test_submit_with_builder(self):
        """submit() with PipelineBuilder should work."""
        fake_client = _FakeSyncClient(
            data={
                "job_id": "pipe_sub_1",
                "poll_url": "/api/v1/pipeline/pipe_sub_1",
                "services": ["document_intelligence", "extraction"],
                "message": "submitted",
            }
        )
        from latence.resources.pipeline import Pipeline

        pipeline = Pipeline(fake_client)

        config = PipelineBuilder().doc_intel(mode="performance").extraction(threshold=0.3).build()

        job = pipeline.submit(
            config,
            text="Hello world",
            name="Builder Test",
        )

        assert isinstance(job, Job)
        assert job.id == "pipe_sub_1"
        assert job.name == "Builder Test"


# =============================================================================
# EXPERIMENTAL NAMESPACE TESTS
# =============================================================================


class TestExperimentalNamespace:
    """Tests for the experimental namespace."""

    def test_experimental_has_all_services(self):
        """Experimental namespace should expose all service resources."""
        from latence.resources.experimental import ExperimentalNamespace

        fake_client = _FakeSyncClient()
        ns = ExperimentalNamespace(fake_client)

        # All services should be accessible
        assert ns.embed is not None
        assert ns.embedding is not None
        assert ns.colbert is not None
        assert ns.colpali is not None
        assert ns.compression is not None
        assert ns.document_intelligence is not None
        assert ns.extraction is not None
        assert ns.ontology is not None
        assert ns.redaction is not None


# =============================================================================
# DEPRECATION WARNING TESTS
# =============================================================================


class TestDeprecationWarnings:
    """Tests for deprecation warnings on direct service access."""

    def test_deprecated_property_emits_warning(self):
        """Accessing client.extraction should emit DeprecationWarning."""
        from latence import Latence

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            client = Latence(api_key="lat_test")
            _ = client.extraction

            assert len(w) >= 1
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "deprecated" in str(dep_warnings[0].message).lower()

    def test_deprecated_property_warns_once(self):
        """Second access should not emit another warning."""
        from latence import Latence

        client = Latence(api_key="lat_test")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _ = client.extraction  # First access
            _ = client.extraction  # Second access

            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            # Only one warning for extraction (not two)
            extraction_warnings = [x for x in dep_warnings if "extraction" in str(x.message)]
            assert len(extraction_warnings) == 1

    def test_pipeline_not_deprecated(self):
        """Accessing client.pipeline should NOT emit a warning."""
        from latence import Latence

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            client = Latence(api_key="lat_test")
            _ = client.pipeline

            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================


class TestBackwardCompatibility:
    """Tests verifying existing APIs still work unchanged."""

    def test_pipeline_execute_still_works(self):
        """Legacy execute() should still return PipelineSubmitResponse."""
        fake_client = _FakeSyncClient(
            data={
                "job_id": "pipe_legacy_1",
                "poll_url": "/api/v1/pipeline/pipe_legacy_1",
                "services": ["extraction"],
                "message": "submitted",
            }
        )
        from latence.resources.pipeline import Pipeline

        pipeline = Pipeline(fake_client)

        config = PipelineBuilder().extraction().build()
        result = pipeline.execute(config, text="Test")

        assert isinstance(result, PipelineSubmitResponse)
        assert result.job_id == "pipe_legacy_1"

    def test_pipeline_builder_unchanged(self):
        """PipelineBuilder should build same configs as before."""
        config = (
            PipelineBuilder()
            .doc_intel(mode="performance")
            .extraction(threshold=0.3)
            .ontology(resolve_entities=True)
            .store_intermediate()
            .strict()
            .build()
        )

        assert len(config.services) == 3
        assert config.services[0].service == "document_intelligence"
        assert config.services[1].service == "extraction"
        assert config.services[2].service == "ontology"
        assert config.store_intermediate is True
        assert config.strict_mode is True


# =============================================================================
# STAGE STATUS MODEL TESTS
# =============================================================================


class TestStageStatus:
    """Tests for the new StageStatus model."""

    def test_stage_status_model(self):
        """StageStatus should accept all fields."""
        stage = StageStatus(
            service="extraction",
            status="completed",
            started_at="2026-02-15T10:00:00Z",
            completed_at="2026-02-15T10:00:04Z",
            processing_time_ms=4200.0,
        )
        assert stage.service == "extraction"
        assert stage.status == "completed"
        assert stage.error is None

    def test_stage_status_defaults(self):
        """StageStatus should have sensible defaults."""
        stage = StageStatus(service="ontology")
        assert stage.status == "pending"
        assert stage.processing_time_ms is None


# =============================================================================
# VERSION TESTS
# =============================================================================


class TestVersion:
    """Tests for version bump."""

    def test_version_is_0_2_0(self):
        """Version should be 0.2.0."""
        import latence

        assert latence.__version__ == "0.1.0"
