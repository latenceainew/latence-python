"""Tests for the Pipeline Builder, Spec, and Validator."""

import pytest

from latence._models.pipeline import (
    FileInput,
    PipelineConfig,
    PipelineInput,
    ServiceConfig,
)
from latence._pipeline.builder import PipelineBuilder
from latence._pipeline.spec import (
    SERVICE_PARENT,
    STEP_ORDER,
    _topological_sort,
    parse_steps_config,
    resolve_step_name,
)
from latence._pipeline.validator import (
    SERVICE_IO,
    PipelineValidationError,
    _detect_input_type,
    validate_pipeline,
)

# =============================================================================
# SPEC / DAG TESTS
# =============================================================================


class TestServiceParentDAG:
    """Tests for the SERVICE_PARENT DAG model."""

    def test_dag_mirrors_worker(self):
        """SERVICE_PARENT must exactly mirror the worker config."""
        assert SERVICE_PARENT["document_intelligence"] is None
        assert SERVICE_PARENT["extraction"] == "document_intelligence"
        assert SERVICE_PARENT["redaction"] == "document_intelligence"
        assert SERVICE_PARENT["compression"] == "document_intelligence"
        assert SERVICE_PARENT["ontology"] == "extraction"
        assert SERVICE_PARENT["embedding"] == "document_intelligence"
        assert SERVICE_PARENT["colbert"] == "document_intelligence"
        assert SERVICE_PARENT["colpali"] == "document_intelligence"

    def test_all_dag_services_in_step_order(self):
        """Every service in DAG should appear in STEP_ORDER."""
        for service in SERVICE_PARENT:
            assert service in STEP_ORDER, f"{service} in DAG but not in STEP_ORDER"

    def test_dag_has_single_root(self):
        """Only document_intelligence should have None parent."""
        roots = [s for s, p in SERVICE_PARENT.items() if p is None]
        assert roots == ["document_intelligence"]

    def test_all_parents_exist_in_dag(self):
        """Every parent referenced must itself be in the DAG."""
        for service, parent in SERVICE_PARENT.items():
            if parent is not None:
                assert parent in SERVICE_PARENT, f"{service}'s parent {parent} missing from DAG"


class TestTopologicalSort:
    """Tests for DAG-aware topological sorting."""

    def test_respects_parent_child_order(self):
        """Children must come after their parents."""
        result = _topological_sort(["ontology", "extraction", "document_intelligence"])
        assert result.index("document_intelligence") < result.index("extraction")
        assert result.index("extraction") < result.index("ontology")

    def test_siblings_grouped_after_parent(self):
        """Siblings all appear after their shared parent."""
        services = ["compression", "redaction", "extraction", "document_intelligence"]
        result = _topological_sort(services)
        di_idx = result.index("document_intelligence")
        for svc in ["extraction", "redaction", "compression"]:
            assert result.index(svc) > di_idx

    def test_missing_parent_tolerated(self):
        """Services whose parent is absent don't crash."""
        result = _topological_sort(["extraction", "ontology"])
        assert "extraction" in result
        assert "ontology" in result
        assert result.index("extraction") < result.index("ontology")

    def test_unknown_services_appended(self):
        """Services not in DAG appear at the end."""
        result = _topological_sort(["document_intelligence", "custom_service"])
        assert result[-1] == "custom_service"


class TestStepAliases:
    """Tests for step alias resolution."""

    def test_canonical_names_resolve_to_self(self):
        """Canonical names should resolve to themselves."""
        for canonical in [
            "document_intelligence",
            "extraction",
            "ontology",
            "redaction",
            "compression",
            "embedding",
            "colbert",
            "colpali",
        ]:
            assert resolve_step_name(canonical) == canonical

    def test_friendly_aliases(self):
        """Friendly aliases should resolve correctly."""
        assert resolve_step_name("ocr") == "document_intelligence"
        assert resolve_step_name("knowledge_graph") == "ontology"
        assert resolve_step_name("redact") == "redaction"
        assert resolve_step_name("extract") == "extraction"
        assert resolve_step_name("compress") == "compression"

    def test_chunking_alias_raises(self):
        """Chunking aliases should raise NotImplementedError."""
        for alias in ["chunking", "chunk", "split", "text_chunking"]:
            with pytest.raises(NotImplementedError, match="not available as a pipeline step"):
                resolve_step_name(alias)

    def test_placeholder_steps_raise(self):
        """Placeholder steps should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="coming soon"):
            resolve_step_name("enrichment")
        with pytest.raises(NotImplementedError, match="coming soon"):
            resolve_step_name("graph_ontology_builder")

    def test_unknown_step_raises(self):
        """Unknown step names should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown pipeline step"):
            resolve_step_name("nonexistent_service")

    def test_case_insensitive(self):
        """Alias resolution should be case-insensitive."""
        assert resolve_step_name("OCR") == "document_intelligence"
        assert resolve_step_name("Knowledge_Graph") == "ontology"


class TestParseStepsConfig:
    """Tests for parse_steps_config with DAG-aware sorting."""

    def test_sorts_by_dag(self):
        """Steps should be topologically sorted by DAG."""
        configs = parse_steps_config(
            {
                "ontology": {"resolve_entities": True},
                "extraction": {},
                "document_intelligence": {"mode": "performance"},
            }
        )
        names = [c.service for c in configs]
        assert names.index("document_intelligence") < names.index("extraction")
        assert names.index("extraction") < names.index("ontology")

    def test_aliases_resolved(self):
        """Aliases should be resolved in output."""
        configs = parse_steps_config({"ocr": {}, "knowledge_graph": {}})
        names = [c.service for c in configs]
        assert "document_intelligence" in names
        assert "ontology" in names

    def test_siblings_both_present(self):
        """Sibling services both appear."""
        configs = parse_steps_config(
            {
                "document_intelligence": {},
                "redaction": {"mode": "strict"},
                "extraction": {},
            }
        )
        names = [c.service for c in configs]
        assert "redaction" in names
        assert "extraction" in names


# =============================================================================
# VALIDATOR TESTS
# =============================================================================


class TestInputTypeDetection:
    """Tests for input type detection."""

    def test_detect_file_input(self):
        """Should detect file input type."""
        input_data = PipelineInput(files=[FileInput(base64="abc123")])
        assert _detect_input_type(input_data) == "file"

    def test_detect_text_input(self):
        """Should detect text input type."""
        input_data = PipelineInput(text="Hello world")
        assert _detect_input_type(input_data) == "text"

    def test_detect_entities_input(self):
        """Should detect entities input type."""
        from latence._models.common import Entity

        input_data = PipelineInput(
            entities=[Entity(start=0, end=5, text="hello", label="test", score=0.9)]
        )
        assert _detect_input_type(input_data) == "entities"

    def test_detect_unknown_input(self):
        """Should return unknown for empty input."""
        input_data = PipelineInput()
        assert _detect_input_type(input_data) == "unknown"

    def test_detect_none_input(self):
        """Should return unknown for None input."""
        assert _detect_input_type(None) == "unknown"


class TestPipelineValidation:
    """Tests for pipeline validation."""

    def test_empty_pipeline_fails(self):
        """Empty pipeline should fail validation."""
        config = PipelineConfig(services=[])
        result = validate_pipeline(config)
        assert not result.valid
        assert "at least one service" in result.errors[0].lower()

    def test_simple_text_pipeline_valid(self):
        """Simple text pipeline should be valid."""
        config = PipelineConfig(
            services=[
                ServiceConfig(service="extraction"),
                ServiceConfig(service="ontology"),
            ]
        )
        input_data = PipelineInput(text="Hello world")
        result = validate_pipeline(config, input_data)
        assert result.valid
        assert len(result.errors) == 0

    def test_file_input_auto_injects_doc_intel(self):
        """File input should auto-inject document_intelligence."""
        config = PipelineConfig(
            services=[ServiceConfig(service="extraction")],
            strict_mode=False,
        )
        input_data = PipelineInput(files=[FileInput(base64="abc123")])
        result = validate_pipeline(config, input_data)
        assert result.valid
        assert "document_intelligence" in result.services
        assert "document_intelligence" in result.auto_injected

    def test_file_input_strict_mode_fails_without_doc_intel(self):
        """File input in strict mode should fail without document_intelligence."""
        config = PipelineConfig(
            services=[ServiceConfig(service="extraction")],
            strict_mode=True,
        )
        input_data = PipelineInput(files=[FileInput(base64="abc123")])
        with pytest.raises(PipelineValidationError) as exc_info:
            validate_pipeline(config, input_data)
        error_msg = str(exc_info.value.errors[0]).lower()
        assert "file" in error_msg or "text" in error_msg

    def test_ontology_auto_injects_extraction_via_dag(self):
        """Ontology should auto-inject extraction (its DAG parent)."""
        config = PipelineConfig(
            services=[
                ServiceConfig(service="document_intelligence"),
                ServiceConfig(service="ontology"),
            ],
            strict_mode=False,
        )
        input_data = PipelineInput(files=[FileInput(base64="abc123")])
        result = validate_pipeline(config, input_data)
        assert result.valid
        assert "extraction" in result.services
        assert "extraction" in result.auto_injected

    def test_ontology_strict_mode_fails_without_extraction(self):
        """Ontology in strict mode should fail without extraction."""
        config = PipelineConfig(
            services=[
                ServiceConfig(service="document_intelligence"),
                ServiceConfig(service="ontology"),
            ],
            strict_mode=True,
        )
        input_data = PipelineInput(files=[FileInput(base64="abc123")])
        with pytest.raises(PipelineValidationError) as exc_info:
            validate_pipeline(config, input_data)
        assert "extraction" in str(exc_info.value.errors[0]).lower()

    def test_colpali_with_text_only_warns(self):
        """ColPali with text-only input should warn."""
        config = PipelineConfig(
            services=[ServiceConfig(service="colpali")],
        )
        input_data = PipelineInput(text="Hello world")
        result = validate_pipeline(config, input_data)
        assert result.valid
        assert len(result.warnings) > 0

    def test_correct_ordering_validation(self):
        """Services should be validated for correct ordering."""
        config = PipelineConfig(
            services=[
                ServiceConfig(service="document_intelligence"),
                ServiceConfig(service="extraction"),
                ServiceConfig(service="ontology"),
            ],
        )
        input_data = PipelineInput(files=[FileInput(base64="abc123")])
        result = validate_pipeline(config, input_data)
        assert result.valid
        assert len(result.auto_injected) == 0

    def test_dag_parent_auto_injection_cascades(self):
        """Auto-injection should cascade through the DAG."""
        config = PipelineConfig(
            services=[ServiceConfig(service="ontology")],
            strict_mode=False,
        )
        input_data = PipelineInput(files=[FileInput(base64="abc123")])
        result = validate_pipeline(config, input_data)
        assert result.valid
        assert "document_intelligence" in result.auto_injected
        assert "extraction" in result.auto_injected


# =============================================================================
# BUILDER TESTS
# =============================================================================


class TestPipelineBuilder:
    """Tests for the PipelineBuilder class."""

    def test_empty_builder_uses_smart_defaults(self):
        """Empty builder with file input should use default intelligence pipeline."""
        builder = PipelineBuilder()
        config = builder.build()
        services = [s.service for s in config.services]
        assert "document_intelligence" in services
        assert "extraction" in services
        assert "ontology" in services

    def test_add_generic_service(self):
        """Should be able to add generic service."""
        builder = PipelineBuilder()
        builder.add("extraction", threshold=0.5)
        config = builder.build()
        ext = [s for s in config.services if s.service == "extraction"]
        assert len(ext) == 1
        assert ext[0].config["threshold"] == 0.5

    def test_fluent_api(self):
        """Builder should support fluent API."""
        config = (
            PipelineBuilder()
            .doc_intel(mode="performance")
            .extraction(threshold=0.3)
            .ontology(resolve_entities=True)
            .build()
        )
        names = [s.service for s in config.services]
        assert "document_intelligence" in names
        assert "extraction" in names
        assert "ontology" in names

    def test_doc_intel_pipeline_options_hardcoded(self):
        """Doc intel should hardcode pipeline_options for layout/chart/seal."""
        config = (
            PipelineBuilder()
            .doc_intel(mode="performance", output_format="json", max_pages=10)
            .build()
        )
        di_config = config.services[0].config
        assert di_config["mode"] == "performance"
        assert di_config["output_format"] == "json"
        assert di_config["max_pages"] == 10
        po = di_config["pipeline_options"]
        assert po["use_layout_detection"] is True
        assert po["use_chart_recognition"] is False
        assert po["use_seal_recognition"] is False

    def test_extraction_config(self):
        """Extraction should accept proper configuration."""
        config = (
            PipelineBuilder()
            .extraction(
                threshold=0.5,
                user_labels=["person", "organization"],
                enable_refinement=True,
            )
            .build()
        )
        ext = [s for s in config.services if s.service == "extraction"][0]
        assert ext.config["threshold"] == 0.5
        assert ext.config["user_labels"] == ["person", "organization"]
        assert ext.config["enable_refinement"] is True

    def test_ontology_config(self):
        """Ontology should accept proper configuration."""
        config = (
            PipelineBuilder()
            .ontology(
                resolve_entities=True,
                predict_missing_relations=True,
                kg_output_format="rdf",
            )
            .build()
        )
        ont = [s for s in config.services if s.service == "ontology"][0]
        assert ont.config["resolve_entities"] is True
        assert ont.config["predict_missing_relations"] is True
        assert ont.config["kg_output_format"] == "rdf"

    def test_redaction_always_enforces_refinement(self):
        """Redaction builder must always set enforce_refinement=True."""
        config = PipelineBuilder().redaction(mode="balanced").build()
        red = [s for s in config.services if s.service == "redaction"][0]
        assert red.config["enforce_refinement"] is True

    def test_compression_config(self):
        """Compression should accept proper configuration."""
        config = (
            PipelineBuilder().compression(compression_rate=0.7, force_tokens=["important"]).build()
        )
        comp = [s for s in config.services if s.service == "compression"][0]
        assert comp.config["compression_rate"] == 0.7
        assert comp.config["force_tokens"] == ["important"]

    def test_embedding_config(self):
        """Embedding should accept proper configuration."""
        config = PipelineBuilder().embedding(dimension=768).build()
        emb = [s for s in config.services if s.service == "embedding"][0]
        assert emb.config["dimension"] == 768

    def test_colbert_config(self):
        """ColBERT should accept proper configuration."""
        config = PipelineBuilder().colbert(query_expansion=True).build()
        cb = [s for s in config.services if s.service == "colbert"][0]
        assert cb.config["query_expansion"] is True

    def test_chunking_raises_not_implemented(self):
        """Chunking should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not available as a pipeline step"):
            PipelineBuilder().chunking()

    def test_enrichment_raises_not_implemented(self):
        """Enrichment should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="coming soon"):
            PipelineBuilder().enrichment()

    def test_graph_ontology_builder_raises_not_implemented(self):
        """Graph ontology builder should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="coming soon"):
            PipelineBuilder().graph_ontology_builder()

    def test_store_intermediate_default_true(self):
        """store_intermediate should default to True."""
        config = PipelineBuilder().doc_intel().extraction().build()
        assert config.store_intermediate is True

    def test_store_intermediate_can_disable(self):
        """store_intermediate(False) should disable it."""
        config = PipelineBuilder().doc_intel().extraction().store_intermediate(False).build()
        assert config.store_intermediate is False

    def test_strict_mode(self):
        """Should set strict_mode flag."""
        config = PipelineBuilder().doc_intel().extraction().strict().build()
        assert config.strict_mode is True

    def test_repr(self):
        """Builder should have useful repr."""
        builder = PipelineBuilder().extraction().ontology()
        repr_str = repr(builder)
        assert "extraction" in repr_str
        assert "ontology" in repr_str


# =============================================================================
# SERVICE IO TESTS
# =============================================================================


class TestServiceIO:
    """Tests for SERVICE_IO definitions."""

    def test_all_dag_services_defined(self):
        """All services in DAG should be defined in SERVICE_IO."""
        for service in SERVICE_PARENT:
            assert service in SERVICE_IO, f"{service} in DAG but not in SERVICE_IO"

    def test_services_have_input_output(self):
        """All services should have input and output types."""
        for service, io in SERVICE_IO.items():
            assert "input" in io, f"{service} missing input"
            assert "output" in io, f"{service} missing output"
            assert isinstance(io["input"], list), f"{service} input should be list"
            assert isinstance(io["output"], str), f"{service} output should be str"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPipelineBuilderValidation:
    """Integration tests for builder + validation."""

    def test_full_document_analysis_pipeline(self):
        """Complete document analysis pipeline should be valid."""
        config = (
            PipelineBuilder()
            .doc_intel(mode="performance")
            .extraction(threshold=0.3, user_labels=["person", "organization", "location"])
            .ontology(resolve_entities=True, predict_missing_relations=True)
            .compression(compression_rate=0.5)
            .colbert(query_expansion=True)
            .store_intermediate()
            .build()
        )

        input_data = PipelineInput(files=[FileInput(url="https://example.com/doc.pdf")])
        result = validate_pipeline(config, input_data)

        assert result.valid
        assert len(result.auto_injected) == 0
        assert len(result.services) == 5

    def test_auto_fix_incomplete_pipeline(self):
        """Incomplete pipeline should be auto-fixed via DAG by build()."""
        config = PipelineBuilder().ontology().build()

        services = [s.service for s in config.services]
        assert "extraction" in services
        assert "document_intelligence" in services
        assert services.index("extraction") < services.index("ontology")


# =============================================================================
# RESUME / CHECKPOINT MODEL TESTS
# =============================================================================


class TestPipelineResumeModels:
    """Tests for the new checkpoint/resume pipeline models."""

    def test_stage_download_model(self):
        """StageDownload model parses correctly."""
        from latence._models.pipeline import StageDownload

        data = {
            "service": "document_intelligence",
            "stage_index": 0,
            "status": "completed",
            "download_url": "https://b2.example.com/signed",
            "files_total": 10,
            "files_ok": 10,
            "files_failed": 0,
            "cost_usd": 0.50,
            "duration_s": 45.2,
            "from_checkpoint": False,
        }
        sd = StageDownload.model_validate(data)
        assert sd.service == "document_intelligence"
        assert sd.download_url == "https://b2.example.com/signed"
        assert sd.files_ok == 10
        assert sd.from_checkpoint is False

    def test_pipeline_report_model(self):
        """PipelineReport model parses correctly."""
        from latence._models.pipeline import PipelineReport

        data = {
            "stages": [
                {
                    "service": "document_intelligence",
                    "files_total": 10,
                    "files_ok": 10,
                    "files_failed": 0,
                    "cost_usd": 0.50,
                    "duration_s": 45.2,
                    "from_checkpoint": False,
                },
            ],
            "dataset": {
                "total_files": 10,
                "files_ok": 10,
                "files_failed": 0,
                "total_pages": 87,
                "avg_pages_per_file": 8.7,
            },
            "cost": {"total_usd": 0.50, "per_stage": {"document_intelligence": 0.50}},
            "timing": {"total_duration_s": 45.2, "per_stage": {"document_intelligence": 45.2}},
            "resume_count": 0,
        }
        report = PipelineReport.model_validate(data)
        assert len(report.stages) == 1
        assert report.dataset["total_pages"] == 87
        assert report.cost["total_usd"] == 0.50
        assert report.resume_count == 0

    def test_pipeline_status_response_with_resume_fields(self):
        """PipelineStatusResponse includes new resume/report fields."""
        from latence._models.pipeline import PipelineStatusResponse

        data = {
            "job_id": "pipe_abc123",
            "status": "RESUMABLE",
            "stages_completed": 1,
            "total_stages": 3,
            "is_resumable": True,
            "resume_count": 1,
            "failed_stage": "extraction",
            "pipeline_report": {
                "stages": [],
                "dataset": {},
                "cost": {},
                "timing": {},
                "resume_count": 1,
            },
        }
        resp = PipelineStatusResponse.model_validate(data)
        assert resp.status == "RESUMABLE"
        assert resp.is_resumable is True
        assert resp.resume_count == 1
        assert resp.failed_stage == "extraction"
        assert resp.pipeline_report is not None

    def test_job_error_is_resumable(self):
        """JobError includes is_resumable attribute."""
        from latence._exceptions import JobError

        err = JobError(
            "Pipeline failed at extraction",
            job_id="pipe_abc",
            error_code="ALL_FILES_FAILED",
            is_resumable=True,
        )
        assert err.is_resumable is True
        assert err.job_id == "pipe_abc"

        err2 = JobError("Not resumable", job_id="pipe_xyz")
        assert err2.is_resumable is False


# =============================================================================
# DATA PACKAGE TESTS
# =============================================================================


class TestDataPackage:
    """Tests for DataPackage composition."""

    def test_quality_report_fields(self):
        """QualityReport should have the documented fields."""
        from latence._pipeline.data_package import ConfidenceScores, QualityReport, StageReport

        report = QualityReport(
            stages=[
                StageReport(
                    service="document_intelligence", status="completed", processing_time_ms=1500.0
                ),
                StageReport(service="extraction", status="completed", processing_time_ms=800.0),
            ],
            confidence=ConfidenceScores(entity_avg_confidence=0.87),
            total_processing_time_ms=2300.0,
            total_cost_usd=0.05,
        )
        assert report.total_cost_usd == 0.05
        assert report.total_processing_time_ms == 2300.0
        assert len(report.stages) == 2
        assert report.confidence.entity_avg_confidence == 0.87
        assert not hasattr(report, "pipeline_name")
        assert not hasattr(report, "services_run")

    def test_chunking_summary_uses_chunk_size(self):
        """ChunkingSummary should use chunk_size, not avg_chunk_size."""
        from latence._pipeline.data_package import ChunkingSummary

        summary = ChunkingSummary(num_chunks=10, strategy="hybrid", chunk_size=512)
        assert summary.chunk_size == 512
        assert not hasattr(summary, "avg_chunk_size")

    def test_enrichment_features_is_dict(self):
        """EnrichmentSection.features should be a dict, not a list."""
        from latence._pipeline.data_package import EnrichmentSection

        section = EnrichmentSection(features={"quality": {"score": 0.9}, "density": {"score": 0.7}})
        assert isinstance(section.features, dict)
        assert "quality" in section.features

    def test_data_package_merge_structure(self):
        """merge() output should match documented structure."""
        from latence._pipeline.data_package import (
            DataPackage,
            DocumentMetadataInfo,
            DocumentSection,
        )

        pkg = DataPackage(
            id="pipe_test",
            name="Test",
            created_at="2025-01-01T00:00:00Z",
            status="COMPLETED",
            document=DocumentSection(
                markdown="# Hello",
                pages=["# Hello"],
                metadata=DocumentMetadataInfo(filename="test.pdf", pages_processed=1),
            ),
        )
        merged = pkg.merge()
        assert "id" in merged
        assert "name" in merged
        assert "status" in merged
        assert "created_at" in merged
        assert "documents" in merged
        assert "summary" in merged
        assert "stats" not in merged
        assert "meta" not in merged

    def test_download_archive_structure(self, tmp_path):
        """Archive should contain quality_report.json and metadata.json."""
        import zipfile

        from latence._pipeline.data_package import (
            DataPackage,
            DocumentMetadataInfo,
            DocumentSection,
        )

        pkg = DataPackage(
            id="pipe_test",
            name="Test",
            created_at="2025-01-01T00:00:00Z",
            status="COMPLETED",
            document=DocumentSection(
                markdown="# Hello",
                metadata=DocumentMetadataInfo(filename="test.pdf"),
            ),
        )
        archive_path = tmp_path / "test.zip"
        pkg.download_archive(archive_path)

        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()
            assert any("quality_report.json" in n for n in names)
            assert any("metadata.json" in n for n in names)
            assert not any("summary.json" in n for n in names)


# =============================================================================
# ERROR HIERARCHY TESTS
# =============================================================================


class TestErrorHierarchy:
    """Tests for the exception hierarchy."""

    def test_all_errors_inherit_from_latence_error(self):
        from latence import (
            APIConnectionError,
            APIError,
            APITimeoutError,
            AuthenticationError,
            InsufficientCreditsError,
            JobError,
            JobTimeoutError,
            LatenceError,
            NotFoundError,
            RateLimitError,
            ServerError,
            ValidationError,
        )

        for cls in [
            APIError,
            AuthenticationError,
            InsufficientCreditsError,
            NotFoundError,
            ValidationError,
            RateLimitError,
            ServerError,
            APIConnectionError,
            APITimeoutError,
            JobError,
            JobTimeoutError,
        ]:
            assert issubclass(cls, LatenceError)

    def test_job_timeout_is_job_error(self):
        from latence import JobError, JobTimeoutError

        assert issubclass(JobTimeoutError, JobError)

    def test_service_config_public_import(self):
        """ServiceConfig should be importable from latence directly."""
        from latence import ServiceConfig

        sc = ServiceConfig(service="extraction", config={"threshold": 0.3})
        assert sc.service == "extraction"
