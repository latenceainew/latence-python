"""Tests for DataPackage composition."""

from __future__ import annotations

import zipfile
from pathlib import Path

from latence._models.pipeline import (
    PipelineExecutionSummary,
    PipelineResultResponse,
    StageResult,
)
from latence._pipeline.data_package import DataPackage


def _make_execution_summary(
    total_stages: int = 5,
    completed_stages: int = 5,
    total_processing_time_ms: float = 5000.0,
    total_credits_used: float = 1.5,
) -> PipelineExecutionSummary:
    return PipelineExecutionSummary(
        total_stages=total_stages,
        completed_stages=completed_stages,
        total_processing_time_ms=total_processing_time_ms,
        total_credits_used=total_credits_used,
    )


# -----------------------------------------------------------------------------
# test_from_pipeline_result_with_all_stages
# -----------------------------------------------------------------------------


def test_from_pipeline_result_with_all_stages() -> None:
    """Create a realistic PipelineResultResponse with all stages and verify all sections."""
    result = PipelineResultResponse(
        job_id="job-123",
        status="COMPLETED",
        created_at="2025-01-15T12:00:00Z",
        execution_summary=_make_execution_summary(),
        intermediate_results={
            "document_intelligence": StageResult(
                service="document_intelligence",
                status="completed",
                output={
                    "content": "# Hello World\n\nTest document content.",
                    "pages_processed": 2,
                },
                credits_used=0.5,
                processing_time_ms=1200,
            ),
            "extraction": StageResult(
                service="extraction",
                status="completed",
                output={
                    "entities": [
                        {
                            "text": "Apple",
                            "label": "ORG",
                            "score": 0.95,
                            "start": 0,
                            "end": 5,
                        }
                    ]
                },
                credits_used=0.3,
                processing_time_ms=800,
            ),
            "ontology": StageResult(
                service="ontology",
                status="completed",
                output={
                    "entities": [
                        {"text": "Apple", "label": "ORG", "start": 0, "end": 5, "score": 0.95},
                        {"text": "Cupertino", "label": "LOC", "start": 10, "end": 18, "score": 0.9},
                    ],
                    "relations": [
                        {
                            "entity1": {"text": "Apple", "label": "ORG", "start": 0, "end": 5},
                            "entity2": {
                                "text": "Cupertino",
                                "label": "LOC",
                                "start": 10,
                                "end": 18,
                            },
                            "relation_type": "headquartered_in",
                            "relation_label": "headquartered_in",
                            "score": 0.88,
                        }
                    ],
                    "knowledge_graph": {"nodes": [], "edges": []},
                },
                credits_used=0.4,
                processing_time_ms=1500,
            ),
            "redaction": StageResult(
                service="redaction",
                status="completed",
                output={
                    "redacted_text": "*** is in ***",
                    "entities": [
                        {"text": "Apple", "label": "ORG", "score": 0.95, "start": 0, "end": 5},
                        {"text": "Cupertino", "label": "LOC", "score": 0.9, "start": 10, "end": 18},
                    ],
                },
                credits_used=0.2,
                processing_time_ms=400,
            ),
            "compression": StageResult(
                service="compression",
                status="completed",
                output={
                    "compressed_text": "Hello World",
                    "original_tokens": 100,
                    "compressed_tokens": 50,
                },
                credits_used=0.1,
                processing_time_ms=300,
            ),
        },
    )

    pkg = DataPackage.from_pipeline_result(result, name="test-pipeline", services=None)

    assert pkg.id == "job-123"
    assert pkg.name == "test-pipeline"
    assert pkg.status == "COMPLETED"
    assert pkg.created_at == "2025-01-15T12:00:00Z"

    # Document section
    assert pkg.document is not None
    assert pkg.document.markdown == "# Hello World\n\nTest document content."
    assert pkg.document.metadata.pages_processed == 2

    # Entities section
    assert pkg.entities is not None
    assert len(pkg.entities.items) == 1
    assert pkg.entities.items[0].text == "Apple"
    assert pkg.entities.items[0].label == "ORG"
    assert pkg.entities.summary.total == 1

    # Knowledge graph section
    assert pkg.knowledge_graph is not None
    assert len(pkg.knowledge_graph.entities) == 2
    assert len(pkg.knowledge_graph.relations) == 1
    assert pkg.knowledge_graph.relations[0].relation_type == "headquartered_in"
    assert pkg.knowledge_graph.summary.total_entities == 2
    assert pkg.knowledge_graph.summary.total_relations == 1

    # Redaction section
    assert pkg.redaction is not None
    assert pkg.redaction.redacted_text == "*** is in ***"
    assert len(pkg.redaction.pii_detected) == 2

    # Compression section
    assert pkg.compression is not None
    assert pkg.compression.compressed_text == "Hello World"
    assert pkg.compression.summary.original_tokens == 100
    assert pkg.compression.summary.compressed_tokens == 50

    # Quality report
    assert pkg.quality is not None
    assert len(pkg.quality.stages) == 5


# -----------------------------------------------------------------------------
# test_from_pipeline_result_minimal
# -----------------------------------------------------------------------------


def test_from_pipeline_result_minimal() -> None:
    """Test with only final_output and no intermediate_results."""
    result = PipelineResultResponse(
        job_id="job-minimal",
        status="COMPLETED",
        execution_summary=_make_execution_summary(total_stages=1, completed_stages=1),
        final_output={"some": "data"},
        intermediate_results=None,
    )

    pkg = DataPackage.from_pipeline_result(result)

    assert pkg.id == "job-minimal"
    assert pkg.status == "COMPLETED"
    assert pkg.document is None
    assert pkg.entities is None
    assert pkg.knowledge_graph is None
    assert pkg.redaction is None
    assert pkg.compression is None
    assert pkg.quality is not None


# -----------------------------------------------------------------------------
# test_merge_output_shape
# -----------------------------------------------------------------------------


def test_merge_output_shape() -> None:
    """Call merge() and verify the dict structure has required keys."""
    result = PipelineResultResponse(
        job_id="job-merge",
        status="COMPLETED",
        execution_summary=_make_execution_summary(),
        intermediate_results={
            "document_intelligence": StageResult(
                service="document_intelligence",
                status="completed",
                output={"content": "Hello", "pages_processed": 1},
            ),
        },
    )
    pkg = DataPackage.from_pipeline_result(result, name="merge-test")

    merged = pkg.merge()

    assert "id" in merged
    assert "name" in merged
    assert "status" in merged
    assert "created_at" in merged
    assert "documents" in merged
    assert "summary" in merged
    assert merged["id"] == "job-merge"
    assert merged["name"] == "merge-test"
    assert isinstance(merged["documents"], list)
    assert isinstance(merged["summary"], dict)


# -----------------------------------------------------------------------------
# test_download_archive_produces_valid_zip
# -----------------------------------------------------------------------------


def test_download_archive_produces_valid_zip(tmp_path: Path) -> None:
    """Call download_archive and verify the resulting file is a valid ZIP."""
    result = PipelineResultResponse(
        job_id="job-archive",
        status="COMPLETED",
        execution_summary=_make_execution_summary(),
        intermediate_results={
            "document_intelligence": StageResult(
                service="document_intelligence",
                status="completed",
                output={"content": "# Doc", "pages_processed": 1},
            ),
            "extraction": StageResult(
                service="extraction",
                status="completed",
                output={
                    "entities": [{"text": "X", "label": "ORG", "score": 0.9, "start": 0, "end": 1}]
                },
            ),
        },
    )
    pkg = DataPackage.from_pipeline_result(result, name="archive-test")

    archive_path = tmp_path / "test.zip"
    out = pkg.download_archive(archive_path)

    assert out == archive_path
    assert archive_path.exists()

    with zipfile.ZipFile(archive_path, "r") as zf:
        names = zf.namelist()
        assert len(names) > 0
        # Must have README, metadata, quality_report
        assert any("README.md" in n for n in names)
        assert any("metadata.json" in n for n in names)
        assert any("quality_report.json" in n for n in names)
        # With document and entities we expect those too
        assert any("document.md" in n for n in names)
        assert any("entities.json" in n for n in names)


# -----------------------------------------------------------------------------
# test_parse_warnings_for_malformed_relations
# -----------------------------------------------------------------------------


def test_parse_warnings_for_malformed_relations() -> None:
    """Ontology output with malformed relation produces parse_warnings."""
    result = PipelineResultResponse(
        job_id="job-malformed",
        status="COMPLETED",
        execution_summary=_make_execution_summary(),
        intermediate_results={
            "ontology": StageResult(
                service="ontology",
                status="completed",
                output={
                    "entities": [],
                    "relations": [
                        # Valid relation
                        {
                            "entity1": {"text": "A", "label": "ORG", "start": 0, "end": 1},
                            "entity2": {"text": "B", "label": "LOC", "start": 2, "end": 3},
                            "relation_label": "related",
                            "score": 0.9,
                        },
                        # Malformed: missing required entity1
                        {
                            "entity2": {"text": "B", "label": "LOC", "start": 2, "end": 3},
                            "score": 0.8,
                        },
                    ],
                },
            ),
        },
    )

    pkg = DataPackage.from_pipeline_result(result)

    assert len(pkg.parse_warnings) >= 1
    assert any("malformed" in w.lower() or "skipped" in w.lower() for w in pkg.parse_warnings)


# -----------------------------------------------------------------------------
# test_created_at_uses_server_timestamp
# -----------------------------------------------------------------------------


def test_created_at_uses_server_timestamp() -> None:
    """DataPackage uses created_at from PipelineResultResponse, not datetime.now()."""
    result = PipelineResultResponse(
        job_id="job-ts",
        status="COMPLETED",
        created_at="2025-01-01T00:00:00Z",
        execution_summary=_make_execution_summary(),
        intermediate_results=None,
    )

    pkg = DataPackage.from_pipeline_result(result)

    assert pkg.created_at == "2025-01-01T00:00:00Z"
