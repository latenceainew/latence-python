"""Tests for response model parsing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from latence._models.common import Entity, Relation
from latence._models.jobs import CreditsResponse, JobCancelResponse
from latence._models.pipeline import (
    FileInput,
    PipelineConfig,
    PipelineResultResponse,
    PipelineStatusResponse,
    ServiceConfig,
)

# ---------------------------------------------------------------------------
# PipelineResultResponse
# ---------------------------------------------------------------------------


class TestPipelineResultResponse:
    """Tests for PipelineResultResponse model."""

    def test_parse_minimal_valid_fixture(self) -> None:
        """Parse minimal valid API response with required fields only."""
        fixture = {
            "job_id": "job_abc123",
            "status": "COMPLETED",
            "execution_summary": {
                "total_stages": 2,
                "completed_stages": 2,
            },
        }
        result = PipelineResultResponse.model_validate(fixture)
        assert result.job_id == "job_abc123"
        assert result.status == "COMPLETED"
        assert result.execution_summary.total_stages == 2
        assert result.execution_summary.completed_stages == 2
        assert result.final_output is None
        assert result.intermediate_results is None

    def test_parse_full_realistic_fixture(self) -> None:
        """Parse realistic API response including all optional fields."""
        fixture = {
            "success": True,
            "request_id": "req_xyz789",
            "job_id": "job_abc123",
            "status": "COMPLETED",
            "created_at": "2025-03-01T10:00:00Z",
            "final_output": {"documents": [], "entities": []},
            "intermediate_results": {
                "document_intelligence": {
                    "service": "document_intelligence",
                    "status": "completed",
                    "output": {"page_count": 5},
                    "credits_used": 0.1,
                    "processing_time_ms": 150.0,
                },
            },
            "execution_summary": {
                "total_stages": 2,
                "completed_stages": 2,
                "failed_stage": None,
                "total_credits_used": 0.5,
                "total_processing_time_ms": 300.0,
            },
            "output_url": "https://storage.example.com/results.jsonl",
            "download_url": "https://storage.example.com/archive.zip",
            "b2_prefix": "jobs/job_abc123",
            "output_expires_at": "2025-03-02T10:00:00Z",
            "usage": {"credits": 0.5, "total_credits": 100.0},
        }
        result = PipelineResultResponse.model_validate(fixture)
        assert result.job_id == "job_abc123"
        assert result.status == "COMPLETED"
        assert result.created_at == "2025-03-01T10:00:00Z"
        assert result.final_output == {"documents": [], "entities": []}
        assert result.intermediate_results is not None
        assert "document_intelligence" in result.intermediate_results
        stage = result.intermediate_results["document_intelligence"]
        assert stage.service == "document_intelligence"
        assert stage.status == "completed"
        assert stage.output == {"page_count": 5}
        assert stage.credits_used == 0.1
        assert stage.processing_time_ms == 150.0
        assert result.execution_summary.total_credits_used == 0.5
        assert result.execution_summary.total_processing_time_ms == 300.0
        assert result.output_url == "https://storage.example.com/results.jsonl"
        assert result.download_url == "https://storage.example.com/archive.zip"
        assert result.b2_prefix == "jobs/job_abc123"
        assert result.output_expires_at == "2025-03-02T10:00:00Z"
        assert result.usage is not None
        assert result.usage.credits == 0.5
        assert result.usage.total_credits == 100.0
        assert result.success is True
        assert result.request_id == "req_xyz789"


# ---------------------------------------------------------------------------
# PipelineStatusResponse
# ---------------------------------------------------------------------------


class TestPipelineStatusResponse:
    """Tests for PipelineStatusResponse model."""

    @pytest.mark.parametrize(
        "status",
        [
            "QUEUED",
            "IN_PROGRESS",
            "COMPLETED",
            "CACHED",
            "PULLED",
            "FAILED",
            "CANCELLED",
            "RESUMABLE",
        ],
    )
    def test_all_status_values_accepted(self, status: str) -> None:
        """All literal status values are accepted."""
        fixture = {"job_id": "job_123", "status": status}
        result = PipelineStatusResponse.model_validate(fixture)
        assert result.status == status
        assert result.job_id == "job_123"

    def test_parse_full_status_response(self) -> None:
        """Parse status response with optional fields."""
        fixture = {
            "job_id": "job_456",
            "status": "IN_PROGRESS",
            "current_stage": 1,
            "current_service": "extraction",
            "stages_completed": 1,
            "total_stages": 3,
            "created_at": "2025-03-01T10:00:00Z",
            "is_resumable": False,
            "resume_count": 0,
        }
        result = PipelineStatusResponse.model_validate(fixture)
        assert result.status == "IN_PROGRESS"
        assert result.current_stage == 1
        assert result.current_service == "extraction"
        assert result.stages_completed == 1
        assert result.total_stages == 3
        assert result.created_at == "2025-03-01T10:00:00Z"

    def test_invalid_status_raises_validation_error(self) -> None:
        """Invalid status value raises ValidationError."""
        fixture = {"job_id": "job_789", "status": "INVALID_STATUS"}
        with pytest.raises(ValidationError):
            PipelineStatusResponse.model_validate(fixture)


# ---------------------------------------------------------------------------
# FileInput
# ---------------------------------------------------------------------------


class TestFileInput:
    """Tests for FileInput model with base64/url validator."""

    def test_base64_only_works(self) -> None:
        """FileInput with base64 only should parse successfully."""
        result = FileInput(base64="abc")
        assert result.base64 == "abc"
        assert result.url is None

    def test_url_only_works(self) -> None:
        """FileInput with url only should parse successfully."""
        result = FileInput(url="https://example.com/file.pdf")
        assert result.url == "https://example.com/file.pdf"
        assert result.base64 is None

    def test_base64_and_url_works(self) -> None:
        """FileInput with both base64 and url should parse successfully."""
        result = FileInput(base64="abc", url="https://example.com/file.pdf")
        assert result.base64 == "abc"
        assert result.url == "https://example.com/file.pdf"

    def test_neither_base64_nor_url_raises_validation_error(self) -> None:
        """FileInput with neither base64 nor url should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            FileInput()
        assert "base64" in str(exc_info.value).lower() or "url" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# PipelineConfig and ServiceConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    """Tests for PipelineConfig model."""

    def test_parse_minimal_config(self) -> None:
        """Parse minimal pipeline config."""
        fixture = {
            "services": [
                {"service": "document_intelligence", "config": {}},
            ],
        }
        result = PipelineConfig.model_validate(fixture)
        assert len(result.services) == 1
        assert result.services[0].service == "document_intelligence"
        assert result.store_intermediate is False
        assert result.strict_mode is False

    def test_parse_full_config(self) -> None:
        """Parse config with all options."""
        fixture = {
            "services": [
                {"service": "extraction", "config": {"model": "default"}},
            ],
            "store_intermediate": True,
            "strict_mode": True,
            "name": "My Pipeline",
        }
        result = PipelineConfig.model_validate(fixture)
        assert result.store_intermediate is True
        assert result.strict_mode is True
        assert result.name == "My Pipeline"


class TestServiceConfig:
    """Tests for ServiceConfig model."""

    def test_parse_service_config(self) -> None:
        """Parse service config with service name and config dict."""
        fixture = {"service": "document_intelligence", "config": {"page_limit": 10}}
        result = ServiceConfig.model_validate(fixture)
        assert result.service == "document_intelligence"
        assert result.config == {"page_limit": 10}

    def test_parse_all_valid_service_names(self) -> None:
        """All valid service names parse correctly."""
        services = [
            "document_intelligence",
            "extraction",
            "ontology",
            "redaction",
            "compression",
            "embedding",
            "colbert",
            "colpali",
        ]
        for svc in services:
            result = ServiceConfig.model_validate({"service": svc, "config": {}})
            assert result.service == svc


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------


class TestEntity:
    """Tests for Entity model with score constraint."""

    def test_valid_score_in_range(self) -> None:
        """Entity with score 0.5 should parse successfully."""
        result = Entity(text="test", label="ORG", score=0.5)
        assert result.text == "test"
        assert result.label == "ORG"
        assert result.score == 0.5

    def test_valid_score_boundaries(self) -> None:
        """Entity with score 0.0 and 1.0 should parse successfully."""
        e0 = Entity(text="a", label="X", score=0.0)
        assert e0.score == 0.0
        e1 = Entity(text="b", label="Y", score=1.0)
        assert e1.score == 1.0

    def test_score_above_one_raises_validation_error(self) -> None:
        """Entity with score > 1 should raise ValidationError."""
        with pytest.raises(ValidationError):
            Entity(text="test", label="ORG", score=1.5)

    def test_score_below_zero_raises_validation_error(self) -> None:
        """Entity with score < 0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            Entity(text="test", label="ORG", score=-0.1)


# ---------------------------------------------------------------------------
# Relation
# ---------------------------------------------------------------------------


class TestRelation:
    """Tests for Relation model with confidence constraint."""

    def test_valid_confidence_in_range(self) -> None:
        """Relation with confidence 0.5 should parse successfully."""
        result = Relation(
            source_entity="Alice",
            target_entity="Bob",
            relation_type="works_with",
            confidence=0.5,
        )
        assert result.source_entity == "Alice"
        assert result.target_entity == "Bob"
        assert result.relation_type == "works_with"
        assert result.confidence == 0.5

    def test_valid_confidence_boundaries(self) -> None:
        """Relation with confidence 0.0 and 1.0 should parse successfully."""
        r0 = Relation(source_entity="A", target_entity="B", relation_type="R", confidence=0.0)
        assert r0.confidence == 0.0
        r1 = Relation(source_entity="A", target_entity="B", relation_type="R", confidence=1.0)
        assert r1.confidence == 1.0

    def test_confidence_above_one_raises_validation_error(self) -> None:
        """Relation with confidence > 1 should raise ValidationError."""
        with pytest.raises(ValidationError):
            Relation(
                source_entity="A",
                target_entity="B",
                relation_type="R",
                confidence=1.5,
            )

    def test_confidence_below_zero_raises_validation_error(self) -> None:
        """Relation with confidence < 0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            Relation(
                source_entity="A",
                target_entity="B",
                relation_type="R",
                confidence=-0.1,
            )


# ---------------------------------------------------------------------------
# CreditsResponse
# ---------------------------------------------------------------------------


class TestCreditsResponse:
    """Tests for CreditsResponse model."""

    def test_parse_credits_response(self) -> None:
        """Parse credits/balance response."""
        fixture = {"balance_usd": 10.5}
        result = CreditsResponse.model_validate(fixture)
        assert result.balance_usd == 10.5

    def test_parse_credits_response_with_base_fields(self) -> None:
        """Parse credits response with BaseResponse fields."""
        fixture = {"success": True, "request_id": "req_1", "balance_usd": 0.0}
        result = CreditsResponse.model_validate(fixture)
        assert result.balance_usd == 0.0
        assert result.success is True
        assert result.request_id == "req_1"


# ---------------------------------------------------------------------------
# JobCancelResponse
# ---------------------------------------------------------------------------


class TestJobCancelResponse:
    """Tests for JobCancelResponse model."""

    def test_parse_job_cancel_response(self) -> None:
        """Parse job cancellation response."""
        fixture = {"message": "Job cancelled successfully"}
        result = JobCancelResponse.model_validate(fixture)
        assert result.message == "Job cancelled successfully"
        assert result.warning is None

    def test_parse_job_cancel_response_with_warning(self) -> None:
        """Parse job cancellation response with warning."""
        fixture = {
            "message": "Job cancelled",
            "warning": "Job was already completed",
        }
        result = JobCancelResponse.model_validate(fixture)
        assert result.message == "Job cancelled"
        assert result.warning == "Job was already completed"
