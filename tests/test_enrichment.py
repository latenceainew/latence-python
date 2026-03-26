"""Tests for the Enrichment service integration."""

import pytest

from latence._models.enrichment import (
    ChunkData,
    ChunkItem,
    ChunkResponse,
    EnrichData,
    EnrichResponse,
)
from latence._pipeline.builder import PipelineBuilder
from latence._pipeline.spec import (
    PLACEHOLDER_STEPS,
    STEP_ALIASES,
    STEP_ORDER,
    resolve_step_name,
)
from latence._pipeline.data_package import EnrichmentSection, EnrichmentSummary


# =============================================================================
# MODEL TESTS
# =============================================================================


class TestChunkModels:
    """Tests for ChunkItem and ChunkResponse models."""

    def test_chunk_item_required_fields(self):
        """ChunkItem should parse with all required fields."""
        item = ChunkItem(
            content="Hello world",
            index=0,
            start=0,
            end=11,
            char_count=11,
        )
        assert item.content == "Hello world"
        assert item.index == 0
        assert item.token_count is None
        assert item.semantic_score is None
        assert item.section_path is None

    def test_chunk_item_optional_fields(self):
        """ChunkItem should accept optional fields."""
        item = ChunkItem(
            content="Test chunk",
            index=3,
            start=100,
            end=200,
            char_count=100,
            token_count=25,
            semantic_score=0.87,
            section_path=["Chapter 1", "Section 1.1"],
        )
        assert item.token_count == 25
        assert item.semantic_score == 0.87
        assert item.section_path == ["Chapter 1", "Section 1.1"]

    def test_chunk_response_parse(self):
        """ChunkResponse should parse a valid API response."""
        raw = {
            "success": True,
            "data": {
                "chunks": [
                    {
                        "content": "First chunk",
                        "index": 0,
                        "start": 0,
                        "end": 100,
                        "char_count": 100,
                    }
                ],
                "num_chunks": 1,
                "strategy": "hybrid",
                "chunk_size": 512,
                "processing_time_ms": 42.5,
            },
        }
        resp = ChunkResponse.model_validate(raw)
        assert resp.success is True
        assert resp.data.num_chunks == 1
        assert resp.data.strategy == "hybrid"
        assert resp.data.chunks[0].content == "First chunk"
        assert resp.usage is None


class TestEnrichModels:
    """Tests for EnrichResponse models."""

    def test_enrich_response_parse(self):
        """EnrichResponse should parse a valid API response."""
        raw = {
            "success": True,
            "data": {
                "chunks": [
                    {
                        "content": "Test text",
                        "index": 0,
                        "start": 0,
                        "end": 9,
                        "char_count": 9,
                    }
                ],
                "num_chunks": 1,
                "embeddings": [[0.01, -0.02, 0.03]],
                "embedding_dim": 3,
                "encoding_format": "float",
                "features": {
                    "quality": {
                        "per_chunk": [
                            {
                                "coherence_score": 0.72,
                                "is_short": False,
                                "is_long": False,
                                "word_count": 2,
                                "avg_word_length": 4.0,
                            }
                        ],
                        "aggregate": {
                            "mean_coherence": 0.72,
                            "short_chunks": 0,
                            "long_chunks": 0,
                        },
                    }
                },
                "strategy": "hybrid",
                "processing_time_ms": 150.3,
            },
            "usage": {"credits": 0.5},
        }
        resp = EnrichResponse.model_validate(raw)
        assert resp.success is True
        assert resp.data.num_chunks == 1
        assert resp.data.embedding_dim == 3
        assert "quality" in resp.data.features
        assert resp.data.encoding_format == "float"
        assert resp.usage is not None

    def test_enrich_response_extra_fields(self):
        """EnrichResponse data should tolerate extra fields from future versions."""
        raw = {
            "success": True,
            "data": {
                "chunks": [],
                "num_chunks": 0,
                "embeddings": [],
                "embedding_dim": 0,
                "encoding_format": "float",
                "features": {},
                "strategy": "hybrid",
                "processing_time_ms": 0.0,
                "new_future_field": "should not raise",
            },
        }
        resp = EnrichResponse.model_validate(raw)
        assert resp.data.num_chunks == 0


# =============================================================================
# PIPELINE SPEC TESTS
# =============================================================================


class TestPipelineSpec:
    """Tests for enrichment pipeline spec updates."""

    def test_enrichment_in_step_order(self):
        """Enrichment should still be in the canonical step order (as placeholder)."""
        assert "enrichment" in STEP_ORDER

    def test_enrichment_is_placeholder(self):
        """Enrichment should be a placeholder step (Coming Soon)."""
        assert "enrichment" in PLACEHOLDER_STEPS

    def test_chunking_is_standalone(self):
        """Chunking is standalone -- not a pipeline step, not a placeholder."""
        assert "chunking" not in STEP_ORDER
        assert "chunking" not in PLACEHOLDER_STEPS
        with pytest.raises(NotImplementedError, match="not available"):
            resolve_step_name("chunking")

    def test_alias_enrich_resolves_to_enrichment(self):
        """'enrich' alias should resolve to 'enrichment' (but raises as placeholder)."""
        assert STEP_ALIASES["enrich"] == "enrichment"
        with pytest.raises(NotImplementedError, match="coming soon"):
            resolve_step_name("enrich")

    def test_alias_feature_enrichment_resolves_to_enrichment(self):
        """'feature_enrichment' alias should resolve to 'enrichment' (placeholder)."""
        assert STEP_ALIASES["feature_enrichment"] == "enrichment"
        with pytest.raises(NotImplementedError, match="coming soon"):
            resolve_step_name("feature_enrichment")

    def test_resolve_enrichment_raises(self):
        """resolve_step_name('enrichment') should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="coming soon"):
            resolve_step_name("enrichment")

    def test_enrichment_order_after_compression(self):
        """Enrichment should come after compression in the pipeline."""
        comp_idx = STEP_ORDER.index("compression")
        enrich_idx = STEP_ORDER.index("enrichment")
        assert enrich_idx > comp_idx


# =============================================================================
# PIPELINE BUILDER TESTS
# =============================================================================


class TestPipelineBuilderEnrichment:
    """Tests for PipelineBuilder.enrichment() (now raises NotImplementedError)."""

    def test_enrichment_method_exists(self):
        """PipelineBuilder should have an enrichment() method."""
        builder = PipelineBuilder()
        assert hasattr(builder, "enrichment")
        assert callable(builder.enrichment)

    def test_enrichment_raises_not_implemented(self):
        """enrichment() should raise NotImplementedError (Coming Soon)."""
        with pytest.raises(NotImplementedError, match="coming soon"):
            PipelineBuilder().enrichment()

    def test_chunking_raises_not_implemented(self):
        """PipelineBuilder.chunking() should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not available as a pipeline step"):
            PipelineBuilder().chunking(strategy="token")


# =============================================================================
# DATA PACKAGE TESTS
# =============================================================================


class TestEnrichmentSection:
    """Tests for EnrichmentSection in DataPackage."""

    def test_enrichment_summary_defaults(self):
        """EnrichmentSummary should have sensible defaults."""
        summary = EnrichmentSummary()
        assert summary.num_chunks == 0
        assert summary.strategy == "hybrid"
        assert summary.embedding_dim == 0
        assert summary.features_computed == []

    def test_enrichment_section_construction(self):
        """EnrichmentSection should construct with all fields."""
        section = EnrichmentSection(
            chunks=[{"content": "test", "index": 0}],
            embeddings=[[0.1, 0.2]],
            features={"quality": {"per_chunk": [], "aggregate": {}}},
            summary=EnrichmentSummary(
                num_chunks=1,
                strategy="hybrid",
                embedding_dim=2,
                features_computed=["quality"],
                processing_time_ms=100.0,
            ),
        )
        assert section.summary.num_chunks == 1
        assert len(section.chunks) == 1
        assert "quality" in section.features
