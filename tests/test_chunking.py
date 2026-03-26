"""Tests for the Chunking service integration."""

import pytest

from latence._models.chunking import (
    ChunkData,
    ChunkItem,
    ChunkResponse,
)
from latence._pipeline.builder import PipelineBuilder
from latence._pipeline.spec import (
    PLACEHOLDER_STEPS,
    STEP_ALIASES,
    STEP_ORDER,
    resolve_step_name,
)
from latence._pipeline.data_package import ChunkingSection, ChunkingSummary


# =============================================================================
# MODEL TESTS
# =============================================================================


class TestChunkModels:
    """Tests for ChunkItem and ChunkResponse models."""

    def test_chunk_item_required_fields(self):
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
        assert resp.data.chunk_size == 512
        assert resp.data.processing_time_ms == 42.5
        assert len(resp.data.chunks) == 1
        assert resp.data.chunks[0].content == "First chunk"

    def test_chunk_response_multiple_chunks(self):
        raw = {
            "success": True,
            "data": {
                "chunks": [
                    {"content": f"Chunk {i}", "index": i, "start": i * 100,
                     "end": (i + 1) * 100, "char_count": 100}
                    for i in range(5)
                ],
                "num_chunks": 5,
                "strategy": "semantic",
                "chunk_size": 256,
                "processing_time_ms": 123.4,
            },
        }
        resp = ChunkResponse.model_validate(raw)
        assert resp.data.num_chunks == 5
        assert len(resp.data.chunks) == 5


# =============================================================================
# PIPELINE SPEC TESTS
# =============================================================================


class TestChunkingPipelineSpec:
    """Tests for chunking in pipeline specification."""

    def test_chunking_not_in_step_order(self):
        """Chunking is standalone, not a pipeline step."""
        assert "chunking" not in STEP_ORDER

    def test_chunking_aliases_raise_not_implemented(self):
        """Chunking aliases should raise NotImplementedError."""
        for alias in ["chunk", "chunking", "split", "text_chunking"]:
            with pytest.raises(NotImplementedError, match="not available as a pipeline step"):
                resolve_step_name(alias)

    def test_chunking_not_in_step_aliases(self):
        """Chunking should not be in STEP_ALIASES (handled by resolve_step_name)."""
        assert "chunking" not in STEP_ALIASES
        assert "chunk" not in STEP_ALIASES

    def test_chunking_not_placeholder(self):
        assert "chunking" not in PLACEHOLDER_STEPS

    def test_enrichment_is_placeholder(self):
        assert "enrichment" in PLACEHOLDER_STEPS
        with pytest.raises(NotImplementedError, match="coming soon"):
            resolve_step_name("enrichment")
        with pytest.raises(NotImplementedError, match="coming soon"):
            resolve_step_name("enrich")


# =============================================================================
# PIPELINE BUILDER TESTS
# =============================================================================


class TestChunkingPipelineBuilder:
    """Tests for PipelineBuilder.chunking() method."""

    def test_chunking_method_exists(self):
        builder = PipelineBuilder()
        assert hasattr(builder, "chunking")

    def test_chunking_raises_not_implemented(self):
        """Chunking should raise NotImplementedError as it's standalone only."""
        with pytest.raises(NotImplementedError, match="not available as a pipeline step"):
            PipelineBuilder().chunking()

    def test_chunking_raises_with_params(self):
        """Chunking should raise even with parameters."""
        with pytest.raises(NotImplementedError, match="not available as a pipeline step"):
            PipelineBuilder().chunking(strategy="hybrid", chunk_size=512)

    def test_enrichment_raises(self):
        builder = PipelineBuilder()
        with pytest.raises(NotImplementedError, match="coming soon"):
            builder.enrichment()


# =============================================================================
# DATA PACKAGE TESTS
# =============================================================================


class TestChunkingSection:
    """Tests for ChunkingSection in DataPackage."""

    def test_chunking_section_construction(self):
        section = ChunkingSection(
            chunks=[{"content": "Test", "index": 0, "start": 0, "end": 4, "char_count": 4}],
            summary=ChunkingSummary(
                num_chunks=1,
                strategy="hybrid",
                chunk_size=512,
                processing_time_ms=10.0,
            ),
        )
        assert section.summary.num_chunks == 1
        assert section.summary.strategy == "hybrid"
        assert len(section.chunks) == 1
        assert section.chunks[0]["content"] == "Test"

    def test_chunking_summary_defaults(self):
        summary = ChunkingSummary()
        assert summary.num_chunks == 0
        assert summary.strategy == "hybrid"
        assert summary.chunk_size == 512
        assert summary.processing_time_ms == 0.0
