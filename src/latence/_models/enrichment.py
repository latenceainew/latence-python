"""Pydantic models for the Enrichment service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .chunking import ChunkData, ChunkItem
from .common import BaseResponse, Usage


class ChunkResponse(BaseResponse):
    """Response from the enrichment chunk task."""

    data: ChunkData = Field(description="Chunk data payload")
    usage: Usage | None = Field(default=None, description="Credit usage")


# ---------------------------------------------------------------------------
# Feature sub-models (per-chunk + aggregate patterns)
# ---------------------------------------------------------------------------

class QualityPerChunk(BaseModel):
    """Per-chunk quality metrics."""

    model_config = ConfigDict(extra="allow")

    coherence_score: float = Field(description="Intra-chunk coherence (0-1)")
    is_short: bool = Field(description="Whether chunk is below short threshold")
    is_long: bool = Field(description="Whether chunk exceeds long threshold")
    word_count: int = Field(description="Number of words")
    avg_word_length: float = Field(description="Average word length in characters")


class QualityAggregate(BaseModel):
    """Aggregate quality metrics across all chunks."""

    model_config = ConfigDict(extra="allow")

    mean_coherence: float = Field(description="Mean coherence score")
    short_chunks: int = Field(description="Number of short chunks")
    long_chunks: int = Field(description="Number of long chunks")


class QualityFeature(BaseModel):
    """Quality feature group."""

    per_chunk: list[QualityPerChunk] = Field(description="Per-chunk quality metrics")
    aggregate: QualityAggregate = Field(description="Aggregate quality metrics")


class DensityPerChunk(BaseModel):
    """Per-chunk density metrics."""

    model_config = ConfigDict(extra="allow")

    unique_word_ratio: float = Field(description="Ratio of unique words to total words")
    technical_density: float = Field(description="Proportion of technical terms")
    avg_sentence_length: float = Field(description="Average sentence length in words")
    punctuation_density: float = Field(description="Punctuation to word ratio")


class DensityAggregate(BaseModel):
    """Aggregate density metrics."""

    model_config = ConfigDict(extra="allow")

    mean_unique_ratio: float = Field(description="Mean unique word ratio")
    mean_technical_density: float = Field(description="Mean technical density")


class DensityFeature(BaseModel):
    """Density feature group."""

    per_chunk: list[DensityPerChunk] = Field(description="Per-chunk density metrics")
    aggregate: DensityAggregate = Field(description="Aggregate density metrics")


class StructuralPerChunk(BaseModel):
    """Per-chunk structural metrics."""

    model_config = ConfigDict(extra="allow")

    heading_count: int = Field(description="Number of headings")
    list_count: int = Field(description="Number of list items")
    code_block_count: int = Field(description="Number of code blocks")
    link_count: int = Field(description="Number of links")
    cross_reference_count: int = Field(description="Number of cross-references")
    relative_position: float = Field(description="Position in document (0.0-1.0)")
    recency_score: float | None = Field(default=None, description="Temporal decay score")


class StructuralAggregate(BaseModel):
    """Aggregate structural metrics."""

    model_config = ConfigDict(extra="allow")

    total_headings: int = Field(description="Total headings across all chunks")
    total_lists: int = Field(description="Total list items")
    total_cross_refs: int = Field(description="Total cross-references")


class StructuralFeature(BaseModel):
    """Structural feature group."""

    per_chunk: list[StructuralPerChunk] = Field(description="Per-chunk structural metrics")
    aggregate: StructuralAggregate = Field(description="Aggregate structural metrics")


class SemanticPerChunk(BaseModel):
    """Per-chunk semantic metrics."""

    model_config = ConfigDict(extra="allow")

    rhetorical_role: str = Field(description="Classified rhetorical role")
    rhetorical_confidence: float = Field(description="Classification confidence (0-1)")
    centrality: float = Field(description="Similarity to parent document embedding (0-1)")


class SemanticAggregate(BaseModel):
    """Aggregate semantic metrics."""

    model_config = ConfigDict(extra="allow")

    role_distribution: dict[str, float] = Field(
        description="Distribution of rhetorical roles"
    )
    mean_centrality: float = Field(description="Mean centrality score")


class SemanticFeature(BaseModel):
    """Semantic feature group."""

    per_chunk: list[SemanticPerChunk] = Field(description="Per-chunk semantic metrics")
    aggregate: SemanticAggregate = Field(description="Aggregate semantic metrics")


class CompressionPerChunk(BaseModel):
    """Per-chunk compression metrics."""

    model_config = ConfigDict(extra="allow")

    compression_ratio: float = Field(description="Token-to-character compression ratio")
    unique_token_ratio: float = Field(description="Unique token ratio")


class CompressionAggregate(BaseModel):
    """Aggregate compression metrics."""

    model_config = ConfigDict(extra="allow")

    mean_compression_ratio: float = Field(description="Mean compression ratio")
    mean_unique_token_ratio: float = Field(description="Mean unique token ratio")


class CompressionFeature(BaseModel):
    """Compression feature group."""

    per_chunk: list[CompressionPerChunk] = Field(description="Per-chunk compression metrics")
    aggregate: CompressionAggregate = Field(description="Aggregate compression metrics")


class ZipfPerChunk(BaseModel):
    """Per-chunk Zipf's law metrics."""

    model_config = ConfigDict(extra="allow")

    alpha: float = Field(description="Zipf exponent (power-law fit)")
    vocab_size: int = Field(description="Vocabulary size in chunk")
    fit_quality: float = Field(description="R-squared of Zipf fit (0-1)")


class ZipfAggregate(BaseModel):
    """Aggregate Zipf metrics."""

    model_config = ConfigDict(extra="allow")

    mean_alpha: float = Field(description="Mean Zipf exponent")
    mean_vocab_size: float = Field(description="Mean vocabulary size")


class ZipfFeature(BaseModel):
    """Zipf feature group."""

    per_chunk: list[ZipfPerChunk] = Field(description="Per-chunk Zipf metrics")
    aggregate: ZipfAggregate = Field(description="Aggregate Zipf metrics")


# ---------------------------------------------------------------------------
# Aggregate-only feature groups
# ---------------------------------------------------------------------------

class CoherenceFeature(BaseModel):
    """Coherence feature group (aggregate only — inter-chunk similarity)."""

    model_config = ConfigDict(extra="allow")

    mean_similarity: float = Field(description="Mean pairwise chunk similarity")
    min_similarity: float = Field(description="Minimum pairwise similarity")
    max_similarity: float = Field(description="Maximum pairwise similarity")
    std_similarity: float = Field(description="Standard deviation of pairwise similarity")


class SpectralFeature(BaseModel):
    """Spectral feature group (aggregate only — SVD analysis of embedding matrix)."""

    model_config = ConfigDict(extra="allow")

    effective_rank: float = Field(description="Effective rank of embedding matrix")
    num_chunks: int = Field(description="Number of chunks in analysis")
    rank_ratio: float = Field(description="Effective rank / num_chunks")
    top_singular_values: list[float] = Field(
        description="Top singular values from SVD"
    )


class DriftFeature(BaseModel):
    """Drift feature group (aggregate only — topic drift between adjacent chunks)."""

    model_config = ConfigDict(extra="allow")

    similarities: list[float] = Field(
        description="Cosine similarity between consecutive chunks"
    )
    mean_similarity: float = Field(description="Mean consecutive similarity")
    major_breaks: list[int] = Field(
        description="Chunk indices where major topic drift occurs"
    )
    num_major_breaks: int = Field(description="Number of major topic breaks")


class RedundancyFeature(BaseModel):
    """Redundancy feature group (aggregate only — duplicate chunk detection)."""

    model_config = ConfigDict(extra="allow")

    redundant_pairs: list[list[int]] = Field(
        description="Pairs of chunk indices with high similarity"
    )
    num_redundant: int = Field(description="Number of redundant pairs")
    redundancy_rate: float = Field(description="Fraction of chunks that are redundant")
    threshold: float = Field(description="Similarity threshold used")


# ---------------------------------------------------------------------------
# Enrich response (task=enrich)
# ---------------------------------------------------------------------------

class EnrichData(BaseModel):
    """Data payload for enrich response."""

    model_config = ConfigDict(extra="allow")

    chunks: list[ChunkItem] = Field(description="Chunk objects with metadata")
    num_chunks: int = Field(description="Total number of chunks")
    embeddings: list[Any] = Field(
        description="Chunk embeddings (list[list[float]] or list[str] for base64)"
    )
    embedding_dim: int = Field(description="Embedding dimension")
    encoding_format: str = Field(description="Embedding format used (float or base64)")
    features: dict[str, Any] = Field(description="Feature groups keyed by name")
    strategy: str = Field(description="Chunking strategy used")
    processing_time_ms: float = Field(description="Processing time in milliseconds")


class EnrichResponse(BaseResponse):
    """Response from the enrichment enrich task."""

    data: EnrichData = Field(description="Enrich data payload")
    usage: Usage | None = Field(default=None, description="Credit usage")
