"""
Smoke tests for Latence API integration.
These tests verify basic connectivity and functionality against the staging API.

Run with:
    LATENCE_BASE_URL=https://staging.api.latence.ai LATENCE_API_KEY=your_key pytest tests/integration/ -v
"""

import os
import pytest
from latence import Latence


# Skip all tests if no API key is configured
pytestmark = pytest.mark.skipif(
    not os.environ.get("LATENCE_API_KEY"),
    reason="LATENCE_API_KEY not set - skipping integration tests"
)


@pytest.fixture
def client():
    """Create a Latence client for testing."""
    return Latence()


class TestAPIConnectivity:
    """Test basic API connectivity."""

    def test_client_initializes(self, client):
        """Test that client initializes with environment config."""
        assert client is not None
        # Client should have picked up base URL from environment
        base_url = os.environ.get("LATENCE_BASE_URL", "https://api.latence.ai")
        assert client.base_url == base_url


class TestEmbeddingService:
    """Test embedding service functionality."""

    def test_embed_single_text(self, client):
        """Test embedding a single text string."""
        result = client.embedding.embed(
            text="Hello, world!",
            dimension=256
        )
        assert result is not None
        assert hasattr(result, "embeddings")
        # embeddings is a list of vectors; single text returns 1 vector of `dimension` floats
        assert len(result.embeddings) >= 1
        assert len(result.embeddings[0]) == 256

    def test_embed_batch(self, client):
        """Test embedding multiple texts."""
        texts = ["First text", "Second text", "Third text"]
        result = client.embedding.embed(
            text=texts,
            dimension=256
        )
        assert result is not None
        # Should return embeddings for each text
        assert hasattr(result, "embeddings")


class TestCreditsService:
    """Test credits checking functionality."""

    def test_check_credits(self, client):
        """Test credits balance check."""
        result = client.credits.balance()
        assert result is not None
        assert hasattr(result, "balance_usd")


class TestHealthCheck:
    """Test API health endpoints."""

    def test_api_reachable(self, client):
        """Test that API is reachable and responding."""
        # Simple embedding call to verify API is up
        try:
            result = client.embedding.embed(
                text="health check",
                dimension=256
            )
            assert result is not None
        except Exception as e:
            pytest.fail(f"API not reachable: {e}")


# =============================================================================
# Chunking Service Tests
# =============================================================================

CHUNKING_TEXT = (
    "# Introduction to Machine Learning\n\n"
    "Machine learning is a subset of artificial intelligence that focuses on "
    "building systems that learn from data. Unlike traditional programming where "
    "rules are explicitly coded, ML systems discover patterns autonomously.\n\n"
    "## Supervised Learning\n\n"
    "In supervised learning, models are trained on labeled datasets. Each training "
    "example consists of an input-output pair, and the model learns a mapping "
    "function from inputs to outputs. Common algorithms include linear regression, "
    "decision trees, random forests, and neural networks.\n\n"
    "### Classification\n\n"
    "Classification assigns discrete labels to inputs. For example, spam detection "
    "classifies emails as spam or not-spam. The model learns decision boundaries "
    "that separate different classes in the feature space.\n\n"
    "### Regression\n\n"
    "Regression predicts continuous values. House price prediction is a classic "
    "regression task where the model learns to estimate prices based on features "
    "like square footage, location, and number of bedrooms.\n\n"
    "## Unsupervised Learning\n\n"
    "Unsupervised learning discovers hidden patterns in unlabeled data. Clustering "
    "algorithms like K-means group similar data points together, while dimensionality "
    "reduction techniques like PCA compress high-dimensional data into fewer dimensions "
    "while preserving important structure.\n\n"
    "## Reinforcement Learning\n\n"
    "Reinforcement learning trains agents through trial and error. The agent takes "
    "actions in an environment and receives rewards or penalties. Over time, it learns "
    "a policy that maximizes cumulative reward. Applications include game playing, "
    "robotics, and autonomous vehicles.\n\n"
    "## Deep Learning\n\n"
    "Deep learning uses neural networks with many layers to learn hierarchical "
    "representations. Convolutional neural networks excel at image recognition, "
    "recurrent networks handle sequential data, and transformers have revolutionized "
    "natural language processing. Modern large language models like GPT are built on "
    "the transformer architecture and are trained on massive text corpora.\n\n"
    "## Evaluation Metrics\n\n"
    "Model performance is measured using metrics appropriate to the task. Accuracy, "
    "precision, recall, and F1-score are used for classification. Mean squared error "
    "and R-squared are common for regression. Cross-validation provides robust "
    "estimates by testing on multiple data splits."
)


class TestChunkingService:
    """Test chunking service with all 4 strategies against the live API."""

    def test_chunk_hybrid(self, client):
        """Hybrid strategy: character splits refined with semantic coherence."""
        result = client.experimental.chunking.chunk(
            text=CHUNKING_TEXT,
            strategy="hybrid",
            chunk_size=256,
            chunk_overlap=30,
        )
        assert result is not None
        assert result.data.num_chunks > 0
        assert result.data.strategy == "hybrid"
        assert result.data.chunk_size == 256
        assert len(result.data.chunks) == result.data.num_chunks

        chunk = result.data.chunks[0]
        assert chunk.content, "Chunk content should be non-empty"
        assert chunk.index == 0
        assert chunk.char_count > 0

        print(f"\n  [PASS] Hybrid chunking: {result.data.num_chunks} chunks, "
              f"{result.data.processing_time_ms:.1f}ms")

    def test_chunk_character_free(self, client):
        """Character strategy: fixed-length splits (free tier)."""
        result = client.experimental.chunking.chunk(
            text=CHUNKING_TEXT,
            strategy="character",
            chunk_size=500,
        )
        assert result is not None
        assert result.data.num_chunks > 0
        assert result.data.strategy == "character"
        assert len(result.data.chunks) == result.data.num_chunks

        for chunk in result.data.chunks:
            assert chunk.char_count > 0
            assert chunk.start >= 0
            assert chunk.end > chunk.start

        print(f"\n  [PASS] Character chunking (free): {result.data.num_chunks} chunks")

    def test_chunk_token_free(self, client):
        """Token strategy: token-boundary splits (free tier)."""
        result = client.experimental.chunking.chunk(
            text=CHUNKING_TEXT,
            strategy="token",
            chunk_size=128,
        )
        assert result is not None
        assert result.data.num_chunks > 0
        assert result.data.strategy == "token"

        for chunk in result.data.chunks:
            assert chunk.content, "Chunk content should be non-empty"
            if chunk.token_count is not None:
                assert chunk.token_count > 0

        print(f"\n  [PASS] Token chunking (free): {result.data.num_chunks} chunks")

    def test_chunk_semantic(self, client):
        """Semantic strategy: embedding-based grouping (charged)."""
        result = client.experimental.chunking.chunk(
            text=CHUNKING_TEXT,
            strategy="semantic",
            chunk_size=512,
        )
        assert result is not None
        assert result.data.num_chunks > 0
        assert result.data.strategy == "semantic"

        print(f"\n  [PASS] Semantic chunking: {result.data.num_chunks} chunks")

    def test_chunk_metadata_fields(self, client):
        """Verify all expected metadata fields are present on chunks."""
        result = client.experimental.chunking.chunk(
            text=CHUNKING_TEXT,
            strategy="hybrid",
            chunk_size=512,
        )
        assert result.data.num_chunks > 0

        chunk = result.data.chunks[0]
        assert hasattr(chunk, "content")
        assert hasattr(chunk, "index")
        assert hasattr(chunk, "start")
        assert hasattr(chunk, "end")
        assert hasattr(chunk, "char_count")
        assert isinstance(chunk.index, int)
        assert isinstance(chunk.start, int)
        assert isinstance(chunk.end, int)
        assert isinstance(chunk.char_count, int)

        print(f"\n  [PASS] Chunk metadata fields verified")

    def test_chunk_min_chunk_size(self, client):
        """Verify min_chunk_size parameter discards small chunks."""
        result_low = client.experimental.chunking.chunk(
            text=CHUNKING_TEXT,
            strategy="character",
            chunk_size=200,
            min_chunk_size=10,
        )
        result_high = client.experimental.chunking.chunk(
            text=CHUNKING_TEXT,
            strategy="character",
            chunk_size=200,
            min_chunk_size=150,
        )
        assert result_low.data.num_chunks >= result_high.data.num_chunks

        print(f"\n  [PASS] min_chunk_size: low={result_low.data.num_chunks}, "
              f"high={result_high.data.num_chunks}")


# =============================================================================
# Pipeline with Chunking integration test
# =============================================================================

PIPELINE_CHUNKING_TEXT = (
    "# Data Processing Overview\n\n"
    "Data processing is the collection and transformation of raw data into "
    "meaningful information. Organizations use data pipelines to automate "
    "the ingestion, cleaning, and enrichment of data.\n\n"
    "## ETL Pipelines\n\n"
    "Extract-Transform-Load pipelines move data from source systems into "
    "data warehouses. The extraction phase reads from databases, APIs, or "
    "files. Transformation cleans, deduplicates, and reshapes the data. "
    "Loading writes the processed data into the target system.\n\n"
    "## Stream Processing\n\n"
    "Stream processing handles data in real-time as it arrives. Apache Kafka "
    "and Apache Flink are popular frameworks for building event-driven "
    "architectures that process millions of events per second."
)


class TestStandaloneChunkingNotInPipeline:
    """Verify chunking is standalone and cannot be added to pipelines."""

    def test_builder_chunking_raises(self):
        """PipelineBuilder.chunking() must raise NotImplementedError."""
        from latence import PipelineBuilder
        with pytest.raises(NotImplementedError, match="not available as a pipeline step"):
            PipelineBuilder().chunking()

    def test_standalone_chunking_works(self, client):
        """Standalone chunking via direct API should work."""
        result = client.experimental.chunking.chunk(
            text=PIPELINE_CHUNKING_TEXT,
            strategy="character",
            chunk_size=300,
        )
        assert result is not None
        assert result.data.num_chunks > 0
        print(f"\n  [PASS] Standalone chunking: {result.data.num_chunks} chunks")


# Optional: Longer running tests that can be skipped in CI
class TestDocumentIntelligence:
    """Test document intelligence service (slower tests)."""

    @pytest.mark.slow
    def test_process_text_document(self, client):
        """Test processing a simple text document."""
        # This test is marked slow - skip in quick CI runs
        # Run with: pytest -m "not slow" to skip
        pass  # Placeholder - implement when needed
