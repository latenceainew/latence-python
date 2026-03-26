"""Service resource classes for the Latence SDK."""

from __future__ import annotations

from .chunking import AsyncChunking, Chunking
from .colbert import AsyncColBERT, ColBERT
from .colpali import AsyncColPali, ColPali
from .compression import AsyncCompression, Compression
from .credits import AsyncCredits, Credits
from .dataset_intelligence_service import (
    AsyncDatasetIntelligenceService,
    DatasetIntelligenceService,
)
from .document_intelligence import AsyncDocumentIntelligence, DocumentIntelligence
from .embed import AsyncEmbed, Embed
from .embedding import AsyncEmbedding, Embedding
from .enrichment import AsyncEnrichment, Enrichment
from .experimental import AsyncExperimentalNamespace, ExperimentalNamespace
from .extraction import AsyncExtraction, Extraction
from .jobs import AsyncJobs, Jobs
from .ontology import AsyncOntology, Ontology
from .pipeline import AsyncPipeline, Pipeline
from .redaction import AsyncRedaction, Redaction

__all__ = [
    # Sync resources
    "Chunking",
    "ColBERT",
    "ColPali",
    "Compression",
    "Credits",
    "DatasetIntelligenceService",
    "DocumentIntelligence",
    "Embed",
    "Embedding",
    "Enrichment",
    "Extraction",
    "Jobs",
    "Ontology",
    "Pipeline",
    "Redaction",
    "ExperimentalNamespace",
    # Async resources
    "AsyncChunking",
    "AsyncColBERT",
    "AsyncColPali",
    "AsyncCompression",
    "AsyncCredits",
    "AsyncDatasetIntelligenceService",
    "AsyncDocumentIntelligence",
    "AsyncEmbed",
    "AsyncEmbedding",
    "AsyncEnrichment",
    "AsyncExtraction",
    "AsyncJobs",
    "AsyncOntology",
    "AsyncPipeline",
    "AsyncRedaction",
    "AsyncExperimentalNamespace",
]
