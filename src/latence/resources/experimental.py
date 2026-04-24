"""Experimental namespace: self-service direct access to individual services.

This module contains the ``ExperimentalNamespace`` and
``AsyncExperimentalNamespace`` classes that group all direct service
resources behind ``client.experimental``.

Direct endpoint usage is not covered by Enterprise SLAs.  For production
workloads, use ``client.pipeline``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .chunking import AsyncChunking, Chunking
from .colbert import AsyncColBERT, ColBERT
from .colpali import AsyncColPali, ColPali
from .compression import AsyncCompression, Compression
from .dataset_intelligence_service import (
    AsyncDatasetIntelligenceService,
    DatasetIntelligenceService,
)
from .document_intelligence import AsyncDocumentIntelligence, DocumentIntelligence
from .embed import AsyncEmbed, Embed
from .embedding import AsyncEmbedding, Embedding
from .enrichment import AsyncEnrichment, Enrichment
from .extraction import AsyncExtraction, Extraction
from .ontology import AsyncOntology, Ontology
from .redaction import AsyncRedaction, Redaction
from .trace import AsyncTrace, Trace

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient

logger = logging.getLogger("latence")


class ExperimentalNamespace:
    """Container for direct service resources (self-service / developer sandbox).

    Accessible via ``client.experimental``.  All services are available
    exactly as before, but grouped under this namespace to signal that
    direct endpoint usage is billed at on-demand rates and is not
    covered by Enterprise SLAs.

    Example::

        result = client.experimental.extraction.extract(text="Apple is in Cupertino.")
        result = client.experimental.embed.dense(text="Hello world", dimension=512)
    """

    def __init__(self, client: BaseSyncClient) -> None:
        self._client = client
        self._warned = False

        # Eagerly instantiate all service resources
        self._embed = Embed(client)
        self._embedding = Embedding(client)
        self._colbert = ColBERT(client)
        self._colpali = ColPali(client)
        self._chunking = Chunking(client)
        self._compression = Compression(client)
        self._document_intelligence = DocumentIntelligence(client)
        self._enrichment = Enrichment(client)
        self._extraction = Extraction(client)
        self._dataset_intelligence_service = DatasetIntelligenceService(client)
        self._ontology = Ontology(client)
        self._redaction = Redaction(client)
        self._trace = Trace(client)

    def _log_once(self) -> None:
        if not self._warned:
            logger.info(
                "Experimental mode: direct service requests are billed at "
                "on-demand rates and are not covered by Enterprise SLAs. "
                "Use client.pipeline for production workloads."
            )
            self._warned = True

    @property
    def dataset_intelligence_service(self) -> DatasetIntelligenceService:
        """Dataset Intelligence service (corpus-level KG, ontology, incremental ingestion)."""
        self._log_once()
        return self._dataset_intelligence_service

    @property
    def embed(self) -> Embed:
        """Unified embedding service (dense, late interaction, image)."""
        self._log_once()
        return self._embed

    @property
    def embedding(self) -> Embedding:
        """Legacy dense embedding service."""
        self._log_once()
        return self._embedding

    @property
    def colbert(self) -> ColBERT:
        """ColBERT token-level embedding service."""
        self._log_once()
        return self._colbert

    @property
    def colpali(self) -> ColPali:
        """ColPali visual embedding service."""
        self._log_once()
        return self._colpali

    @property
    def chunking(self) -> Chunking:
        """Text chunking service — 4 strategies, Markdown-aware."""
        self._log_once()
        return self._chunking

    @property
    def compression(self) -> Compression:
        """Text compression service."""
        self._log_once()
        return self._compression

    @property
    def document_intelligence(self) -> DocumentIntelligence:
        """Document Intelligence (OCR) service."""
        self._log_once()
        return self._document_intelligence

    @property
    def enrichment(self) -> Enrichment:
        """Enrichment service (Coming Soon — corpus-level feature computation)."""
        self._log_once()
        return self._enrichment

    @property
    def extraction(self) -> Extraction:
        """Entity extraction service."""
        self._log_once()
        return self._extraction

    @property
    def ontology(self) -> Ontology:
        """Relation Extraction service."""
        self._log_once()
        return self._ontology

    @property
    def redaction(self) -> Redaction:
        """PII redaction service."""
        self._log_once()
        return self._redaction

    @property
    def trace(self) -> Trace:
        """Trace service — groundedness + phantom-hallucination scoring (RAG / code / rollup)."""
        self._log_once()
        return self._trace


class AsyncExperimentalNamespace:
    """Async container for direct service resources.

    Accessible via ``client.experimental`` on :class:`AsyncLatence`.
    See :class:`ExperimentalNamespace` for full documentation.
    """

    def __init__(self, client: BaseAsyncClient) -> None:
        self._client = client
        self._warned = False

        self._embed = AsyncEmbed(client)
        self._embedding = AsyncEmbedding(client)
        self._colbert = AsyncColBERT(client)
        self._colpali = AsyncColPali(client)
        self._chunking = AsyncChunking(client)
        self._compression = AsyncCompression(client)
        self._document_intelligence = AsyncDocumentIntelligence(client)
        self._enrichment = AsyncEnrichment(client)
        self._extraction = AsyncExtraction(client)
        self._dataset_intelligence_service = AsyncDatasetIntelligenceService(client)
        self._ontology = AsyncOntology(client)
        self._redaction = AsyncRedaction(client)
        self._trace = AsyncTrace(client)

    def _log_once(self) -> None:
        if not self._warned:
            logger.info(
                "Experimental mode: direct service requests are billed at "
                "on-demand rates and are not covered by Enterprise SLAs. "
                "Use client.pipeline for production workloads."
            )
            self._warned = True

    @property
    def dataset_intelligence_service(self) -> AsyncDatasetIntelligenceService:
        """Dataset Intelligence service (async)."""
        self._log_once()
        return self._dataset_intelligence_service

    @property
    def embed(self) -> AsyncEmbed:
        """Unified embedding service (async)."""
        self._log_once()
        return self._embed

    @property
    def embedding(self) -> AsyncEmbedding:
        """Legacy dense embedding service (async)."""
        self._log_once()
        return self._embedding

    @property
    def colbert(self) -> AsyncColBERT:
        """ColBERT token-level embedding service (async)."""
        self._log_once()
        return self._colbert

    @property
    def colpali(self) -> AsyncColPali:
        """ColPali visual embedding service (async)."""
        self._log_once()
        return self._colpali

    @property
    def chunking(self) -> AsyncChunking:
        """Text chunking service — 4 strategies, Markdown-aware (async)."""
        self._log_once()
        return self._chunking

    @property
    def compression(self) -> AsyncCompression:
        """Text compression service (async)."""
        self._log_once()
        return self._compression

    @property
    def document_intelligence(self) -> AsyncDocumentIntelligence:
        """Document Intelligence (OCR) service (async)."""
        self._log_once()
        return self._document_intelligence

    @property
    def enrichment(self) -> AsyncEnrichment:
        """Enrichment service (Coming Soon — corpus-level feature computation, async)."""
        self._log_once()
        return self._enrichment

    @property
    def extraction(self) -> AsyncExtraction:
        """Entity extraction service (async)."""
        self._log_once()
        return self._extraction

    @property
    def ontology(self) -> AsyncOntology:
        """Relation Extraction service (async)."""
        self._log_once()
        return self._ontology

    @property
    def redaction(self) -> AsyncRedaction:
        """PII redaction service (async)."""
        self._log_once()
        return self._redaction

    @property
    def trace(self) -> AsyncTrace:
        """Trace service (async) — groundedness + phantom-hallucination scoring."""
        self._log_once()
        return self._trace
