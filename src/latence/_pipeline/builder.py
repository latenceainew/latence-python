"""Pipeline builder with fluent API for constructing pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .._models.pipeline import PipelineConfig, ServiceConfig

logger = logging.getLogger("latence.pipeline")


class PipelineBuilder:
    """Build and configure pipelines with a fluent API.

    Example:
        >>> pipeline = (
        ...     PipelineBuilder()
        ...     .doc_intel(mode="performance")
        ...     .extraction(threshold=0.3, user_labels=["person", "org"])
        ...     .ontology(resolve_entities=True)
        ...     .store_intermediate()
        ...     .build()
        ... )
    """

    _VALID_MODES = {"default", "performance"}
    _VALID_LABEL_MODES = {"user", "hybrid", "generated"}
    _VALID_REDACTION_MODES = {"balanced", "strict", "recall", "precision"}
    _VALID_KG_FORMATS = {"custom", "property_graph", "rdf"}
    _VALID_DIMENSIONS = {256, 512, 768, 1024}

    def __init__(self) -> None:
        """Initialize an empty pipeline builder."""
        self._services: list[ServiceConfig] = []
        self._store_intermediate = True
        self._strict_mode = False

    @staticmethod
    def _check_range(name: str, value: float, lo: float = 0.0, hi: float = 1.0) -> None:
        if not (lo <= value <= hi):
            raise ValueError(f"{name} must be between {lo} and {hi}, got {value}")

    @staticmethod
    def _check_choice(name: str, value: str, choices: set[str]) -> None:
        if value not in choices:
            raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}")

    @classmethod
    def from_yaml(cls, path: str | Path, *, strict: bool = False) -> "PipelineBuilder":
        """Load a pipeline configuration from a YAML file.

        Returns a ``PipelineBuilder`` pre-populated with the services from
        the YAML so you can chain additional methods before calling
        ``build()``.

        Example YAML::

            steps:
              document_intelligence:
                mode: performance
              extraction:
                label_mode: hybrid
                user_labels: [person, organization]
              ontology:
                resolve_entities: true
            name: my-pipeline

        Args:
            path: Path to the YAML configuration file.
            strict: If True, raise on unknown parameters instead of warning.

        Returns:
            A ``PipelineBuilder`` pre-populated with services.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            PipelineConfigError: If the YAML is invalid.
            ImportError: If PyYAML is not installed.
        """
        from .config_loader import load_pipeline_config

        config, warnings = load_pipeline_config(path, strict=strict)
        if warnings:
            for w in warnings:
                logger.warning("Pipeline config: %s", w)

        builder = cls()
        builder._services = list(config.services)
        if config.store_intermediate is not None:
            builder._store_intermediate = config.store_intermediate
        if config.strict_mode:
            builder._strict_mode = True
        return builder

    def add(self, service: str, **config: Any) -> "PipelineBuilder":
        """Add a service with optional configuration.

        Aliases (e.g. ``"ocr"``) are resolved to canonical names.
        Duplicate services are rejected.

        Args:
            service: Service name or alias (e.g., ``"extraction"``, ``"ocr"``)
            **config: Service-specific configuration parameters

        Returns:
            Self for method chaining

        Raises:
            ValueError: If the service is already in the pipeline or unknown.
        """
        from .spec import resolve_step_name

        resolved = resolve_step_name(service)

        existing = {s.service for s in self._services}
        if resolved in existing:
            raise ValueError(
                f"Service '{resolved}' already added to pipeline. "
                f"Each service can only appear once."
            )

        self._services.append(ServiceConfig(service=resolved, config=config))  # type: ignore[arg-type]
        return self

    # =========================================================================
    # Document Intelligence
    # =========================================================================

    def doc_intel(
        self,
        mode: str = "default",
        output_format: str = "markdown",
        max_pages: int | None = None,
        use_ocr_for_image_block: bool = False,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add document intelligence service.

        Layout detection, auto-rotate, and other processing options are
        pre-configured for optimal end-to-end pipeline results and cannot
        be changed here.  For full parameter control, use the direct API.

        Args:
            mode: Processing mode ("default" or "performance")
            output_format: Output format ("markdown", "json", "html", "xlsx")
            max_pages: Maximum pages to process (None = all)
            use_ocr_for_image_block: Extract text from embedded images and
                output bounding box coordinates (default: False, +$0.25/1k pages)
            **kwargs: Additional configuration options

        Returns:
            Self for method chaining
        """
        config = {
            "mode": mode,
            "output_format": output_format,
            **kwargs,
        }

        if max_pages is not None:
            config["max_pages"] = max_pages

        config["pipeline_options"] = {
            "use_layout_detection": True,
            "use_chart_recognition": False,
            "use_seal_recognition": False,
            "use_doc_orientation_classify": True,
            "use_doc_unwarping": False,
            "use_ocr_for_image_block": use_ocr_for_image_block,
        }

        return self.add("document_intelligence", **config)

    # Alias for doc_intel
    document_intelligence = doc_intel

    # =========================================================================
    # Entity Extraction
    # =========================================================================

    def extraction(
        self,
        threshold: float = 0.3,
        user_labels: list[str] | None = None,
        label_mode: str = "generated",
        enable_refinement: bool = False,
        refinement_threshold: float = 0.5,
        enforce_refinement: bool = False,
        chunk_size: int = 1024,
        flat_ner: bool = True,
        multi_label: bool = False,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add Entity Extraction service.

        Args:
            threshold: Confidence threshold (0-1)
            user_labels: List of entity labels to extract
            label_mode: Label generation mode ("user", "hybrid", "generated")
            enable_refinement: Enable LLM refinement for better accuracy
            refinement_threshold: Threshold for refinement decisions
            enforce_refinement: Force refinement on all entities
            chunk_size: Chunk size in tokens
            flat_ner: Disable nested entities
            multi_label: Allow multiple labels per span
            **kwargs: Additional configuration options

        Returns:
            Self for method chaining
        """
        self._check_range("threshold", threshold)
        self._check_choice("label_mode", label_mode, self._VALID_LABEL_MODES)

        config: dict[str, Any] = {
            "threshold": threshold,
            "label_mode": label_mode,
            "chunk_size": chunk_size,
            "flat_ner": flat_ner,
            "multi_label": multi_label,
            **kwargs,
        }

        if user_labels:
            config["user_labels"] = user_labels

        if enable_refinement:
            config["enable_refinement"] = True
            config["refinement_threshold"] = refinement_threshold

        if enforce_refinement:
            config["enforce_refinement"] = True

        return self.add("extraction", **config)

    # =========================================================================
    # Relation Extraction
    # =========================================================================

    def ontology(
        self,
        resolve_entities: bool = True,
        optimize_relations: bool = True,
        predict_missing_relations: bool = False,
        relation_threshold: float = 0.6,
        kg_output_format: str = "custom",
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add Relation Extraction service for knowledge graph construction.

        Args:
            resolve_entities: Merge duplicate entities using hybrid similarity + embeddings.
            optimize_relations: Refine relation labels for better semantic accuracy.
            predict_missing_relations: Predict implicit relations via heuristics + embeddings.
            relation_threshold: Confidence threshold for relations (0.0-1.0).
            kg_output_format: Output format ("custom", "property_graph", "rdf").
            **kwargs: Additional configuration options.

        Returns:
            Self for method chaining
        """
        self._check_range("relation_threshold", relation_threshold)
        self._check_choice("kg_output_format", kg_output_format, self._VALID_KG_FORMATS)

        config: dict[str, Any] = {
            "resolve_entities": resolve_entities,
            "optimize_relations": optimize_relations,
            "predict_missing_relations": predict_missing_relations,
            "relation_threshold": relation_threshold,
            "kg_output_format": kg_output_format,
            **kwargs,
        }
        return self.add("ontology", **config)

    relation_extraction = ontology

    # =========================================================================
    # Redaction
    # =========================================================================

    def redaction(
        self,
        mode: str = "balanced",
        threshold: float = 0.3,
        redact: bool = True,
        redaction_mode: str = "mask",
        chunk_size: int = 1024,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add redaction service for PII detection and masking.

        Full LLM refinement is always enabled for pipeline quality.
        To control refinement manually, use the direct API instead.

        Args:
            mode: Detection mode ("balanced", "strict", "recall", "precision")
            threshold: Confidence threshold (0-1)
            redact: Whether to redact detected entities
            redaction_mode: How to redact ("mask" or "replace")
            chunk_size: Chunk size in tokens
            **kwargs: Additional configuration options

        Returns:
            Self for method chaining
        """
        self._check_choice("mode", mode, self._VALID_REDACTION_MODES)
        self._check_range("threshold", threshold)

        config: dict[str, Any] = {
            "mode": mode,
            "threshold": threshold,
            "redact": redact,
            "redaction_mode": redaction_mode,
            "chunk_size": chunk_size,
            **kwargs,
        }
        config["enforce_refinement"] = True

        return self.add("redaction", **config)

    # =========================================================================
    # Compression
    # =========================================================================

    def compression(
        self,
        compression_rate: float = 0.5,
        force_preserve_digit: bool = True,
        force_tokens: list[str] | None = None,
        apply_toon: bool = False,
        chunk_size: int = 4096,
        fallback_mode: bool = True,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add compression service for context compression.

        Args:
            compression_rate: Target compression ratio (0-1)
            force_preserve_digit: Preserve digits
            force_tokens: Tokens to always preserve
            apply_toon: Apply TOON optimization
            chunk_size: Chunk size in tokens
            fallback_mode: Enable fallback for edge cases
            **kwargs: Additional configuration options

        Returns:
            Self for method chaining
        """
        self._check_range("compression_rate", compression_rate)

        config: dict[str, Any] = {
            "compression_rate": compression_rate,
            "force_preserve_digit": force_preserve_digit,
            "apply_toon": apply_toon,
            "chunk_size": chunk_size,
            "fallback_mode": fallback_mode,
            **kwargs,
        }

        if force_tokens:
            config["force_tokens"] = force_tokens

        return self.add("compression", **config)

    # =========================================================================
    # Chunking (standalone only — not available as a pipeline step)
    # =========================================================================

    def chunking(self, **kwargs: Any) -> "PipelineBuilder":
        """Chunking is not available as a pipeline step.

        Use ``client.chunking.chunk()`` for standalone text chunking via the
        API. Chunking will be integrated into the ingestion pipeline in a
        future release.

        Raises:
            NotImplementedError: Always — use the standalone chunking API.
        """
        raise NotImplementedError(
            "Chunking is not available as a pipeline step. "
            "Use client.chunking.chunk() for standalone text chunking. "
            "Pipeline-integrated chunking is coming in a future release."
        )

    # =========================================================================
    # Enrichment (Coming Soon — corpus-level feature computation)
    # =========================================================================

    def enrichment(self, **kwargs: Any) -> "PipelineBuilder":
        """Add Feature Enrichment service.

        Computes 10 retrieval-optimized feature groups per chunk and at
        corpus level: quality, density, structural, semantic, compression,
        zipf, coherence, spectral, drift, redundancy.

        .. note:: This service is not yet available. Corpus-level processing
           requires a dedicated streaming architecture.

        Raises:
            NotImplementedError: Always — Feature Enrichment is coming soon.
        """
        raise NotImplementedError(
            "Feature Enrichment (10-dimensional per-chunk and corpus-level feature "
            "computation for retrieval-optimized data) is coming soon. Corpus-level "
            "processing requires a dedicated architecture. "
            "Follow our roadmap at https://latence.ai for updates."
        )

    # =========================================================================
    # Graph and Ontology Builder (coming soon)
    # =========================================================================

    def graph_ontology_builder(self, **kwargs: Any) -> "PipelineBuilder":
        """Add Graph and Ontology Builder.

        Resolves entities across documents, builds ontologies, and links
        entities to external knowledge bases.

        .. note:: This service is not yet available.

        Raises:
            NotImplementedError: Always -- Graph and Ontology Builder is coming soon.
        """
        raise NotImplementedError(
            "Graph and Ontology Builder (cross-document entity resolution, "
            "ontology construction, and knowledge base linking) is coming soon. "
            "Follow our roadmap at https://latence.ai for updates."
        )

    # Keep legacy aliases for backwards compatibility
    graph_resolution = graph_ontology_builder
    entity_linking = graph_ontology_builder

    # =========================================================================
    # Embedding
    # =========================================================================

    def embedding(
        self,
        dimension: int = 512,
        encoding_format: str = "float",
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add embedding service for dense vector embeddings.

        Args:
            dimension: Embedding dimension (256, 512, 768, or 1024)
            encoding_format: Output format ("float" or "base64")
            **kwargs: Additional configuration options

        Returns:
            Self for method chaining
        """
        if dimension not in self._VALID_DIMENSIONS:
            raise ValueError(
                f"dimension must be one of {sorted(self._VALID_DIMENSIONS)}, got {dimension}"
            )

        config: dict[str, Any] = {
            "dimension": dimension,
            "encoding_format": encoding_format,
            **kwargs,
        }
        return self.add("embedding", **config)

    # =========================================================================
    # ColBERT
    # =========================================================================

    def colbert(
        self,
        is_query: bool = False,
        query_expansion: bool = False,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add ColBERT service for token-level embeddings.

        Args:
            is_query: Whether this is a query (vs document) embedding
            query_expansion: Enable query expansion for better recall
            **kwargs: Additional configuration options

        Returns:
            Self for method chaining
        """
        config: dict[str, Any] = {
            "is_query": is_query,
            "query_expansion": query_expansion,
            **kwargs,
        }
        return self.add("colbert", **config)

    # =========================================================================
    # ColPali
    # =========================================================================

    def colpali(
        self,
        is_query: bool = False,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add ColPali service for visual embeddings.

        Args:
            is_query: Whether this is a query embedding
            **kwargs: Additional configuration options

        Returns:
            Self for method chaining
        """
        config: dict[str, Any] = {
            "is_query": is_query,
            **kwargs,
        }
        return self.add("colpali", **config)

    # =========================================================================
    # Pipeline Configuration
    # =========================================================================

    def store_intermediate(self, enabled: bool = True) -> "PipelineBuilder":
        """Store results from each pipeline stage.

        When enabled, intermediate results from each service will be
        stored and available in the final result.

        Args:
            enabled: Whether to store intermediate results

        Returns:
            Self for method chaining
        """
        self._store_intermediate = enabled
        return self

    def strict(self) -> "PipelineBuilder":
        """Enable strict mode (no auto-injection of services).

        In strict mode, the pipeline will raise an error if validation
        fails, rather than automatically injecting missing services.

        Returns:
            Self for method chaining
        """
        self._strict_mode = True
        return self

    # =========================================================================
    # Build
    # =========================================================================

    def build(self, *, input_type: str = "file") -> PipelineConfig:
        """Build the pipeline configuration.

        Runs the DAG-aware validator to auto-inject missing parent services
        (e.g. ``document_intelligence`` for file-based pipelines) and to
        check dependency correctness.

        Args:
            input_type: Hint for input kind (``"file"``, ``"text"``, or
                ``"entities"``).  Defaults to ``"file"`` for backwards
                compatibility.  Text-only and entity-only pipelines will
                NOT get ``document_intelligence`` auto-injected.

        Returns:
            PipelineConfig ready for execution

        Raises:
            PipelineValidationError: In strict mode when validation fails.
        """
        from .._models.pipeline import PipelineInput
        from .validator import validate_pipeline

        services = list(self._services)

        if not services and input_type == "file":
            from .spec import DEFAULT_INTELLIGENCE_PIPELINE

            services = list(DEFAULT_INTELLIGENCE_PIPELINE)

        config = PipelineConfig(
            services=services,
            store_intermediate=self._store_intermediate,
            strict_mode=self._strict_mode,
        )

        fake_input: PipelineInput | None = None
        if input_type == "text":
            fake_input = PipelineInput(text="<placeholder>")
        elif input_type == "entities":
            fake_input = PipelineInput(entities=[])
        elif input_type == "file":
            from .._models.pipeline import FileInput

            fake_input = PipelineInput(files=[FileInput(base64="<placeholder>")])

        result = validate_pipeline(config, fake_input)

        if result.auto_injected:
            svc_map = {sc.service: sc for sc in services}
            services = [
                svc_map.get(s, ServiceConfig(service=s, config={}))  # type: ignore[call-overload, arg-type]
                for s in result.services
            ]
            if not self._strict_mode:
                for name in result.auto_injected:
                    logger.info("Auto-injected service '%s' to satisfy DAG dependencies", name)

        if result.warnings and not self._strict_mode:
            for w in result.warnings:
                logger.info("Pipeline validation: %s", w)

        return PipelineConfig(
            services=services,
            store_intermediate=self._store_intermediate,
            strict_mode=self._strict_mode,
        )

    def __repr__(self) -> str:
        """String representation of the builder."""
        services = [s.service for s in self._services]
        return f"PipelineBuilder(services={services}, strict={self._strict_mode})"
