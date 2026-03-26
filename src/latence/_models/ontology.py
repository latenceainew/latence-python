"""Pydantic models for the Relation Extraction service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .common import BaseResponse, Entity, KnowledgeGraph, Usage


class OntologyEntityRef(BaseModel):
    """Entity reference in a relation (from super-pod)."""

    model_config = ConfigDict(extra="allow")

    text: str = Field(description="Entity text")
    label: str = Field(description="Entity type/label")
    start: int = Field(description="Start position")
    end: int = Field(description="End position")
    index: int | None = Field(default=None, description="Entity index")


class OntologyRelation(BaseModel):
    """A relation between two entities (matches super-pod output)."""

    model_config = ConfigDict(extra="allow")

    entity1: OntologyEntityRef = Field(description="Source entity")
    entity2: OntologyEntityRef = Field(description="Target entity")
    score: float = Field(description="Confidence score (0-1)")
    relation_label: str | None = Field(default=None, description="Relation type label")
    relation: str | None = Field(default=None, description="Relation type (alternative)")
    predicted: bool = Field(default=False, description="Whether relation was predicted")

    @property
    def relation_type(self) -> str:
        """Get the relation type (handles both field names)."""
        return self.relation_label or self.relation or "RELATED_TO"

    @property
    def confidence(self) -> float:
        """Alias for score to match common Relation interface."""
        return self.score

    @property
    def source_entity(self) -> str:
        """Get source entity text."""
        return self.entity1.text

    @property
    def target_entity(self) -> str:
        """Get target entity text."""
        return self.entity2.text


class OntologyConfig(BaseModel):
    """Configuration for knowledge graph construction."""

    relation_threshold: float = Field(default=0.6, description="Relation confidence threshold")
    symmetric: bool = Field(default=True, description="Create bidirectional relations")
    generate_knowledge_graph: bool = Field(default=True, description="Generate KG output")
    max_relations_per_decode: int = Field(default=30, description="Max relations per batch")
    resolve_entities: bool = Field(default=True, description="Merge duplicate entities")
    optimize_relations: bool = Field(default=True, description="Refine relation labels")
    optimize_entity_resolution: bool = Field(default=False, description="Optimize entity resolution")
    predict_missing_relations: bool = Field(default=False, description="Predict missing links")
    link_prediction_verify_with_ai: bool = Field(
        default=False, description="AI verification for predicted links"
    )
    kg_output_format: Literal["custom", "rdf", "property_graph"] = Field(
        default="custom", description="Knowledge graph output format"
    )
    namespace_uri: str | None = Field(default=None, description="RDF namespace URI")


class EntityInput(BaseModel):
    """Entity input for ontology service."""

    text: str = Field(description="Entity text")
    label: str = Field(description="Entity type/label")
    start: int = Field(description="Start position")
    end: int = Field(description="End position")
    score: float = Field(default=1.0, description="Confidence score")
    index: int = Field(description="Entity index")


class BuildGraphResponse(BaseResponse):
    """Response from knowledge graph construction."""

    entities: list[Entity] = Field(default_factory=list, description="Processed entities")
    relations: list[OntologyRelation] = Field(default_factory=list, description="Extracted relations")
    # knowledge_graph can be:
    # - KnowledgeGraph (dict with nodes/edges) for "custom" and "property_graph" formats
    # - str (Turtle syntax) for "rdf" format
    knowledge_graph: KnowledgeGraph | str | None = Field(
        default=None, description="Generated knowledge graph (KnowledgeGraph or RDF string)"
    )
    entity_count: int = Field(default=0, description="Number of entities")
    relation_count: int = Field(default=0, description="Number of relations")
    resolved_entities: int | None = Field(default=None, description="Entities after resolution")
    processing_stats: dict[str, Any] | None = Field(default=None, description="Processing stats")
    usage: Usage | None = Field(default=None, description="Credit usage")
