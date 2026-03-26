"""Relation Extraction service resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Union, overload

from .._models import BuildGraphResponse, EntityInput, JobSubmittedResponse, OntologyConfig
from ._base import AsyncResource, SyncResource

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient


class Ontology(SyncResource):
    """Relation Extraction service.

    Extract relations and build knowledge graphs from text
    and entities.

    Example::

        result = client.ontology.build_graph(
            text="Microsoft is headquartered in Redmond.",
            entities=[
                {"text": "Microsoft", "label": "ORG",
                 "start": 0, "end": 9, "score": 0.98},
                {"text": "Redmond", "label": "GPE",
                 "start": 29, "end": 36, "score": 0.95},
            ],
        )
        print(result.relations)
    """

    def __init__(self, client: BaseSyncClient) -> None:
        super().__init__(client)

    @overload
    def build_graph(
        self,
        text: str,
        entities: list[dict[str, Any]] | list[EntityInput],
        *,
        config: dict[str, Any] | OntologyConfig | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> BuildGraphResponse: ...

    @overload
    def build_graph(
        self,
        text: str,
        entities: list[dict[str, Any]] | list[EntityInput],
        *,
        config: dict[str, Any] | OntologyConfig | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def build_graph(
        self,
        text: str,
        entities: list[dict[str, Any]] | list[EntityInput],
        *,
        config: dict[str, Any] | OntologyConfig | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[BuildGraphResponse, JobSubmittedResponse]:
        """
        Build a knowledge graph from text and entities.

        Args:
            text: Input text
            entities: List of entities with positions
            config: Ontology configuration:
                - resolve_entities: Merge duplicate entities
                - optimize_relations: Refine relation labels with AI
                - predict_missing_relations: Predict missing links
                - kg_output_format: "custom", "rdf", or "property_graph"
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling

        Returns:
            BuildGraphResponse with relations and knowledge graph
        """
        # Convert Pydantic models to dicts
        cfg: dict[str, Any] | None = None
        if config is not None:
            if isinstance(config, OntologyConfig):
                cfg = config.model_dump(exclude_none=True)
            else:
                cfg = config

        ents: list[dict[str, Any]] = []
        for e in entities:
            if isinstance(e, EntityInput):
                ents.append(e.model_dump())
            else:
                ents.append(e)

        body = self._build_request_body(
            text=text,
            entities=ents,
            config=cfg,
            request_id=request_id,
            return_job=return_job,
        )

        response = self._client.post("/api/v1/ontology/build", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = BuildGraphResponse.model_validate(response.data)  # type: ignore[assignment]

        return self._inject_metadata(result, response)


class AsyncOntology(AsyncResource):
    """Async Relation Extraction service."""

    def __init__(self, client: BaseAsyncClient) -> None:
        super().__init__(client)

    @overload
    async def build_graph(
        self,
        text: str,
        entities: list[dict[str, Any]] | list[EntityInput],
        *,
        config: dict[str, Any] | OntologyConfig | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> BuildGraphResponse: ...

    @overload
    async def build_graph(
        self,
        text: str,
        entities: list[dict[str, Any]] | list[EntityInput],
        *,
        config: dict[str, Any] | OntologyConfig | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def build_graph(
        self,
        text: str,
        entities: list[dict[str, Any]] | list[EntityInput],
        *,
        config: dict[str, Any] | OntologyConfig | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[BuildGraphResponse, JobSubmittedResponse]:
        """
        Build a knowledge graph from text and entities.

        Args:
            text: Input text
            entities: List of entities with positions
            config: Ontology configuration:
                - resolve_entities: Merge duplicate entities
                - optimize_relations: Refine relation labels with AI
                - predict_missing_relations: Predict missing links
                - kg_output_format: "custom", "rdf", or "property_graph"
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling

        Returns:
            BuildGraphResponse with relations and knowledge graph
        """
        cfg: dict[str, Any] | None = None
        if config is not None:
            if isinstance(config, OntologyConfig):
                cfg = config.model_dump(exclude_none=True)
            else:
                cfg = config

        ents: list[dict[str, Any]] = []
        for e in entities:
            if isinstance(e, EntityInput):
                ents.append(e.model_dump())
            else:
                ents.append(e)

        body = self._build_request_body(
            text=text,
            entities=ents,
            config=cfg,
            request_id=request_id,
            return_job=return_job,
        )

        response = await self._client.post("/api/v1/ontology/build", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = BuildGraphResponse.model_validate(response.data)  # type: ignore[assignment]

        return self._inject_metadata(result, response)
