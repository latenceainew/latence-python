"""Dataset Intelligence service resource.

Provides access to the Dataset Intelligence API for corpus-level
entity resolution, knowledge graph construction, ontology induction,
and incremental dataset ingestion.

Example::

    # Create a new dataset (full pipeline)
    result = client.experimental.dataset_intelligence_service.run(
        input_data={"stage_01_document_intelligence": {...}, ...}
    )
    print(result.dataset_id, result.usage.credits)

    # Append to existing dataset
    result = client.experimental.dataset_intelligence_service.run(
        input_data={...},
        dataset_id="ds_abc123",
    )
    print(result.delta_summary)

    # Submit as async job
    job = client.experimental.dataset_intelligence_service.run(
        input_data={...},
        return_job=True,
    )
    print(job.job_id)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, Union, overload

import httpx

from .._constants import B2_UPLOAD_TIMEOUT
from .._models.dataset_intelligence_service import DatasetIntelligenceResponse
from .._models.jobs import JobSubmittedResponse
from ._base import AsyncResource, SyncResource

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient

_ENDPOINT = "/api/v1/dataset_intelligence/process"
_DI_PRESIGN_ENDPOINT = "/api/v1/di/presign"
_DI_PAYLOAD_THRESHOLD = 8 * 1024 * 1024  # 8 MB — above this, upload to B2


class DatasetIntelligenceService(SyncResource):
    """Dataset Intelligence API resource (synchronous)."""

    def __init__(self, client: BaseSyncClient) -> None:
        super().__init__(client)

    def _process(
        self,
        tier: str,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> DatasetIntelligenceResponse | JobSubmittedResponse:
        payload_bytes = json.dumps(input_data, default=str).encode()
        if len(payload_bytes) > _DI_PAYLOAD_THRESHOLD:
            input_url = self._upload_di_payload(payload_bytes, dataset_id)
            body = self._build_request_body(
                endpoint_id="dataset_intelligence",
                tier=tier,
                input_url=input_url,
                dataset_id=dataset_id,
                config_overrides=config_overrides,
                request_id=request_id,
                return_job=return_job,
            )
        else:
            body = self._build_request_body(
                endpoint_id="dataset_intelligence",
                tier=tier,
                input_data=input_data,
                dataset_id=dataset_id,
                config_overrides=config_overrides,
                request_id=request_id,
                return_job=return_job,
            )
        response = self._client.post(_ENDPOINT, json=body)
        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = DatasetIntelligenceResponse.model_validate(response.data)
        return self._inject_metadata(result, response)

    def _upload_di_payload(self, payload_bytes: bytes, dataset_id: str | None = None) -> str:
        """Upload large DI input payload to B2 via presigned URL, return download URL."""
        presign_body: dict[str, Any] = {"content_type": "application/json"}
        if dataset_id:
            presign_body["dataset_id"] = dataset_id
        resp = self._client.post(_DI_PRESIGN_ENDPOINT, json=presign_body)
        upload_url: str = resp.data["upload_url"]
        download_url: str = resp.data["download_url"]

        put_resp = httpx.put(
            upload_url,
            content=payload_bytes,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(B2_UPLOAD_TIMEOUT),
        )
        put_resp.raise_for_status()
        return download_url

    def enrich(
        self,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> DatasetIntelligenceResponse:
        """Tier 1: Semantic enrichment (CPU-only, fast)."""
        return self._process(
            "tier1", input_data,
            dataset_id=dataset_id,
            config_overrides=config_overrides,
            request_id=request_id,
        )  # type: ignore[return-value]

    @overload
    def build_graph(
        self,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> DatasetIntelligenceResponse: ...

    @overload
    def build_graph(
        self,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def build_graph(
        self,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[DatasetIntelligenceResponse, JobSubmittedResponse]:
        """Tier 2: Knowledge graph construction (GPU, entity linking + RotatE)."""
        return self._process(
            "tier2", input_data,
            dataset_id=dataset_id,
            config_overrides=config_overrides,
            request_id=request_id,
            return_job=return_job,
        )

    def build_ontology(
        self,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> DatasetIntelligenceResponse:
        """Tier 3: Ontology induction (concept clustering + SHACL shapes)."""
        return self._process(
            "tier3", input_data,
            dataset_id=dataset_id,
            config_overrides=config_overrides,
            request_id=request_id,
        )  # type: ignore[return-value]

    @overload
    def run(
        self,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> DatasetIntelligenceResponse: ...

    @overload
    def run(
        self,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def run(
        self,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[DatasetIntelligenceResponse, JobSubmittedResponse]:
        """Full pipeline: all 3 tiers."""
        return self._process(
            "full", input_data,
            dataset_id=dataset_id,
            config_overrides=config_overrides,
            request_id=request_id,
            return_job=return_job,
        )


class AsyncDatasetIntelligenceService(AsyncResource):
    """Dataset Intelligence API resource (asynchronous)."""

    def __init__(self, client: BaseAsyncClient) -> None:
        super().__init__(client)

    async def _process(
        self,
        tier: str,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> DatasetIntelligenceResponse | JobSubmittedResponse:
        payload_bytes = json.dumps(input_data, default=str).encode()
        if len(payload_bytes) > _DI_PAYLOAD_THRESHOLD:
            input_url = await self._upload_di_payload(payload_bytes, dataset_id)
            body = self._build_request_body(
                endpoint_id="dataset_intelligence",
                tier=tier,
                input_url=input_url,
                dataset_id=dataset_id,
                config_overrides=config_overrides,
                request_id=request_id,
                return_job=return_job,
            )
        else:
            body = self._build_request_body(
                endpoint_id="dataset_intelligence",
                tier=tier,
                input_data=input_data,
                dataset_id=dataset_id,
                config_overrides=config_overrides,
                request_id=request_id,
                return_job=return_job,
            )
        response = await self._client.post(_ENDPOINT, json=body)
        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = DatasetIntelligenceResponse.model_validate(response.data)
        return self._inject_metadata(result, response)

    async def _upload_di_payload(self, payload_bytes: bytes, dataset_id: str | None = None) -> str:
        """Upload large DI input payload to B2 via presigned URL, return download URL."""
        presign_body: dict[str, Any] = {"content_type": "application/json"}
        if dataset_id:
            presign_body["dataset_id"] = dataset_id
        resp = await self._client.post(_DI_PRESIGN_ENDPOINT, json=presign_body)
        upload_url: str = resp.data["upload_url"]
        download_url: str = resp.data["download_url"]

        async with httpx.AsyncClient(timeout=httpx.Timeout(B2_UPLOAD_TIMEOUT)) as upload_client:
            put_resp = await upload_client.put(
                upload_url,
                content=payload_bytes,
                headers={"Content-Type": "application/json"},
            )
            put_resp.raise_for_status()
        return download_url

    async def enrich(
        self,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> DatasetIntelligenceResponse:
        """Tier 1: Semantic enrichment (CPU-only, fast)."""
        return await self._process(
            "tier1", input_data,
            dataset_id=dataset_id,
            config_overrides=config_overrides,
            request_id=request_id,
        )  # type: ignore[return-value]

    @overload
    async def build_graph(
        self,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> DatasetIntelligenceResponse: ...

    @overload
    async def build_graph(
        self,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def build_graph(
        self,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[DatasetIntelligenceResponse, JobSubmittedResponse]:
        """Tier 2: Knowledge graph construction (GPU, entity linking + RotatE)."""
        return await self._process(
            "tier2", input_data,
            dataset_id=dataset_id,
            config_overrides=config_overrides,
            request_id=request_id,
            return_job=return_job,
        )

    async def build_ontology(
        self,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> DatasetIntelligenceResponse:
        """Tier 3: Ontology induction (concept clustering + SHACL shapes)."""
        return await self._process(
            "tier3", input_data,
            dataset_id=dataset_id,
            config_overrides=config_overrides,
            request_id=request_id,
        )  # type: ignore[return-value]

    @overload
    async def run(
        self,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> DatasetIntelligenceResponse: ...

    @overload
    async def run(
        self,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def run(
        self,
        input_data: dict[str, Any],
        *,
        dataset_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[DatasetIntelligenceResponse, JobSubmittedResponse]:
        """Full pipeline: all 3 tiers."""
        return await self._process(
            "full", input_data,
            dataset_id=dataset_id,
            config_overrides=config_overrides,
            request_id=request_id,
            return_job=return_job,
        )
