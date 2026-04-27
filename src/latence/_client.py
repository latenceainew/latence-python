"""Main client classes for the Latence SDK.

The primary interface is ``client.pipeline`` -- an async/job-based
Data Intelligence Pipeline.  Direct service access is available via
``client.experimental`` (self-service, not covered by Enterprise SLAs).

Example -- pipeline-first (recommended)::

    from latence import Latence

    client = Latence(api_key="lat_xxx")
    job = client.pipeline.run(files=["contract.pdf"])
    pkg = job.wait_for_completion()
    print(pkg.document.markdown)
    print(pkg.entities.summary)

Example -- experimental direct access::

    result = client.experimental.extraction.extract(text="Apple is in Cupertino.")
"""

from __future__ import annotations

from typing import Any

from ._base import BaseAsyncClient, BaseSyncClient
from ._constants import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
from ._deprecation import DeprecatedServiceProperty
from .resources import (
    AsyncCredits,
    AsyncJobs,
    AsyncPipeline,
    Credits,
    Jobs,
    Pipeline,
)
from .resources.experimental import (
    AsyncExperimentalNamespace,
    ExperimentalNamespace,
)


class Latence:
    """
    Synchronous client for the Latence API.

    The **primary interface** is ``client.pipeline`` -- an async/job-based
    Data Intelligence Pipeline.  Submit files, get a structured DataPackage
    back with document markdown, entities, knowledge graphs, and quality
    metrics.

    Example -- simplest usage::

        >>> from latence import Latence
        >>> client = Latence(api_key="lat_xxx")
        >>> job = client.pipeline.run(files=["contract.pdf"])
        >>> pkg = job.wait_for_completion()
        >>> print(pkg.document.markdown)
        >>> print(pkg.entities.summary)
        >>> print(pkg.knowledge_graph.summary)

    Example -- explicit steps::

        >>> job = client.pipeline.run(
        ...     files=["contract.pdf"],
        ...     steps={"ocr": {"mode": "performance"}, "extraction": {}, "knowledge_graph": {}},
        ...     name="Legal Contracts",
        ... )
        >>> pkg = job.wait_for_completion()

    Example -- experimental direct access (self-service)::

        >>> result = client.experimental.embed.dense(text="Hello world")
        >>> print(result.embeddings)

    With context manager::

        >>> with Latence(api_key="lat_xxx") as client:
        ...     job = client.pipeline.run(files=["doc.pdf"])
        ...     pkg = job.wait_for_completion()
    """

    pipeline: Pipeline
    """Primary interface: Data Intelligence Pipeline (async/job-based)."""

    jobs: Jobs
    """Job management service."""

    credits: Credits
    """Credit balance service."""

    # Deprecated direct service access -- delegates to self.experimental
    embed = DeprecatedServiceProperty("embed")
    embedding = DeprecatedServiceProperty("embedding")
    colbert = DeprecatedServiceProperty("colbert")
    colpali = DeprecatedServiceProperty("colpali")
    compression = DeprecatedServiceProperty("compression")
    document_intelligence = DeprecatedServiceProperty("document_intelligence")
    extraction = DeprecatedServiceProperty("extraction")
    ontology = DeprecatedServiceProperty("ontology")
    redaction = DeprecatedServiceProperty("redaction")

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """
        Initialize the Latence client.

        Args:
            api_key: API key for authentication. If not provided,
                reads from LATENCE_API_KEY environment variable.
            base_url: API base URL. Defaults to https://api.latence.ai
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self._client = BaseSyncClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Primary interface
        self.pipeline = Pipeline(self._client)

        # Supporting services
        self.jobs = Jobs(self._client)
        self.credits = Credits(self._client)

        # Experimental namespace (lazy-initialized)
        self._experimental: ExperimentalNamespace | None = None

    @property
    def experimental(self) -> ExperimentalNamespace:
        """Self-service access to individual services.

        Direct endpoint usage is billed at on-demand rates and is
        not covered by Enterprise SLAs.  For production workloads,
        use ``client.pipeline``.

        Example::

            result = client.experimental.extraction.extract(text="...")
        """
        if self._experimental is None:
            self._experimental = ExperimentalNamespace(self._client)
        return self._experimental

    @property
    def base_url(self) -> str:
        """The API base URL."""
        return self._client.base_url

    @property
    def max_retries(self) -> int:
        """Maximum number of retry attempts."""
        return self._client.max_retries

    def close(self) -> None:
        """Close the HTTP client and release resources."""
        self._client.close()

    def __enter__(self) -> Latence:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        key = self._client.api_key
        masked = f"{key[:7]}...{key[-3:]}" if key and len(key) > 10 else "***"
        return f"Latence(base_url={self._client.base_url!r}, api_key={masked!r})"


class AsyncLatence:
    """
    Asynchronous client for the Latence API.

    Async equivalent of :class:`Latence`.  All pipeline and service
    methods are ``async/await``.

    Example::

        >>> from latence import AsyncLatence
        >>> async with AsyncLatence(api_key="lat_xxx") as client:
        ...     job = await client.pipeline.run(files=["contract.pdf"])
        ...     pkg = await job.wait_for_completion()
        ...     print(pkg.document.markdown)
    """

    pipeline: AsyncPipeline
    """Primary interface: Data Intelligence Pipeline (async/job-based)."""

    jobs: AsyncJobs
    """Job management service."""

    credits: AsyncCredits
    """Credit balance service."""

    # Deprecated direct service access -- delegates to self.experimental
    embed = DeprecatedServiceProperty("embed")
    embedding = DeprecatedServiceProperty("embedding")
    colbert = DeprecatedServiceProperty("colbert")
    colpali = DeprecatedServiceProperty("colpali")
    compression = DeprecatedServiceProperty("compression")
    document_intelligence = DeprecatedServiceProperty("document_intelligence")
    extraction = DeprecatedServiceProperty("extraction")
    ontology = DeprecatedServiceProperty("ontology")
    redaction = DeprecatedServiceProperty("redaction")

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """
        Initialize the async Latence client.

        Args:
            api_key: API key for authentication. If not provided,
                reads from LATENCE_API_KEY environment variable.
            base_url: API base URL. Defaults to https://api.latence.ai
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self._client = BaseAsyncClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Primary interface
        self.pipeline = AsyncPipeline(self._client)

        # Supporting services
        self.jobs = AsyncJobs(self._client)
        self.credits = AsyncCredits(self._client)

        # Experimental namespace (lazy-initialized)
        self._experimental: AsyncExperimentalNamespace | None = None

    @property
    def experimental(self) -> AsyncExperimentalNamespace:
        """Self-service access to individual services (async).

        See :attr:`Latence.experimental` for full documentation.
        """
        if self._experimental is None:
            self._experimental = AsyncExperimentalNamespace(self._client)
        return self._experimental

    @property
    def base_url(self) -> str:
        """The API base URL."""
        return self._client.base_url

    @property
    def max_retries(self) -> int:
        """Maximum number of retry attempts."""
        return self._client.max_retries

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        await self._client.close()

    async def __aenter__(self) -> AsyncLatence:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        key = self._client.api_key
        masked = f"{key[:7]}...{key[-3:]}" if key and len(key) > 10 else "***"
        return f"AsyncLatence(base_url={self._client.base_url!r}, api_key={masked!r})"
