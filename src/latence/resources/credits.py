"""Credits service resource for balance checking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._models import CreditsResponse
from ._base import AsyncResource, SyncResource

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient


class Credits(SyncResource):
    """
    Credits service - check account balance.

    Example:
        >>> balance = client.credits.balance()
        >>> print(f"Remaining balance: ${balance.balance_usd:.2f}")
    """

    def __init__(self, client: BaseSyncClient) -> None:
        super().__init__(client)

    def balance(self) -> CreditsResponse:
        """
        Get current USD balance.

        Returns:
            CreditsResponse with balance_usd
        """
        response = self._client.get("/api/v1/credits")
        result = CreditsResponse.model_validate(response.data)
        return self._inject_metadata(result, response)


class AsyncCredits(AsyncResource):
    """Async Credits service."""

    def __init__(self, client: BaseAsyncClient) -> None:
        super().__init__(client)

    async def balance(self) -> CreditsResponse:
        """
        Get current USD balance.

        Returns:
            CreditsResponse with balance_usd
        """
        response = await self._client.get("/api/v1/credits")
        result = CreditsResponse.model_validate(response.data)
        return self._inject_metadata(result, response)
