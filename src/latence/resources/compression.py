"""Compression service resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Union, overload

from .._models import CompressMessagesResponse, CompressResponse, JobSubmittedResponse, Message
from ._base import AsyncResource, SyncResource

if TYPE_CHECKING:
    from .._base import BaseAsyncClient, BaseSyncClient


class Compression(SyncResource):
    """
    Compression service - compress text while preserving meaning.

    Example:
        >>> result = client.compression.compress(text="Long text...", compression_rate=0.5)
        >>> print(f"Saved {result.tokens_saved} tokens")
    """

    def __init__(self, client: BaseSyncClient) -> None:
        super().__init__(client)

    @overload
    def compress(
        self,
        text: str,
        *,
        compression_rate: float = 0.5,
        force_preserve_digit: bool = True,
        force_tokens: list[str] | None = None,
        apply_toon: bool = False,
        chunk_size: int = 4096,
        fallback_mode: bool = True,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> CompressResponse: ...

    @overload
    def compress(
        self,
        text: str,
        *,
        compression_rate: float = 0.5,
        force_preserve_digit: bool = True,
        force_tokens: list[str] | None = None,
        apply_toon: bool = False,
        chunk_size: int = 4096,
        fallback_mode: bool = True,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def compress(
        self,
        text: str,
        *,
        compression_rate: float = 0.5,
        force_preserve_digit: bool = True,
        force_tokens: list[str] | None = None,
        apply_toon: bool = False,
        chunk_size: int = 4096,
        fallback_mode: bool = True,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[CompressResponse, JobSubmittedResponse]:
        """
        Compress text while preserving meaning.

        Args:
            text: Input text to compress
            compression_rate: Target compression (0.0-1.0, 0.5 = remove 50% of tokens)
            force_preserve_digit: Preserve numbers
            force_tokens: Tokens to always preserve
            apply_toon: Apply TOON encoding (adds $0.50/1M tokens)
            chunk_size: Chunk size (512-16384)
            fallback_mode: Enable fallback mode
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling

        Returns:
            CompressResponse with compressed text
        """
        body = self._build_request_body(
            text=text,
            compression_rate=compression_rate,
            force_preserve_digit=force_preserve_digit,
            force_tokens=force_tokens,
            apply_toon=apply_toon,
            chunk_size=chunk_size,
            fallback_mode=fallback_mode,
            request_id=request_id,
            return_job=return_job,
        )

        response = self._client.post("/api/v1/compression/compress", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = CompressResponse.model_validate(response.data)

        return self._inject_metadata(result, response)

    @overload
    def compress_messages(
        self,
        messages: list[dict[str, str]] | list[Message],
        *,
        target_compression: float = 0.5,
        max_compression: float | None = None,
        force_tokens: list[str] | None = None,
        force_preserve_digit: bool = True,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> CompressMessagesResponse: ...

    @overload
    def compress_messages(
        self,
        messages: list[dict[str, str]] | list[Message],
        *,
        target_compression: float = 0.5,
        max_compression: float | None = None,
        force_tokens: list[str] | None = None,
        force_preserve_digit: bool = True,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    def compress_messages(
        self,
        messages: list[dict[str, str]] | list[Message],
        *,
        target_compression: float = 0.5,
        max_compression: float | None = None,
        force_tokens: list[str] | None = None,
        force_preserve_digit: bool = True,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[CompressMessagesResponse, JobSubmittedResponse]:
        """
        Compress chat messages with gradient annealing.

        Recent messages are preserved more than older ones.

        Args:
            messages: List of {role, content} messages
            target_compression: Target compression rate
            max_compression: Maximum compression rate cap (optional)
            force_tokens: Tokens to preserve
            force_preserve_digit: Preserve numbers
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling

        Returns:
            CompressMessagesResponse with compressed messages
        """
        # Convert Message objects to dicts if needed
        msgs: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m, Message):
                msgs.append({"role": m.role, "content": m.content})
            else:
                msgs.append(m)

        body = self._build_request_body(
            messages=msgs,
            target_compression=target_compression,
            max_compression=max_compression,
            force_tokens=force_tokens,
            force_preserve_digit=force_preserve_digit,
            request_id=request_id,
            return_job=return_job,
        )

        response = self._client.post("/api/v1/compression/compress_messages", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = CompressMessagesResponse.model_validate(response.data)

        return self._inject_metadata(result, response)


class AsyncCompression(AsyncResource):
    """Async compression service."""

    def __init__(self, client: BaseAsyncClient) -> None:
        super().__init__(client)

    @overload
    async def compress(
        self,
        text: str,
        *,
        compression_rate: float = 0.5,
        force_preserve_digit: bool = True,
        force_tokens: list[str] | None = None,
        apply_toon: bool = False,
        chunk_size: int = 4096,
        fallback_mode: bool = True,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> CompressResponse: ...

    @overload
    async def compress(
        self,
        text: str,
        *,
        compression_rate: float = 0.5,
        force_preserve_digit: bool = True,
        force_tokens: list[str] | None = None,
        apply_toon: bool = False,
        chunk_size: int = 4096,
        fallback_mode: bool = True,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def compress(
        self,
        text: str,
        *,
        compression_rate: float = 0.5,
        force_preserve_digit: bool = True,
        force_tokens: list[str] | None = None,
        apply_toon: bool = False,
        chunk_size: int = 4096,
        fallback_mode: bool = True,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[CompressResponse, JobSubmittedResponse]:
        """
        Compress text while preserving meaning.

        Args:
            text: Input text to compress
            compression_rate: Target compression (0.0-1.0, 0.5 = remove 50% of tokens)
            force_preserve_digit: Preserve numbers
            force_tokens: Tokens to always preserve
            apply_toon: Apply TOON encoding (adds $0.50/1M tokens)
            chunk_size: Chunk size (512-16384)
            fallback_mode: Enable fallback mode
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling

        Returns:
            CompressResponse with compressed text
        """
        body = self._build_request_body(
            text=text,
            compression_rate=compression_rate,
            force_preserve_digit=force_preserve_digit,
            force_tokens=force_tokens,
            apply_toon=apply_toon,
            chunk_size=chunk_size,
            fallback_mode=fallback_mode,
            request_id=request_id,
            return_job=return_job,
        )

        response = await self._client.post("/api/v1/compression/compress", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = CompressResponse.model_validate(response.data)

        return self._inject_metadata(result, response)

    @overload
    async def compress_messages(
        self,
        messages: list[dict[str, str]] | list[Message],
        *,
        target_compression: float = 0.5,
        max_compression: float | None = None,
        force_tokens: list[str] | None = None,
        force_preserve_digit: bool = True,
        request_id: str | None = None,
        return_job: Literal[False] = False,
    ) -> CompressMessagesResponse: ...

    @overload
    async def compress_messages(
        self,
        messages: list[dict[str, str]] | list[Message],
        *,
        target_compression: float = 0.5,
        max_compression: float | None = None,
        force_tokens: list[str] | None = None,
        force_preserve_digit: bool = True,
        request_id: str | None = None,
        return_job: Literal[True],
    ) -> JobSubmittedResponse: ...

    async def compress_messages(
        self,
        messages: list[dict[str, str]] | list[Message],
        *,
        target_compression: float = 0.5,
        max_compression: float | None = None,
        force_tokens: list[str] | None = None,
        force_preserve_digit: bool = True,
        request_id: str | None = None,
        return_job: bool = False,
    ) -> Union[CompressMessagesResponse, JobSubmittedResponse]:
        """
        Compress chat messages with gradient annealing.

        Recent messages are preserved more than older ones.

        Args:
            messages: List of {role, content} messages
            target_compression: Target compression rate
            max_compression: Maximum compression rate cap (optional)
            force_tokens: Tokens to preserve
            force_preserve_digit: Preserve numbers
            request_id: Optional tracking ID
            return_job: If True, return job_id for polling

        Returns:
            CompressMessagesResponse with compressed messages
        """
        msgs: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m, Message):
                msgs.append({"role": m.role, "content": m.content})
            else:
                msgs.append(m)

        body = self._build_request_body(
            messages=msgs,
            target_compression=target_compression,
            max_compression=max_compression,
            force_tokens=force_tokens,
            force_preserve_digit=force_preserve_digit,
            request_id=request_id,
            return_job=return_job,
        )

        response = await self._client.post("/api/v1/compression/compress_messages", json=body)

        if return_job:
            result = JobSubmittedResponse.model_validate(response.data)
        else:
            result = CompressMessagesResponse.model_validate(response.data)

        return self._inject_metadata(result, response)
