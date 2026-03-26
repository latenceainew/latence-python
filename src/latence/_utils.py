"""Utility functions for the Latence SDK."""

from __future__ import annotations

import base64
import mimetypes
import struct
from pathlib import Path
from typing import BinaryIO, TypeVar, Callable, Awaitable, List, Union
import asyncio

T = TypeVar("T")
R = TypeVar("R")


async def process_batch_concurrently(
    items: List[T],
    processor: Callable[[T], Awaitable[R]],
    max_concurrency: int = 64
) -> List[Union[R, Exception]]:
    """
    Process a batch of items concurrently with a semaphore limit.

    Args:
        items: List of items to process.
        processor: Async function that takes an item and returns a result.
        max_concurrency: Maximum number of concurrent tasks (default 64).

    Returns:
        List of results or Exceptions, in the same order as items.
    """
    if not items:
        return []

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _sem_task(item: T) -> Union[R, Exception]:
        async with semaphore:
            try:
                return await processor(item)
            except Exception as e:
                return e

    tasks = [_sem_task(item) for item in items]
    return await asyncio.gather(*tasks)


def file_to_base64(file: str | Path | BinaryIO, *, filename: str | None = None) -> tuple[str, str]:
    """
    Convert a file to base64-encoded string.

    Args:
        file: File path, Path object, or file-like object
        filename: Optional filename (used for MIME type detection if file is file-like)

    Returns:
        Tuple of (base64_data, detected_filename)

    Example:
        >>> data, name = file_to_base64("/path/to/document.pdf")
        >>> result = client.document_intelligence.process(file_base64=data, filename=name)
    """
    if isinstance(file, (str, Path)):
        path = Path(file)
        with open(path, "rb") as f:
            data = f.read()
        detected_filename = path.name
    else:
        # File-like object
        data = file.read()
        if filename is not None:
            detected_filename = filename
        elif hasattr(file, "name"):
            detected_filename = Path(file.name).name
        else:
            detected_filename = "document"

    encoded = base64.b64encode(data).decode("utf-8")
    return encoded, detected_filename


def image_to_base64(
    image: str | Path | BinaryIO,
    *,
    include_data_uri: bool = True,
) -> str:
    """
    Convert an image to base64-encoded string for ColPali.

    Args:
        image: Image file path, Path object, or file-like object
        include_data_uri: If True, prepend data URI prefix (data:image/png;base64,)

    Returns:
        Base64-encoded image string

    Example:
        >>> image_data = image_to_base64("/path/to/page.png")
        >>> result = client.colpali.embed(image=image_data, is_query=False)
    """
    if isinstance(image, (str, Path)):
        path = Path(image)
        with open(path, "rb") as f:
            data = f.read()
        mime_type, _ = mimetypes.guess_type(str(path))
    else:
        data = image.read()
        name = getattr(image, "name", "image.png")
        mime_type, _ = mimetypes.guess_type(name)

    mime_type = mime_type or "image/png"
    encoded = base64.b64encode(data).decode("utf-8")

    if include_data_uri:
        return f"data:{mime_type};base64,{encoded}"
    return encoded


def decode_base64_embeddings(
    encoded: str,
    shape: list[int],
    dtype: str = "float32",
) -> list[list[float]]:
    """
    Decode base64-encoded embeddings to float arrays.

    Args:
        encoded: Base64-encoded embedding string
        shape: Shape of the embedding array [tokens/items, dimension]
        dtype: Data type ("float32" or "float16")

    Returns:
        List of embedding vectors

    Example:
        >>> result = client.colbert.embed(text="query", encoding_format="base64")
        >>> embeddings = decode_base64_embeddings(result.embeddings, result.shape)
        >>> print(len(embeddings), len(embeddings[0]))  # tokens, dimension
    """
    # Decode base64
    raw_bytes = base64.b64decode(encoded)

    # Determine format string based on dtype
    if dtype == "float32":
        fmt = "f"
        size = 4
    elif dtype == "float16":
        fmt = "e"
        size = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Use 'float32' or 'float16'")

    # Calculate expected size
    total_elements = 1
    for dim in shape:
        total_elements *= dim

    expected_bytes = total_elements * size
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"Size mismatch: expected {expected_bytes} bytes for shape {shape}, got {len(raw_bytes)}"
        )

    # Unpack all floats
    num_floats = len(raw_bytes) // size
    floats = list(struct.unpack(f"<{num_floats}{fmt}", raw_bytes))

    # Reshape to 2D array
    if len(shape) == 2:
        rows, cols = shape
        return [floats[i * cols : (i + 1) * cols] for i in range(rows)]
    elif len(shape) == 1:
        return [floats]
    else:
        raise ValueError(f"Unsupported shape: {shape}. Expected 1D or 2D.")


def encode_embeddings_base64(
    embeddings: list[list[float]],
    dtype: str = "float32",
) -> str:
    """
    Encode float embeddings to base64 string.

    Args:
        embeddings: List of embedding vectors
        dtype: Data type ("float32" or "float16")

    Returns:
        Base64-encoded string

    Example:
        >>> embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        >>> encoded = encode_embeddings_base64(embeddings)
    """
    if dtype == "float32":
        fmt = "f"
    elif dtype == "float16":
        fmt = "e"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Flatten embeddings
    flat = [val for row in embeddings for val in row]

    # Pack to bytes
    raw_bytes = struct.pack(f"<{len(flat)}{fmt}", *flat)

    return base64.b64encode(raw_bytes).decode("utf-8")
