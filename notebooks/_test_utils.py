"""
Shared test utilities for Latence AI SDK notebooks.

This module provides helpers for:
- Client initialization
- Response inspection and debugging
- File loading for test data
- Performance timing and comparison

Usage:
    from _test_utils import get_client, inspect_response, load_test_file
"""

from __future__ import annotations

import base64
import os
import sys
import time
from dataclasses import dataclass  # noqa: F401 - used by TimingResult
from pathlib import Path
from typing import Any, Callable

# Ensure latence is importable (for local development)
SDK_ROOT = Path(__file__).parent.parent / "src"
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from latence import (
    AsyncLatence,
    Latence,
    LatenceError,
)


# =============================================================================
# Configuration
# =============================================================================

# Default API key from environment
DEFAULT_API_KEY = os.getenv("LATENCE_API_KEY", "")

# Default base URL (local wrangler dev server)
DEFAULT_BASE_URL = "https://api.latence.ai"

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data" / "test_files"


# =============================================================================
# Client Initialization
# =============================================================================

def get_client(
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Latence:
    """
    Get a configured Latence client for testing.
    
    Args:
        api_key: API key (defaults to LATENCE_API_KEY env var)
        base_url: API base URL (defaults to local dev)
        **kwargs: Additional arguments passed to Latence()
    
    Returns:
        Configured Latence client
    
    Example:
        >>> client = get_client()
        >>> result = client.embedding.embed(text="Hello")
    """
    key = api_key or DEFAULT_API_KEY
    if not key:
        raise ValueError(
            "API key required. Set LATENCE_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    return Latence(
        api_key=key,
        base_url=base_url or DEFAULT_BASE_URL,
        **kwargs,
    )


def get_async_client(
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> AsyncLatence:
    """
    Get a configured async Latence client for testing.
    
    Args:
        api_key: API key (defaults to LATENCE_API_KEY env var)
        base_url: API base URL (defaults to local dev)
        **kwargs: Additional arguments passed to AsyncLatence()
    
    Returns:
        Configured AsyncLatence client
    """
    key = api_key or DEFAULT_API_KEY
    if not key:
        raise ValueError(
            "API key required. Set LATENCE_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    return AsyncLatence(
        api_key=key,
        base_url=base_url or DEFAULT_BASE_URL,
        **kwargs,
    )


# =============================================================================
# Response Inspection
# =============================================================================

def inspect_response(response: Any, show_data: bool = True) -> dict[str, Any]:
    """
    Inspect a response object and extract useful debugging info.
    
    Args:
        response: Any response object from the SDK
        show_data: Whether to include the full data in output
    
    Returns:
        Dictionary with response metadata
    """
    info: dict[str, Any] = {
        "type": type(response).__name__,
    }
    
    # Extract common fields
    if hasattr(response, "request_id"):
        info["request_id"] = response.request_id
    
    if hasattr(response, "credits_used"):
        info["credits_used"] = response.credits_used
    
    if hasattr(response, "usage"):
        info["usage"] = response.usage
    
    if hasattr(response, "model"):
        info["model"] = response.model
    
    # Response-specific fields
    if hasattr(response, "embeddings"):
        emb = response.embeddings
        if isinstance(emb, list):
            info["embeddings_shape"] = f"{len(emb)} x {len(emb[0]) if emb else 0}"
    
    if hasattr(response, "shape"):
        info["shape"] = response.shape
    
    if hasattr(response, "entities"):
        info["entity_count"] = len(response.entities) if response.entities else 0
    
    if hasattr(response, "pages_processed"):
        info["pages_processed"] = response.pages_processed
    
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            info["content_length"] = len(content)
            info["content_preview"] = content[:200] + "..." if len(content) > 200 else content
    
    if show_data:
        # Try to get model_dump if available (Pydantic)
        if hasattr(response, "model_dump"):
            info["data"] = response.model_dump()
        elif hasattr(response, "__dict__"):
            info["data"] = {k: v for k, v in response.__dict__.items() if not k.startswith("_")}
    
    return info


def print_response(response: Any, title: str = "Response") -> None:
    """Pretty-print a response for notebook display."""
    import json
    
    info = inspect_response(response, show_data=False)
    
    print(f"\n{'='*60}")
    print(f"📦 {title}")
    print(f"{'='*60}")
    
    for key, value in info.items():
        if key == "content_preview":
            print(f"  {key}:")
            print(f"    {value}")
        else:
            print(f"  {key}: {value}")
    
    print(f"{'='*60}\n")


# =============================================================================
# Performance Timing
# =============================================================================

@dataclass
class TimingResult:
    """Result of a timed operation."""
    
    name: str
    duration: float
    success: bool
    error: str | None = None
    result: Any = None


def time_operation(
    name: str,
    operation: Callable[[], Any],
    catch_errors: bool = True,
) -> TimingResult:
    """
    Time an operation and return the result with timing info.
    
    Args:
        name: Name of the operation (for display)
        operation: Callable to execute
        catch_errors: Whether to catch and record errors
    
    Returns:
        TimingResult with duration and result
    
    Example:
        >>> result = time_operation("embed", lambda: client.embedding.embed(text="test"))
        >>> print(f"Took {result.duration:.2f}s")
    """
    start = time.time()
    try:
        result = operation()
        duration = time.time() - start
        return TimingResult(name=name, duration=duration, success=True, result=result)
    except Exception as e:
        duration = time.time() - start
        if catch_errors:
            return TimingResult(name=name, duration=duration, success=False, error=str(e))
        raise


def compare_timings(results: list[TimingResult]) -> None:
    """Print a comparison table of timing results."""
    print("\n📊 Timing Comparison")
    print("-" * 50)
    print(f"{'Operation':<25} {'Duration':>10} {'Status':>10}")
    print("-" * 50)
    
    for r in results:
        status = "✅" if r.success else "❌"
        print(f"{r.name:<25} {r.duration:>9.2f}s {status:>10}")
    
    total = sum(r.duration for r in results)
    print("-" * 50)
    print(f"{'Total':<25} {total:>9.2f}s")
    print()


# =============================================================================
# File Loading Utilities
# =============================================================================

def load_test_file(filename: str) -> bytes:
    """
    Load a test file from the data/test_files directory.
    
    Args:
        filename: Name of the file to load
    
    Returns:
        File contents as bytes
    
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    filepath = TEST_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Test file not found: {filepath}\n"
            f"Available files: {list(TEST_DATA_DIR.glob('*')) if TEST_DATA_DIR.exists() else []}"
        )
    return filepath.read_bytes()


def load_test_file_base64(filename: str) -> str:
    """
    Load a test file and return as base64 string.
    
    Args:
        filename: Name of the file to load
    
    Returns:
        Base64-encoded file contents
    """
    return base64.b64encode(load_test_file(filename)).decode("utf-8")


def list_test_files() -> list[str]:
    """List available test files."""
    if not TEST_DATA_DIR.exists():
        return []
    return [f.name for f in TEST_DATA_DIR.iterdir() if f.is_file()]


def create_sample_text_file(filename: str, content: str) -> Path:
    """Create a sample text file for testing."""
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = TEST_DATA_DIR / filename
    filepath.write_text(content)
    return filepath


# =============================================================================
# Error Testing Utilities
# =============================================================================

def expect_error(
    operation: Callable[[], Any],
    error_type: type[Exception] = LatenceError,
    message_contains: str | None = None,
) -> dict[str, Any]:
    """
    Execute an operation expecting it to raise an error.
    
    Args:
        operation: Callable that should raise an error
        error_type: Expected exception type
        message_contains: Optional substring to check in error message
    
    Returns:
        Dictionary with error details
    
    Raises:
        AssertionError: If no error or wrong error type
    
    Example:
        >>> result = expect_error(
        ...     lambda: client.embedding.embed(text=""),
        ...     error_type=ValidationError,
        ...     message_contains="empty"
        ... )
        >>> print(f"Got expected error: {result['message']}")
    """
    try:
        result = operation()
        raise AssertionError(f"Expected {error_type.__name__} but operation succeeded with: {result}")
    except error_type as e:
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
        }
        
        # Extract additional error attributes
        if hasattr(e, "status_code"):
            error_info["status_code"] = e.status_code
        if hasattr(e, "request_id"):
            error_info["request_id"] = e.request_id
        
        if message_contains and message_contains not in str(e):
            raise AssertionError(
                f"Error message doesn't contain '{message_contains}': {str(e)}"
            )
        
        print(f"✅ Got expected error: {error_info['type']}")
        return error_info
    except Exception as e:
        raise AssertionError(f"Expected {error_type.__name__} but got {type(e).__name__}: {e}")


# =============================================================================
# Display Utilities
# =============================================================================

def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'#'*60}")
    print(f"# {title}")
    print(f"{'#'*60}\n")


def print_subsection(title: str) -> None:
    """Print a subsection header."""
    print(f"\n## {title}")
    print("-" * 40)


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✅ {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"❌ {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"ℹ️  {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠️  {message}")


# =============================================================================
# Notebook Setup
# =============================================================================

def setup_notebook(
    api_key: str | None = None,
) -> Latence:
    """
    Standard notebook setup: create client.
    
    Args:
        api_key: API key (defaults to LATENCE_API_KEY env var)
    
    Returns:
        Configured Latence client
    
    Example:
        >>> # At the start of each notebook:
        >>> client = setup_notebook()
    """
    print_section("Latence AI SDK - Notebook Setup")
    
    client = get_client(api_key=api_key)
    print_success(f"Client initialized (base_url: {client.base_url})")
    
    return client


# =============================================================================
# Export all utilities
# =============================================================================

__all__ = [
    # Client
    "get_client",
    "get_async_client",
    "setup_notebook",
    # Response inspection
    "inspect_response",
    "print_response",
    # Timing
    "TimingResult",
    "time_operation",
    "compare_timings",
    # File loading
    "load_test_file",
    "load_test_file_base64",
    "list_test_files",
    "create_sample_text_file",
    # Error testing
    "expect_error",
    # Display
    "print_section",
    "print_subsection",
    "print_success",
    "print_error",
    "print_info",
    "print_warning",
    # Constants
    "DEFAULT_API_KEY",
    "DEFAULT_BASE_URL",
    "TEST_DATA_DIR",
]
