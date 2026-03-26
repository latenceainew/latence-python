"""Deprecation utilities for the Latence SDK pivot.

Provides a descriptor that emits a DeprecationWarning on first access
to a deprecated service attribute (e.g. ``client.extraction``), guiding
users toward the pipeline-first interface.
"""

from __future__ import annotations

import warnings
from typing import Any


class DeprecatedServiceProperty:
    """Descriptor that emits a DeprecationWarning on first access per instance.

    Usage in a client class::

        class Latence:
            extraction = DeprecatedServiceProperty("extraction")
            # ...

    When ``client.extraction`` is accessed the first time, a warning is
    emitted pointing the user to ``client.pipeline`` or
    ``client.experimental.extraction``.  Subsequent accesses on the
    *same instance* return silently.

    The actual resource is fetched from ``instance.experimental.<attr_name>``.
    """

    def __init__(self, attr_name: str) -> None:
        self._attr_name = attr_name
        # Instance-level tracking key stored on the instance's __dict__
        self._warned_key = f"_deprecated_warned_{attr_name}"

    def __set_name__(self, owner: type, name: str) -> None:
        # Verify that the descriptor name matches the attr_name
        # (they should, but just in case)
        pass

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self  # Class-level access returns the descriptor

        # Emit warning on first access
        if not getattr(instance, self._warned_key, False):
            warnings.warn(
                f"client.{self._attr_name} is deprecated and not covered by SLA. "
                f"Use client.pipeline for production workloads, or "
                f"client.experimental.{self._attr_name} for self-service access. "
                f"See https://latence.ai/docs for migration guide.",
                DeprecationWarning,
                stacklevel=2,
            )
            object.__setattr__(instance, self._warned_key, True)

        # Delegate to the experimental namespace
        return getattr(instance.experimental, self._attr_name)
