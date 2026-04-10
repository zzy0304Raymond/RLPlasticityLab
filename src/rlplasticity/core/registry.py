"""Simple registries for extensible analysis modules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar


T = TypeVar("T")


class FactoryRegistry(Generic[T]):
    """Stores named factories for analysis modules."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], T]] = {}

    def register(self, name: str, factory: Callable[[], T], *, overwrite: bool = False) -> None:
        if name in self._factories and not overwrite:
            raise ValueError(f"Factory '{name}' is already registered.")
        self._factories[name] = factory

    def create(self, name: str) -> T:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "<empty>"
            raise KeyError(f"Unknown factory '{name}'. Available: {available}") from exc
        return factory()

    def names(self) -> list[str]:
        return sorted(self._factories)
