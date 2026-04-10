"""Small shared helpers for probe workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rlplasticity.ingest.checkpoints import extract_state_dict, load_checkpoint


def maybe_load_model_checkpoint(
    model,
    checkpoint: str | Mapping[str, Any] | None,
    *,
    map_location: str = "cpu",
) -> None:
    """Load checkpoint weights into a model when a checkpoint is provided."""

    if checkpoint is None:
        return
    payload = load_checkpoint(checkpoint, map_location=map_location) if isinstance(checkpoint, str) else checkpoint
    state_dict = extract_state_dict(payload)
    model.load_state_dict(state_dict)


def normalize_batches(batches: Any) -> list[Any]:
    """Turn a single batch or an iterable of batches into a concrete list."""

    if isinstance(batches, (str, bytes, bytearray)):
        return [batches]
    if isinstance(batches, Mapping):
        return [batches]
    if isinstance(batches, Iterable):
        return list(batches)
    return [batches]
