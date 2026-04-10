"""Checkpoint loading and static summarization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlplasticity.core.enums import AnalysisKind, EvidenceLevel
from rlplasticity.core.naming import infer_module_group
from rlplasticity.core.types import LayerSnapshot, Snapshot


def _ensure_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for loading .pt checkpoints from disk."
        ) from exc
    return torch


def load_checkpoint(path: str, *, map_location: str = "cpu") -> Any:
    """Load a checkpoint from disk without touching GPU memory."""

    torch = _ensure_torch()
    return torch.load(path, map_location=map_location)


def extract_state_dict(payload: Any) -> Mapping[str, Any]:
    """Extract the most likely state_dict mapping from a checkpoint payload."""

    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint payload must be a mapping or a state_dict-like object.")

    preferred_keys = (
        "state_dict",
        "model_state_dict",
        "model",
        "actor_state_dict",
        "actor",
        "policy_state_dict",
        "policy",
    )
    for key in preferred_keys:
        maybe = payload.get(key)
        if isinstance(maybe, Mapping):
            return maybe
    return payload


def _flatten_numeric(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric(item))
        return flattened

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "reshape"):
        try:
            value = value.reshape(-1)
        except Exception:
            pass
    if hasattr(value, "tolist"):
        return _flatten_numeric(value.tolist())
    return []


def _shape_of(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    try:
        return tuple(int(item) for item in shape)
    except TypeError:
        return ()


def summarize_state_dict(
    state_dict: Mapping[str, Any],
    *,
    source: str = "state_dict",
    group_keywords: dict[str, list[str]] | None = None,
) -> Snapshot:
    """Turn a state_dict into a static structural snapshot."""

    layers: list[LayerSnapshot] = []
    for name, value in state_dict.items():
        flat = _flatten_numeric(value)
        if not flat:
            continue

        abs_values = [abs(item) for item in flat]
        parameter_count = len(flat)
        parameter_norm = sum(item * item for item in flat) ** 0.5
        parameter_mean_abs = sum(abs_values) / parameter_count
        zero_fraction = sum(1 for item in abs_values if item < 1e-12) / parameter_count
        parameter_max_abs = max(abs_values) if abs_values else 0.0
        group = infer_module_group(name, group_keywords)
        layers.append(
            LayerSnapshot(
                name=name,
                group=group,
                module_type="Parameter",
                parameter_count=parameter_count,
                parameter_norm=parameter_norm,
                parameter_mean_abs=parameter_mean_abs,
                parameter_zero_fraction=zero_fraction,
                parameter_max_abs=parameter_max_abs,
                metadata={"shape": _shape_of(value), "source": source},
            )
        )

    caveats = [
        "This is a static checkpoint scan. It cannot measure gradient flow, update effectiveness, or true plasticity loss.",
    ]
    return Snapshot(
        kind=AnalysisKind.CHECKPOINT_SCAN,
        evidence_level=EvidenceLevel.STATIC,
        step=None,
        loss=None,
        layers=layers,
        metadata={"source": source, "parameter_entries": len(layers)},
        caveats=caveats,
    )
