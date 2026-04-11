"""Helpers for aggregating multiple snapshots into one reportable view."""

from __future__ import annotations

from statistics import mean

from .enums import EvidenceLevel
from .types import LayerSnapshot, Snapshot


NUMERIC_FIELDS = (
    "parameter_count",
    "parameter_norm",
    "parameter_mean_abs",
    "parameter_zero_fraction",
    "parameter_max_abs",
    "gradient_norm",
    "update_norm",
    "relative_update",
    "grad_to_weight_ratio",
    "activation_mean",
    "activation_std",
    "activation_shift",
    "zero_activation_fraction",
    "max_activation_abs",
)


def _average_field(layers: list[LayerSnapshot], field_name: str):
    values = [getattr(layer, field_name) for layer in layers if getattr(layer, field_name) is not None]
    if not values:
        return None
    if field_name == "parameter_count":
        return int(round(mean(values)))
    return mean(values)


def _nonnull(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def _mean_or_none(values: list[float | None]) -> float | None:
    concrete = _nonnull(values)
    if not concrete:
        return None
    return mean(concrete)


def build_history_entry(snapshot: Snapshot, *, label: str | None = None) -> dict[str, object]:
    """Summarize one snapshot into a compact history row for trend analysis."""

    grouped = snapshot.by_group()
    group_relative_update = {
        group: _mean_or_none([layer.relative_update for layer in layers])
        for group, layers in grouped.items()
    }
    group_grad_ratio = {
        group: _mean_or_none([layer.grad_to_weight_ratio for layer in layers])
        for group, layers in grouped.items()
    }
    relative_updates = _nonnull([layer.relative_update for layer in snapshot.layers])
    grad_ratios = _nonnull([layer.grad_to_weight_ratio for layer in snapshot.layers])

    return {
        "label": label or f"step-{snapshot.step if snapshot.step is not None else 'unknown'}",
        "step": snapshot.step,
        "loss": snapshot.loss,
        "mean_relative_update": _mean_or_none(relative_updates),
        "mean_grad_to_weight_ratio": _mean_or_none(grad_ratios),
        "group_relative_update": group_relative_update,
        "group_grad_to_weight_ratio": group_grad_ratio,
        "layer_count": len(snapshot.layers),
    }


def aggregate_snapshots(
    snapshots: list[Snapshot],
    *,
    history_label_prefix: str = "step",
    metadata_updates: dict[str, object] | None = None,
    caveat: str | None = None,
) -> Snapshot:
    """Average a short window of snapshots into one stable summary."""

    if not snapshots:
        raise ValueError("Cannot aggregate an empty snapshot list.")
    if len(snapshots) == 1:
        return snapshots[0]

    grouped: dict[str, list[LayerSnapshot]] = {}
    for snapshot in snapshots:
        for layer in snapshot.layers:
            grouped.setdefault(layer.name, []).append(layer)

    layers: list[LayerSnapshot] = []
    for name, matching_layers in grouped.items():
        first = matching_layers[0]
        averaged = {
            field_name: _average_field(matching_layers, field_name)
            for field_name in NUMERIC_FIELDS
        }
        layers.append(
            LayerSnapshot(
                name=name,
                group=first.group,
                module_type=first.module_type,
                parameter_count=averaged["parameter_count"] or 0,
                parameter_norm=averaged["parameter_norm"] or 0.0,
                parameter_mean_abs=averaged["parameter_mean_abs"] or 0.0,
                parameter_zero_fraction=averaged["parameter_zero_fraction"],
                parameter_max_abs=averaged["parameter_max_abs"],
                gradient_norm=averaged["gradient_norm"],
                update_norm=averaged["update_norm"],
                relative_update=averaged["relative_update"],
                grad_to_weight_ratio=averaged["grad_to_weight_ratio"],
                activation_mean=averaged["activation_mean"],
                activation_std=averaged["activation_std"],
                activation_shift=averaged["activation_shift"],
                zero_activation_fraction=averaged["zero_activation_fraction"],
                max_activation_abs=averaged["max_activation_abs"],
                metadata=dict(first.metadata),
            )
        )

    base = snapshots[0]
    metadata = dict(base.metadata)
    metadata.update(metadata_updates or {})
    metadata["window_size"] = len(snapshots)
    metadata["loss_mean"] = mean(
        snapshot.loss for snapshot in snapshots if snapshot.loss is not None
    ) if any(snapshot.loss is not None for snapshot in snapshots) else None
    metadata["history"] = [
        build_history_entry(snapshot, label=f"{history_label_prefix}-{index + 1}")
        for index, snapshot in enumerate(snapshots)
    ]

    caveats = list(base.caveats)
    caveats.append(
        caveat
        or f"This report averages {len(snapshots)} probe steps; inspect per-step results if the signal looks unstable."
    )
    return Snapshot(
        kind=base.kind,
        evidence_level=EvidenceLevel.WINDOW,
        step=base.step,
        loss=metadata["loss_mean"],
        layers=layers,
        metadata=metadata,
        caveats=caveats,
    )
