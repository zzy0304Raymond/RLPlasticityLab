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


def aggregate_snapshots(snapshots: list[Snapshot]) -> Snapshot:
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
    metadata["window_size"] = len(snapshots)
    metadata["loss_mean"] = mean(
        snapshot.loss for snapshot in snapshots if snapshot.loss is not None
    ) if any(snapshot.loss is not None for snapshot in snapshots) else None

    caveats = list(base.caveats)
    caveats.append(
        f"This report averages {len(snapshots)} probe steps; inspect per-step results if the signal looks unstable."
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
