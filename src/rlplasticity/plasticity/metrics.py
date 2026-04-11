"""Plasticity metrics used by update-level probes."""

from __future__ import annotations

from statistics import mean

from rlplasticity.core.base import BaseMetric
from rlplasticity.core.types import MetricResult, Snapshot


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _nonnull(values):
    return [value for value in values if value is not None]


def _history_values(snapshot: Snapshot, key: str) -> list[float]:
    history = snapshot.metadata.get("history", [])
    values = []
    for row in history:
        if not isinstance(row, dict):
            continue
        value = row.get(key)
        if isinstance(value, (float, int)):
            values.append(float(value))
    return values


def _history_group_values(snapshot: Snapshot, group: str, key: str) -> list[float]:
    history = snapshot.metadata.get("history", [])
    values = []
    for row in history:
        if not isinstance(row, dict):
            continue
        grouped = row.get(key)
        if not isinstance(grouped, dict):
            continue
        value = grouped.get(group)
        if isinstance(value, (float, int)):
            values.append(float(value))
    return values


def _delta_summary(name: str, values: list[float]) -> tuple[float, str, dict[str, float | int | None]]:
    if len(values) < 2:
        return 0.0, f"{name} trend unavailable without at least two history points.", {"points": len(values), "start": None, "end": None}
    delta = values[-1] - values[0]
    return delta, f"{name} trend delta={delta:.6f} across {len(values)} observations", {
        "points": len(values),
        "start": values[0],
        "end": values[-1],
    }


def _first_decline_summary(
    name: str,
    values: list[float],
    labels: list[str],
    *,
    min_decline: float,
) -> tuple[float, str, dict[str, float | int | str | None]]:
    if len(values) < 2:
        return 0.0, f"{name} decline onset unavailable without at least two history points.", {
            "points": len(values),
            "first_decline_index": None,
            "first_decline_label": None,
            "previous_label": None,
        }

    for index in range(1, len(values)):
        delta = values[index] - values[index - 1]
        if delta <= -min_decline:
            label = labels[index] if index < len(labels) else f"point-{index + 1}"
            previous_label = labels[index - 1] if index - 1 < len(labels) else f"point-{index}"
            return float(index), (
                f"{name} first decline appears near {label} (delta={delta:.6f} vs {previous_label})"
            ), {
                "points": len(values),
                "first_decline_index": index,
                "first_decline_label": label,
                "previous_label": previous_label,
                "delta": delta,
            }

    return 0.0, f"{name} decline onset not detected in the recorded history.", {
        "points": len(values),
        "first_decline_index": None,
        "first_decline_label": None,
        "previous_label": None,
        "delta": 0.0,
    }


def _history_labels(snapshot: Snapshot) -> list[str]:
    history = snapshot.metadata.get("history", [])
    labels: list[str] = []
    for index, row in enumerate(history):
        if isinstance(row, dict) and isinstance(row.get("label"), str):
            labels.append(row["label"])
        else:
            labels.append(f"point-{index + 1}")
    return labels


class PlasticityScoreMetric(BaseMetric):
    name = "plasticity_score"

    def __init__(
        self,
        *,
        min_relative_update: float = 1e-5,
        min_grad_to_weight_ratio: float = 1e-6,
    ) -> None:
        self.min_relative_update = min_relative_update
        self.min_grad_to_weight_ratio = min_grad_to_weight_ratio

    def compute(self, snapshot: Snapshot) -> MetricResult:
        update_values = _nonnull([layer.relative_update for layer in snapshot.layers])
        grad_values = _nonnull([layer.grad_to_weight_ratio for layer in snapshot.layers])
        activation_values = _nonnull([layer.zero_activation_fraction for layer in snapshot.layers])
        if not update_values and not grad_values:
            return MetricResult(self.name, 0.0, "Plasticity score unavailable without gradients or updates.")

        update_active = [1.0 if value >= self.min_relative_update else 0.0 for value in update_values]
        grad_active = [1.0 if value >= self.min_grad_to_weight_ratio else 0.0 for value in grad_values]
        activation_health = [max(0.0, 1.0 - value) for value in activation_values] or [0.0]
        score = (mean(update_active or [0.0]) + mean(grad_active or [0.0]) + mean(activation_health)) / 3.0
        summary = (
            f"Plasticity score={score:.3f} "
            f"(update-active={mean(update_active or [0.0]):.3f}, grad-active={mean(grad_active or [0.0]):.3f})"
        )
        return MetricResult(
            self.name,
            score,
            summary,
            metadata={
                "update_active_fraction": mean(update_active or [0.0]),
                "grad_active_fraction": mean(grad_active or [0.0]),
                "activation_health": mean(activation_health),
            },
        )


class StagnantLayerFractionMetric(BaseMetric):
    name = "stagnant_layer_fraction"

    def __init__(
        self,
        *,
        max_relative_update: float = 1e-6,
        max_grad_to_weight_ratio: float = 1e-7,
    ) -> None:
        self.max_relative_update = max_relative_update
        self.max_grad_to_weight_ratio = max_grad_to_weight_ratio

    def compute(self, snapshot: Snapshot) -> MetricResult:
        eligible = [
            layer
            for layer in snapshot.layers
            if layer.relative_update is not None and layer.grad_to_weight_ratio is not None
        ]
        stagnant = [
            layer
            for layer in eligible
            if layer.relative_update <= self.max_relative_update
            and layer.grad_to_weight_ratio <= self.max_grad_to_weight_ratio
        ]
        fraction = len(stagnant) / len(eligible) if eligible else 0.0
        names = [layer.name for layer in stagnant[:5]]
        summary = f"Stagnant layer fraction={fraction:.3f}"
        if names:
            summary += f" (examples: {', '.join(names)})"
        return MetricResult(
            self.name,
            fraction,
            summary,
            metadata={"stagnant_layers": [layer.name for layer in stagnant], "eligible_layers": len(eligible)},
        )


class GroupPlasticityMetric(BaseMetric):
    def __init__(
        self,
        group: str,
        *,
        min_relative_update: float = 1e-5,
        min_grad_to_weight_ratio: float = 1e-6,
    ) -> None:
        self.group = group
        self.name = f"{group}_plasticity_score"
        self.min_relative_update = min_relative_update
        self.min_grad_to_weight_ratio = min_grad_to_weight_ratio

    def compute(self, snapshot: Snapshot) -> MetricResult:
        layers = snapshot.by_group().get(self.group, [])
        if not layers:
            return MetricResult(self.name, 0.0, f"No {self.group} layers were captured.")

        update_values = _nonnull([layer.relative_update for layer in layers])
        grad_values = _nonnull([layer.grad_to_weight_ratio for layer in layers])
        if not update_values and not grad_values:
            return MetricResult(self.name, 0.0, f"{self.group} plasticity score unavailable in this probe mode.")

        update_active = [1.0 if value >= self.min_relative_update else 0.0 for value in update_values]
        grad_active = [1.0 if value >= self.min_grad_to_weight_ratio else 0.0 for value in grad_values]
        score = (mean(update_active or [0.0]) + mean(grad_active or [0.0])) / 2.0
        return MetricResult(
            self.name,
            score,
            f"{self.group} plasticity score={score:.3f}",
            metadata={
                "layer_count": len(layers),
                "mean_relative_update": _safe_mean(update_values),
                "mean_grad_to_weight_ratio": _safe_mean(grad_values),
            },
        )


class MeanActivationShiftMetric(BaseMetric):
    name = "mean_activation_shift"

    def compute(self, snapshot: Snapshot) -> MetricResult:
        shifts = _nonnull([layer.activation_shift for layer in snapshot.layers])
        value = _safe_mean([float(shift) for shift in shifts])
        return MetricResult(
            self.name,
            value,
            f"Mean activation shift={value:.6f}",
            metadata={"samples": len(shifts)},
        )


class PlasticityTrendDeltaMetric(BaseMetric):
    name = "plasticity_trend_delta"

    def compute(self, snapshot: Snapshot) -> MetricResult:
        values = _history_values(snapshot, "mean_relative_update")
        delta, summary, metadata = _delta_summary("Plasticity", values)
        return MetricResult(self.name, delta, summary, metadata=metadata)


class GroupPlasticityTrendDeltaMetric(BaseMetric):
    def __init__(self, group: str) -> None:
        self.group = group
        self.name = f"{group}_plasticity_trend_delta"

    def compute(self, snapshot: Snapshot) -> MetricResult:
        values = _history_group_values(snapshot, self.group, "group_relative_update")
        delta, summary, metadata = _delta_summary(f"{self.group} plasticity", values)
        return MetricResult(self.name, delta, summary, metadata=metadata)


class PlasticityFirstDeclineMetric(BaseMetric):
    name = "plasticity_first_decline"

    def __init__(self, *, min_decline: float = 1e-4) -> None:
        self.min_decline = min_decline

    def compute(self, snapshot: Snapshot) -> MetricResult:
        values = _history_values(snapshot, "mean_relative_update")
        labels = _history_labels(snapshot)
        value, summary, metadata = _first_decline_summary(
            "Plasticity",
            values,
            labels,
            min_decline=self.min_decline,
        )
        return MetricResult(self.name, value, summary, metadata=metadata)


class GroupPlasticityFirstDeclineMetric(BaseMetric):
    def __init__(self, group: str, *, min_decline: float = 1e-4) -> None:
        self.group = group
        self.min_decline = min_decline
        self.name = f"{group}_plasticity_first_decline"

    def compute(self, snapshot: Snapshot) -> MetricResult:
        values = _history_group_values(snapshot, self.group, "group_relative_update")
        labels = _history_labels(snapshot)
        value, summary, metadata = _first_decline_summary(
            f"{self.group} plasticity",
            values,
            labels,
            min_decline=self.min_decline,
        )
        return MetricResult(self.name, value, summary, metadata=metadata)
