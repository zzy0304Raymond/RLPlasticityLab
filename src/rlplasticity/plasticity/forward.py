"""Forward-only probe metrics and findings."""

from __future__ import annotations

from statistics import mean

from rlplasticity.core.base import BaseDiagnosticRule, BaseMetric
from rlplasticity.core.types import DiagnosticFinding, MetricResult, Snapshot


def _nonnull(values):
    return [value for value in values if value is not None]


class ActivationHealthScoreMetric(BaseMetric):
    name = "activation_health_score"

    def compute(self, snapshot: Snapshot) -> MetricResult:
        zero_fractions = _nonnull([layer.zero_activation_fraction for layer in snapshot.layers])
        if not zero_fractions:
            return MetricResult(self.name, 0.0, "Activation health unavailable; no activation stats were captured.")
        health = [1.0 - value for value in zero_fractions]
        score = mean(health)
        return MetricResult(
            self.name,
            score,
            f"Activation health score={score:.3f}",
            metadata={"mean_zero_activation_fraction": mean(zero_fractions)},
        )


class LowVariationLayerFractionMetric(BaseMetric):
    name = "low_variation_layer_fraction"

    def __init__(self, *, max_activation_std: float = 1e-3) -> None:
        self.max_activation_std = max_activation_std

    def compute(self, snapshot: Snapshot) -> MetricResult:
        stds = [(layer.name, layer.activation_std) for layer in snapshot.layers if layer.activation_std is not None]
        low = [name for name, std in stds if std is not None and std <= self.max_activation_std]
        fraction = len(low) / len(stds) if stds else 0.0
        return MetricResult(
            self.name,
            fraction,
            f"Low-variation layer fraction={fraction:.3f}",
            metadata={"low_variation_layers": low},
        )


class GroupActivationHealthMetric(BaseMetric):
    def __init__(self, group: str) -> None:
        self.group = group
        self.name = f"{group}_activation_health"

    def compute(self, snapshot: Snapshot) -> MetricResult:
        layers = snapshot.by_group().get(self.group, [])
        values = _nonnull([layer.zero_activation_fraction for layer in layers])
        if not values:
            return MetricResult(self.name, 0.0, f"No {self.group} activation stats were captured.")
        score = mean([1.0 - value for value in values])
        return MetricResult(
            self.name,
            score,
            f"{self.group} activation health={score:.3f}",
            metadata={"layer_count": len(values)},
        )


class WidespreadInactiveActivationRule(BaseDiagnosticRule):
    name = "widespread_inactive_activations"

    def __init__(self, *, max_activation_health: float = 0.25) -> None:
        self.max_activation_health = max_activation_health

    def evaluate(self, snapshot: Snapshot, metrics: dict[str, MetricResult]) -> DiagnosticFinding | None:
        score = metrics.get("activation_health_score")
        if score is None:
            return None
        if score.value <= self.max_activation_health:
            return DiagnosticFinding(
                name=self.name,
                severity="medium",
                summary="Large parts of the model show weak or near-zero activations on the supplied sample.",
                evidence=[score.summary, f"captured_layers={len(snapshot.layers)}"],
                recommendations=[
                    "Verify that the sample batch matches the model's expected observation preprocessing.",
                    "Use a dynamic plasticity probe before concluding this is a true optimization bottleneck.",
                ],
                confidence="medium",
            )
        return None


class EncoderLowResponseRule(BaseDiagnosticRule):
    name = "encoder_low_response"

    def __init__(self, *, min_gap: float = 0.25) -> None:
        self.min_gap = min_gap

    def evaluate(self, snapshot: Snapshot, metrics: dict[str, MetricResult]) -> DiagnosticFinding | None:
        if "encoder" not in snapshot.by_group():
            return None
        encoder = metrics.get("encoder_activation_health")
        trunk = metrics.get("trunk_activation_health")
        policy = metrics.get("policy_activation_health")
        reference_values = [metric.value for metric in [trunk, policy] if metric is not None]
        if encoder is None or not reference_values:
            return None
        reference = max(reference_values)
        if reference - encoder.value >= self.min_gap:
            return DiagnosticFinding(
                name=self.name,
                severity="low",
                summary="The encoder responds less strongly than downstream layers on this forward probe.",
                evidence=[encoder.summary, f"reference_downstream_activation_health={reference:.3f}"],
                recommendations=[
                    "Try a plasticity probe with real replay samples to check whether this weak response persists under gradient updates.",
                    "Inspect observation normalization and encoder initialization if this pattern repeats across batches.",
                ],
                confidence="low",
            )
        return None
