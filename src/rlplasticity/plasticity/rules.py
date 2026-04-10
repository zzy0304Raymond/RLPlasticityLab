"""Diagnostic rules for plasticity analysis."""

from __future__ import annotations

from rlplasticity.core.base import BaseDiagnosticRule
from rlplasticity.core.types import DiagnosticFinding, MetricResult, Snapshot


def _metric_value(metrics: dict[str, MetricResult], key: str) -> float | None:
    metric = metrics.get(key)
    return None if metric is None else metric.value


class GlobalPlasticityStallRule(BaseDiagnosticRule):
    name = "global_plasticity_stall"

    def __init__(
        self,
        *,
        max_plasticity_score: float = 0.45,
        min_stagnant_fraction: float = 0.5,
    ) -> None:
        self.max_plasticity_score = max_plasticity_score
        self.min_stagnant_fraction = min_stagnant_fraction

    def evaluate(
        self,
        snapshot: Snapshot,
        metrics: dict[str, MetricResult],
    ) -> DiagnosticFinding | None:
        plasticity_score = _metric_value(metrics, "plasticity_score")
        stagnant_fraction = _metric_value(metrics, "stagnant_layer_fraction")
        if plasticity_score is None or stagnant_fraction is None:
            return None

        if plasticity_score <= self.max_plasticity_score and stagnant_fraction >= self.min_stagnant_fraction:
            return DiagnosticFinding(
                name=self.name,
                severity="high",
                summary="The model looks globally plasticity-limited.",
                evidence=[
                    f"plasticity_score={plasticity_score:.3f}",
                    f"stagnant_layer_fraction={stagnant_fraction:.3f}",
                    f"captured_layers={len(snapshot.layers)}",
                ],
                recommendations=[
                    "Check whether learning rates, normalization, or optimizer settings are suppressing effective updates.",
                    "Compare early-training and current checkpoints to verify whether representational change has slowed down.",
                    "Inspect reward scaling before concluding the issue is purely architectural.",
                ],
            )
        return None


class EncoderBottleneckRule(BaseDiagnosticRule):
    name = "encoder_bottleneck"

    def __init__(self, *, min_gap: float = 0.25) -> None:
        self.min_gap = min_gap

    def evaluate(
        self,
        snapshot: Snapshot,
        metrics: dict[str, MetricResult],
    ) -> DiagnosticFinding | None:
        if "encoder" not in snapshot.by_group():
            return None

        encoder_score = _metric_value(metrics, "encoder_plasticity_score")
        trunk_score = _metric_value(metrics, "trunk_plasticity_score")
        policy_score = _metric_value(metrics, "policy_plasticity_score")
        reference = max(value for value in [trunk_score, policy_score] if value is not None) if any(
            value is not None for value in [trunk_score, policy_score]
        ) else None

        if encoder_score is None or reference is None:
            return None

        if reference - encoder_score >= self.min_gap:
            return DiagnosticFinding(
                name=self.name,
                severity="medium",
                summary="The encoder is adapting less than downstream layers.",
                evidence=[
                    f"encoder_plasticity_score={encoder_score:.3f}",
                    f"reference_downstream_score={reference:.3f}",
                ],
                recommendations=[
                    "Inspect encoder-specific gradients and update magnitudes across several consecutive steps.",
                    "Check whether observations or augmentations are causing feature collapse or over-stable representations.",
                    "Consider lower encoder regularization or a separate optimizer schedule for the encoder.",
                ],
            )
        return None


class HeadSaturationRule(BaseDiagnosticRule):
    name = "head_saturation"

    def __init__(self, *, min_gap: float = 0.25) -> None:
        self.min_gap = min_gap

    def evaluate(
        self,
        snapshot: Snapshot,
        metrics: dict[str, MetricResult],
    ) -> DiagnosticFinding | None:
        groups = snapshot.by_group()
        trunk_score = _metric_value(metrics, "trunk_plasticity_score")
        policy_score = _metric_value(metrics, "policy_plasticity_score")
        value_score = _metric_value(metrics, "value_plasticity_score")
        if trunk_score is None:
            return None

        weak_heads = []
        if "policy" in groups and policy_score is not None and trunk_score - policy_score >= self.min_gap:
            weak_heads.append(("policy", policy_score))
        if "value" in groups and value_score is not None and trunk_score - value_score >= self.min_gap:
            weak_heads.append(("value", value_score))

        if weak_heads:
            evidence = [f"trunk_plasticity_score={trunk_score:.3f}"]
            evidence.extend(f"{name}_plasticity_score={score:.3f}" for name, score in weak_heads)
            return DiagnosticFinding(
                name=self.name,
                severity="medium",
                summary="The output heads are adapting less than the trunk.",
                evidence=evidence,
                recommendations=[
                    "Inspect whether policy/value logits are saturating or whether loss scaling is suppressing head updates.",
                    "Check action distribution entropy or critic target magnitudes alongside this report.",
                    "Consider per-head learning-rate tuning or initialization changes.",
                ],
            )
        return None


class RepresentationChurnRule(BaseDiagnosticRule):
    name = "representation_churn"

    def __init__(
        self,
        *,
        min_activation_shift: float = 0.25,
        min_plasticity_score: float = 0.55,
    ) -> None:
        self.min_activation_shift = min_activation_shift
        self.min_plasticity_score = min_plasticity_score

    def evaluate(
        self,
        snapshot: Snapshot,
        metrics: dict[str, MetricResult],
    ) -> DiagnosticFinding | None:
        activation_shift = _metric_value(metrics, "mean_activation_shift")
        plasticity_score = _metric_value(metrics, "plasticity_score")
        if activation_shift is None or plasticity_score is None:
            return None

        if activation_shift >= self.min_activation_shift and plasticity_score >= self.min_plasticity_score:
            return DiagnosticFinding(
                name=self.name,
                severity="low",
                summary="Representations are changing quickly; instability may not be a plasticity loss issue.",
                evidence=[
                    f"mean_activation_shift={activation_shift:.6f}",
                    f"plasticity_score={plasticity_score:.3f}",
                ],
                recommendations=[
                    "Cross-check reward variance, target-network lag, or bootstrap instability before blaming plasticity loss.",
                    "Compare this step with a short rolling window to see whether activation churn is persistent.",
                ],
            )
        return None
