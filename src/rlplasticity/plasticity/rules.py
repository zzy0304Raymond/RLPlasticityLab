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


class TrunkBottleneckRule(BaseDiagnosticRule):
    name = "trunk_bottleneck"

    def __init__(self, *, min_gap: float = 0.25) -> None:
        self.min_gap = min_gap

    def evaluate(
        self,
        snapshot: Snapshot,
        metrics: dict[str, MetricResult],
    ) -> DiagnosticFinding | None:
        groups = snapshot.by_group()
        if "trunk" not in groups:
            return None

        trunk_score = _metric_value(metrics, "trunk_plasticity_score")
        encoder_score = _metric_value(metrics, "encoder_plasticity_score")
        policy_score = _metric_value(metrics, "policy_plasticity_score")
        candidates = [
            value
            for name, value in [("encoder", encoder_score), ("policy", policy_score)]
            if name in groups and value is not None
        ]
        if trunk_score is None or not candidates:
            return None
        reference = max(candidates)
        if reference - trunk_score >= self.min_gap:
            return DiagnosticFinding(
                name=self.name,
                severity="medium",
                summary="The shared trunk is adapting less than surrounding modules.",
                evidence=[
                    f"trunk_plasticity_score={trunk_score:.3f}",
                    f"reference_neighbor_score={reference:.3f}",
                ],
                recommendations=[
                    "Inspect whether the shared trunk is over-regularized or blocked by normalization settings.",
                    "Compare encoder and head gradients against trunk gradients across several nearby steps.",
                    "Check whether trunk activations are saturating before the output heads.",
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


class PlasticityDeclineTrendRule(BaseDiagnosticRule):
    name = "plasticity_decline_trend"

    def __init__(self, *, min_decline: float = 1e-4, min_points: int = 2) -> None:
        self.min_decline = min_decline
        self.min_points = min_points

    def evaluate(
        self,
        snapshot: Snapshot,
        metrics: dict[str, MetricResult],
    ) -> DiagnosticFinding | None:
        metric = metrics.get("plasticity_trend_delta")
        if metric is None or metric.metadata.get("points", 0) < self.min_points:
            return None
        if metric.value <= -self.min_decline:
            return DiagnosticFinding(
                name=self.name,
                severity="medium",
                summary="Plasticity is declining across the observed window or checkpoint sequence.",
                evidence=[
                    f"plasticity_trend_delta={metric.value:.6f}",
                    f"history_points={metric.metadata.get('points')}",
                ],
                recommendations=[
                    "Inspect when the decline begins rather than treating the final checkpoint in isolation.",
                    "Compare learning-rate, normalization, and replay changes around the first degraded history point.",
                    "Keep the history JSON so you can line up model-side decline with training logs.",
                ],
            )
        return None


class EncoderDeclineTrendRule(BaseDiagnosticRule):
    name = "encoder_decline_trend"

    def __init__(self, *, min_decline_gap: float = 1e-4, min_points: int = 2) -> None:
        self.min_decline_gap = min_decline_gap
        self.min_points = min_points

    def evaluate(
        self,
        snapshot: Snapshot,
        metrics: dict[str, MetricResult],
    ) -> DiagnosticFinding | None:
        if "encoder" not in snapshot.by_group():
            return None
        encoder_metric = metrics.get("encoder_plasticity_trend_delta")
        trunk_metric = metrics.get("trunk_plasticity_trend_delta")
        policy_metric = metrics.get("policy_plasticity_trend_delta")
        if encoder_metric is None or encoder_metric.metadata.get("points", 0) < self.min_points:
            return None
        references = [
            value
            for metric in [trunk_metric, policy_metric]
            if metric is not None and metric.metadata.get("points", 0) >= self.min_points
            for value in [metric.value]
        ]
        if not references:
            return None
        reference = max(references)
        if encoder_metric.value + self.min_decline_gap < reference:
            return DiagnosticFinding(
                name=self.name,
                severity="medium",
                summary="Encoder plasticity is declining faster than downstream modules over the observed history.",
                evidence=[
                    f"encoder_plasticity_trend_delta={encoder_metric.value:.6f}",
                    f"reference_downstream_trend_delta={reference:.6f}",
                ],
                recommendations=[
                    "Inspect whether encoder-specific updates collapse earlier than trunk or head updates.",
                    "Check preprocessing, augmentation, and encoder optimizer settings around the onset of decline.",
                ],
            )
        return None
