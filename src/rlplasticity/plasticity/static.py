"""Static checkpoint scan metrics and findings."""

from __future__ import annotations

from statistics import median

from rlplasticity.core.base import BaseDiagnosticRule, BaseMetric
from rlplasticity.core.types import DiagnosticFinding, MetricResult, Snapshot


def _nonnull(values):
    return [value for value in values if value is not None]


class MeanParameterZeroFractionMetric(BaseMetric):
    name = "mean_parameter_zero_fraction"

    def compute(self, snapshot: Snapshot) -> MetricResult:
        values = _nonnull([layer.parameter_zero_fraction for layer in snapshot.layers])
        if not values:
            return MetricResult(self.name, 0.0, "No parameter sparsity stats were captured.")
        score = sum(values) / len(values)
        return MetricResult(
            self.name,
            score,
            f"Mean parameter zero fraction={score:.3f}",
        )


class ParameterNormDispersionMetric(BaseMetric):
    name = "parameter_norm_dispersion"

    def compute(self, snapshot: Snapshot) -> MetricResult:
        norms = [layer.parameter_norm for layer in snapshot.layers if layer.parameter_norm > 0.0]
        if not norms:
            return MetricResult(self.name, 0.0, "No parameter norms were captured.")
        med = median(norms)
        dispersion = max(norms) / med if med > 0 else 0.0
        return MetricResult(
            self.name,
            dispersion,
            f"Parameter norm dispersion={dispersion:.3f}",
            metadata={"median_norm": med},
        )


class SmallNormLayerFractionMetric(BaseMetric):
    name = "small_norm_layer_fraction"

    def __init__(self, *, relative_to_median: float = 0.05) -> None:
        self.relative_to_median = relative_to_median

    def compute(self, snapshot: Snapshot) -> MetricResult:
        norms = [layer.parameter_norm for layer in snapshot.layers if layer.parameter_norm > 0.0]
        if not norms:
            return MetricResult(self.name, 0.0, "No parameter norms were captured.")
        med = median(norms)
        threshold = med * self.relative_to_median
        small = [layer.name for layer in snapshot.layers if layer.parameter_norm <= threshold]
        fraction = len(small) / len(snapshot.layers) if snapshot.layers else 0.0
        return MetricResult(
            self.name,
            fraction,
            f"Small-norm layer fraction={fraction:.3f}",
            metadata={"small_norm_layers": small, "threshold": threshold},
        )


class WidespreadNearZeroParameterRule(BaseDiagnosticRule):
    name = "widespread_near_zero_parameters"

    def __init__(self, *, min_mean_zero_fraction: float = 0.95) -> None:
        self.min_mean_zero_fraction = min_mean_zero_fraction

    def evaluate(self, snapshot: Snapshot, metrics: dict[str, MetricResult]) -> DiagnosticFinding | None:
        metric = metrics.get("mean_parameter_zero_fraction")
        if metric is None:
            return None
        if metric.value >= self.min_mean_zero_fraction:
            return DiagnosticFinding(
                name=self.name,
                severity="medium",
                summary="Many parameters are numerically near zero in this checkpoint.",
                evidence=[metric.summary],
                recommendations=[
                    "Check whether this checkpoint was partially initialized, heavily pruned, or saved before meaningful training.",
                    "Run a forward or update probe before treating this as a plasticity diagnosis.",
                ],
                confidence="low",
            )
        return None


class ExtremeNormOutlierRule(BaseDiagnosticRule):
    name = "extreme_norm_outlier"

    def __init__(self, *, min_dispersion: float = 1000.0) -> None:
        self.min_dispersion = min_dispersion

    def evaluate(self, snapshot: Snapshot, metrics: dict[str, MetricResult]) -> DiagnosticFinding | None:
        metric = metrics.get("parameter_norm_dispersion")
        if metric is None:
            return None
        if metric.value >= self.min_dispersion:
            return DiagnosticFinding(
                name=self.name,
                severity="low",
                summary="Parameter norms are extremely imbalanced across layers.",
                evidence=[metric.summary],
                recommendations=[
                    "Inspect the highest-norm and lowest-norm layers before running more expensive probes.",
                    "Treat this as a structural hint rather than proof of a plasticity issue.",
                ],
                confidence="low",
            )
        return None
