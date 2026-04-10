"""Rule-based analyzers for the first public release."""

from __future__ import annotations

from rlplasticity.core.base import BaseAnalyzer, BaseDiagnosticRule, BaseMetric
from rlplasticity.core.registry import FactoryRegistry
from rlplasticity.core.types import AnalysisReport, Snapshot
from rlplasticity.plasticity.forward import (
    ActivationHealthScoreMetric,
    EncoderLowResponseRule,
    GroupActivationHealthMetric,
    LowVariationLayerFractionMetric,
    WidespreadInactiveActivationRule,
)
from rlplasticity.plasticity.metrics import (
    GroupPlasticityMetric,
    MeanActivationShiftMetric,
    PlasticityScoreMetric,
    StagnantLayerFractionMetric,
)
from rlplasticity.plasticity.rules import (
    EncoderBottleneckRule,
    GlobalPlasticityStallRule,
    HeadSaturationRule,
    RepresentationChurnRule,
)
from rlplasticity.plasticity.static import (
    ExtremeNormOutlierRule,
    MeanParameterZeroFractionMetric,
    ParameterNormDispersionMetric,
    SmallNormLayerFractionMetric,
    WidespreadNearZeroParameterRule,
)


class RuleBasedAnalyzer(BaseAnalyzer):
    """Runs a configurable set of metrics and rules on one snapshot."""

    def __init__(
        self,
        *,
        analyzer_name: str,
        metrics: list[BaseMetric] | None = None,
        rules: list[BaseDiagnosticRule] | None = None,
    ) -> None:
        self.analyzer_name = analyzer_name
        self.metrics = metrics or []
        self.rules = rules or []

    def analyze(self, snapshot: Snapshot) -> AnalysisReport:
        metric_results = {metric.name: metric.compute(snapshot) for metric in self.metrics}
        findings = []
        for rule in self.rules:
            finding = rule.evaluate(snapshot, metric_results)
            if finding is not None:
                findings.append(finding)
        findings.sort(key=lambda item: ("high", "medium", "low").index(item.severity))
        caveats = list(snapshot.caveats)
        return AnalysisReport(
            analyzer_name=self.analyzer_name,
            snapshot=snapshot,
            metrics=metric_results,
            findings=findings,
            caveats=caveats,
        )


class PlasticityAnalyzer(RuleBasedAnalyzer):
    """Backward-compatible name for the dynamic plasticity analyzer."""


def create_default_plasticity_analyzer() -> PlasticityAnalyzer:
    metrics: list[BaseMetric] = [
        PlasticityScoreMetric(),
        StagnantLayerFractionMetric(),
        GroupPlasticityMetric("encoder"),
        GroupPlasticityMetric("trunk"),
        GroupPlasticityMetric("policy"),
        GroupPlasticityMetric("value"),
        MeanActivationShiftMetric(),
    ]
    rules: list[BaseDiagnosticRule] = [
        GlobalPlasticityStallRule(),
        EncoderBottleneckRule(),
        HeadSaturationRule(),
        RepresentationChurnRule(),
    ]
    return PlasticityAnalyzer(
        analyzer_name="plasticity/default",
        metrics=metrics,
        rules=rules,
    )


def create_forward_probe_analyzer() -> RuleBasedAnalyzer:
    metrics: list[BaseMetric] = [
        ActivationHealthScoreMetric(),
        LowVariationLayerFractionMetric(),
        GroupActivationHealthMetric("encoder"),
        GroupActivationHealthMetric("trunk"),
        GroupActivationHealthMetric("policy"),
        GroupActivationHealthMetric("value"),
    ]
    rules: list[BaseDiagnosticRule] = [
        WidespreadInactiveActivationRule(),
        EncoderLowResponseRule(),
    ]
    return RuleBasedAnalyzer(
        analyzer_name="probe/forward",
        metrics=metrics,
        rules=rules,
    )


def create_checkpoint_scan_analyzer() -> RuleBasedAnalyzer:
    metrics: list[BaseMetric] = [
        MeanParameterZeroFractionMetric(),
        ParameterNormDispersionMetric(),
        SmallNormLayerFractionMetric(),
    ]
    rules: list[BaseDiagnosticRule] = [
        WidespreadNearZeroParameterRule(),
        ExtremeNormOutlierRule(),
    ]
    return RuleBasedAnalyzer(
        analyzer_name="scan/checkpoint",
        metrics=metrics,
        rules=rules,
    )


ANALYZERS = FactoryRegistry[RuleBasedAnalyzer]()
ANALYZERS.register("scan/checkpoint", create_checkpoint_scan_analyzer)
ANALYZERS.register("probe/forward", create_forward_probe_analyzer)
ANALYZERS.register("plasticity/default", create_default_plasticity_analyzer)
PLASTICITY_ANALYZERS = ANALYZERS
