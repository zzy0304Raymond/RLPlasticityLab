"""Core data structures shared across all workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any

from .enums import AnalysisKind, EvidenceLevel


@dataclass(slots=True)
class LayerSnapshot:
    """Normalized per-layer evidence for any analysis mode."""

    name: str
    group: str
    module_type: str
    parameter_count: int = 0
    parameter_norm: float = 0.0
    parameter_mean_abs: float = 0.0
    parameter_zero_fraction: float | None = None
    parameter_max_abs: float | None = None
    gradient_norm: float | None = None
    update_norm: float | None = None
    relative_update: float | None = None
    grad_to_weight_ratio: float | None = None
    activation_mean: float | None = None
    activation_std: float | None = None
    activation_shift: float | None = None
    zero_activation_fraction: float | None = None
    max_activation_abs: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Snapshot:
    """Top-level evidence produced by a scan or probe."""

    kind: AnalysisKind
    evidence_level: EvidenceLevel
    step: int | None
    loss: float | None
    layers: list[LayerSnapshot]
    metadata: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    def by_group(self) -> dict[str, list[LayerSnapshot]]:
        grouped: dict[str, list[LayerSnapshot]] = {}
        for layer in self.layers:
            grouped.setdefault(layer.group, []).append(layer)
        return grouped

    def has_gradients(self) -> bool:
        return any(layer.gradient_norm is not None for layer in self.layers)

    def has_updates(self) -> bool:
        return any(layer.update_norm is not None for layer in self.layers)

    def has_activations(self) -> bool:
        return any(layer.activation_mean is not None for layer in self.layers)


@dataclass(slots=True)
class MetricResult:
    """Single metric output."""

    name: str
    value: float
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DiagnosticFinding:
    """Human-readable finding created by a diagnostic rule."""

    name: str
    severity: str
    summary: str
    evidence: list[str]
    recommendations: list[str]
    confidence: str = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnalysisReport:
    """Complete analysis report for one workflow run."""

    analyzer_name: str
    snapshot: Snapshot
    metrics: dict[str, MetricResult]
    findings: list[DiagnosticFinding]
    caveats: list[str] = field(default_factory=list)

    def top_layers_by(
        self,
        field_name: str,
        *,
        reverse: bool = True,
        limit: int = 5,
        none_fallback: float | None = None,
    ) -> list[LayerSnapshot]:
        fallback = none_fallback
        if fallback is None:
            fallback = float("-inf") if reverse else float("inf")

        return sorted(
            self.snapshot.layers,
            key=lambda layer: getattr(layer, field_name)
            if getattr(layer, field_name) is not None
            else fallback,
            reverse=reverse,
        )[:limit]

    def summary(self) -> str:
        if self.findings:
            headline = "; ".join(finding.summary for finding in self.findings[:2])
            if len(self.findings) > 2:
                headline += f" (+{len(self.findings) - 2} more)"
            return headline

        metric = self.metrics.get("plasticity_score") or self.metrics.get("activation_health_score")
        if metric is not None:
            return f"No acute issue detected. {metric.summary}"
        return "No acute issue detected."

    def group_average(self, attribute: str, group: str) -> float | None:
        group_layers = self.snapshot.by_group().get(group, [])
        values = [getattr(layer, attribute) for layer in group_layers if getattr(layer, attribute) is not None]
        if not values:
            return None
        return mean(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analyzer_name": self.analyzer_name,
            "snapshot": {
                **asdict(self.snapshot),
                "kind": self.snapshot.kind.value,
                "evidence_level": self.snapshot.evidence_level.value,
            },
            "metrics": {name: asdict(metric) for name, metric in self.metrics.items()},
            "findings": [asdict(finding) for finding in self.findings],
            "caveats": list(self.caveats),
        }

    def to_text(self) -> str:
        from rlplasticity.reporting.renderers import render_report_text

        return render_report_text(self)

    def to_html(self) -> str:
        from rlplasticity.reporting.renderers import render_report_html

        return render_report_html(self)
