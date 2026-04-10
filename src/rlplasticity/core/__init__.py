"""Core abstractions shared by all analysis modules."""

from .aggregation import aggregate_snapshots
from .base import BaseAnalyzer, BaseDiagnosticRule, BaseMetric, BaseProbe
from .enums import AnalysisKind, EvidenceLevel
from .naming import infer_module_group
from .registry import FactoryRegistry
from .types import AnalysisReport, DiagnosticFinding, LayerSnapshot, MetricResult, Snapshot

__all__ = [
    "aggregate_snapshots",
    "AnalysisReport",
    "AnalysisKind",
    "BaseAnalyzer",
    "BaseProbe",
    "BaseDiagnosticRule",
    "BaseMetric",
    "DiagnosticFinding",
    "EvidenceLevel",
    "FactoryRegistry",
    "LayerSnapshot",
    "MetricResult",
    "Snapshot",
    "infer_module_group",
]
