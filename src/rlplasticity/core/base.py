"""Core interfaces for analysis modules and workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import AnalysisReport, DiagnosticFinding, MetricResult, Snapshot


class BaseMetric(ABC):
    """Computes a single metric from a snapshot."""

    name: str

    @abstractmethod
    def compute(self, snapshot: Snapshot) -> MetricResult:
        """Return a metric result for the provided snapshot."""


class BaseDiagnosticRule(ABC):
    """Turns metrics and snapshots into actionable findings."""

    name: str

    @abstractmethod
    def evaluate(
        self,
        snapshot: Snapshot,
        metrics: dict[str, MetricResult],
    ) -> DiagnosticFinding | None:
        """Return a finding when the rule is triggered."""


class BaseAnalyzer(ABC):
    """Bundles metrics and rules into a concrete analysis module."""

    @abstractmethod
    def analyze(self, snapshot: Snapshot) -> AnalysisReport:
        """Analyze a snapshot and produce a report."""


class BaseProbe(ABC):
    """Produces a normalized snapshot from a concrete artifact or runtime."""

    @abstractmethod
    def collect(self) -> Snapshot:
        """Collect evidence and return a snapshot."""
