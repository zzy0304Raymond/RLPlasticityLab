"""RLPlasticity public API."""

from .api import (
    probe_checkpoint_sequence,
    probe_model,
    probe_model_from_builder,
    probe_plasticity,
    probe_plasticity_from_builder,
    probe_plasticity_window,
    probe_training_step,
    scan_checkpoint,
)
from .plasticity.analyzer import (
    ANALYZERS,
    PLASTICITY_ANALYZERS,
    PlasticityAnalyzer,
    RuleBasedAnalyzer,
    create_checkpoint_scan_analyzer,
    create_default_plasticity_analyzer,
    create_forward_probe_analyzer,
)

try:
    from .adapters.pytorch import PlasticityMonitor
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name != "torch":
        raise
    PlasticityMonitor = None  # type: ignore[assignment]

__all__ = [
    "ANALYZERS",
    "PLASTICITY_ANALYZERS",
    "PlasticityAnalyzer",
    "PlasticityMonitor",
    "RuleBasedAnalyzer",
    "create_checkpoint_scan_analyzer",
    "create_default_plasticity_analyzer",
    "create_forward_probe_analyzer",
    "probe_checkpoint_sequence",
    "probe_model",
    "probe_model_from_builder",
    "probe_plasticity",
    "probe_plasticity_from_builder",
    "probe_plasticity_window",
    "probe_training_step",
    "scan_checkpoint",
]
