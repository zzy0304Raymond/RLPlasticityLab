"""Stable enums used across the public API."""

from __future__ import annotations

from enum import Enum


class AnalysisKind(str, Enum):
    CHECKPOINT_SCAN = "checkpoint_scan"
    MODEL_PROBE = "model_probe"
    PLASTICITY_PROBE = "plasticity_probe"


class EvidenceLevel(str, Enum):
    STATIC = "static"
    FORWARD = "forward"
    UPDATE = "update"
    WINDOW = "window"
