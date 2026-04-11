"""Session-style integration helpers for repeated probing workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from rlplasticity.api import (
    probe_checkpoint_sequence,
    probe_plasticity_from_builder,
    probe_plasticity_window,
    probe_training_step,
)


LossFn = Callable[[Any, Any], Any]
ModelBuilder = Callable[[], Any]
OptimizerBuilder = Callable[[Any], Any]


@dataclass(slots=True)
class TrainingProbeSession:
    """Bundle stable runtime probe inputs for repeated training-side diagnostics."""

    model: Any
    optimizer: Any
    loss_fn: LossFn
    checkpoint: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    group_keywords: dict[str, list[str]] | None = None

    def probe_step(self, batch, *, metadata: dict[str, Any] | None = None):
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        return probe_training_step(
            self.model,
            batch,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            checkpoint=self.checkpoint,
            metadata=merged,
            group_keywords=self.group_keywords,
        )

    def probe_window(self, batches, *, max_steps: int | None = None, metadata: dict[str, Any] | None = None):
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        return probe_plasticity_window(
            self.model,
            batches,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            checkpoint=self.checkpoint,
            max_steps=max_steps,
            metadata=merged,
            group_keywords=self.group_keywords,
        )


@dataclass(slots=True)
class BuilderProbeSession:
    """Bundle builder-based probe inputs for checkpoint-centric diagnostics."""

    model_builder: ModelBuilder
    loss_fn: LossFn
    optimizer_builder: OptimizerBuilder | None = None
    checkpoint: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    group_keywords: dict[str, list[str]] | None = None

    def probe_window(self, batches, *, max_steps: int | None = None, metadata: dict[str, Any] | None = None):
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        return probe_plasticity_from_builder(
            self.model_builder,
            batches,
            loss_fn=self.loss_fn,
            optimizer_builder=self.optimizer_builder,
            checkpoint=self.checkpoint,
            max_steps=max_steps,
            metadata=merged,
            group_keywords=self.group_keywords,
        )

    def probe_sequence(self, checkpoints, batches, *, max_steps: int | None = None, metadata: dict[str, Any] | None = None):
        if self.optimizer_builder is None:
            raise ValueError("BuilderProbeSession.probe_sequence requires an optimizer_builder.")
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        return probe_checkpoint_sequence(
            self.model_builder,
            checkpoints,
            batches,
            loss_fn=self.loss_fn,
            optimizer_builder=self.optimizer_builder,
            max_steps=max_steps,
            metadata=merged,
            group_keywords=self.group_keywords,
        )
