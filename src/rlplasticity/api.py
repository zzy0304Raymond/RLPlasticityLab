"""Public workflows for the first usable release."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlplasticity.plasticity.analyzer import (
    create_checkpoint_scan_analyzer,
    create_default_plasticity_analyzer,
    create_forward_probe_analyzer,
)


def scan_checkpoint(
    checkpoint: str | Mapping[str, Any],
    *,
    map_location: str = "cpu",
    group_keywords: dict[str, list[str]] | None = None,
):
    """Analyze a bare checkpoint with static structural evidence only."""

    from rlplasticity.probes.static import collect_checkpoint_scan

    snapshot = collect_checkpoint_scan(
        checkpoint,
        map_location=map_location,
        group_keywords=group_keywords,
    )
    analyzer = create_checkpoint_scan_analyzer()
    return analyzer.analyze(snapshot)


def probe_model(
    model,
    batch: Any,
    *,
    checkpoint: str | Mapping[str, Any] | None = None,
    forward_fn=None,
    group_keywords: dict[str, list[str]] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Run a forward-only probe against a loadable model and sample batch."""

    from rlplasticity.probes.model import collect_model_probe

    _output, snapshot = collect_model_probe(
        model,
        batch,
        checkpoint=checkpoint,
        forward_fn=forward_fn,
        group_keywords=group_keywords,
        metadata=metadata,
    )
    analyzer = create_forward_probe_analyzer()
    return analyzer.analyze(snapshot)


def probe_plasticity(
    model,
    batches,
    *,
    loss_fn,
    optimizer,
    checkpoint: str | Mapping[str, Any] | None = None,
    group_keywords: dict[str, list[str]] | None = None,
    max_steps: int | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Run one or more low-cost update probes and analyze plasticity evidence."""

    from rlplasticity.probes.plasticity import collect_plasticity_probe

    snapshot = collect_plasticity_probe(
        model,
        batches,
        loss_fn=loss_fn,
        optimizer=optimizer,
        checkpoint=checkpoint,
        group_keywords=group_keywords,
        max_steps=max_steps,
        metadata=metadata,
    )
    analyzer = create_default_plasticity_analyzer()
    return analyzer.analyze(snapshot)


def probe_training_step(
    model,
    batch,
    *,
    loss_fn,
    optimizer,
    checkpoint: str | Mapping[str, Any] | None = None,
    group_keywords: dict[str, list[str]] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Convenience alias for probing one concrete training step."""

    return probe_plasticity(
        model,
        [batch],
        loss_fn=loss_fn,
        optimizer=optimizer,
        checkpoint=checkpoint,
        group_keywords=group_keywords,
        max_steps=1,
        metadata=metadata,
    )


def probe_model_from_builder(
    model_builder,
    batch: Any,
    *,
    checkpoint: str | Mapping[str, Any] | None = None,
    forward_fn=None,
    group_keywords: dict[str, list[str]] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Build a model lazily and run a forward-only probe."""

    model = model_builder()
    return probe_model(
        model,
        batch,
        checkpoint=checkpoint,
        forward_fn=forward_fn,
        group_keywords=group_keywords,
        metadata=metadata,
    )


def probe_plasticity_from_builder(
    model_builder,
    batches,
    *,
    loss_fn,
    optimizer_builder=None,
    checkpoint: str | Mapping[str, Any] | None = None,
    group_keywords: dict[str, list[str]] | None = None,
    max_steps: int | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Build a model and optimizer lazily before running a plasticity probe."""

    model = model_builder()
    if optimizer_builder is None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyTorch is required to build the default optimizer for builder-based plasticity probing."
            ) from exc
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    else:
        optimizer = optimizer_builder(model)
    return probe_plasticity(
        model,
        batches,
        loss_fn=loss_fn,
        optimizer=optimizer,
        checkpoint=checkpoint,
        group_keywords=group_keywords,
        max_steps=max_steps,
        metadata=metadata,
    )


def probe_plasticity_window(
    model,
    batches,
    *,
    loss_fn,
    optimizer,
    checkpoint: str | Mapping[str, Any] | None = None,
    group_keywords: dict[str, list[str]] | None = None,
    max_steps: int | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Run a multi-batch plasticity window probe with trend-aware history metadata."""

    from rlplasticity.probes.sequence import collect_plasticity_window

    snapshot = collect_plasticity_window(
        model,
        batches,
        loss_fn=loss_fn,
        optimizer=optimizer,
        checkpoint=checkpoint,
        group_keywords=group_keywords,
        max_steps=max_steps,
        metadata=metadata,
    )
    analyzer = create_default_plasticity_analyzer()
    return analyzer.analyze(snapshot)


def probe_checkpoint_sequence(
    model_builder,
    checkpoints,
    batches,
    *,
    loss_fn,
    optimizer_builder,
    group_keywords: dict[str, list[str]] | None = None,
    max_steps: int | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Run a trend-aware plasticity analysis across a sequence of checkpoints."""

    from rlplasticity.probes.sequence import collect_checkpoint_sequence_probe

    snapshot = collect_checkpoint_sequence_probe(
        model_builder,
        checkpoints,
        batches,
        loss_fn=loss_fn,
        optimizer_builder=optimizer_builder,
        group_keywords=group_keywords,
        max_steps=max_steps,
        metadata=metadata,
    )
    analyzer = create_default_plasticity_analyzer()
    return analyzer.analyze(snapshot)


def load_checkpoint_for_api(path: str, *, map_location: str = "cpu"):
    """Convenience helper exposed for CLI and example code."""

    from rlplasticity.ingest.checkpoints import extract_state_dict, load_checkpoint

    payload = load_checkpoint(path, map_location=map_location)
    return extract_state_dict(payload)
