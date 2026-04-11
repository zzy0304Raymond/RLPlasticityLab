"""Window and checkpoint-sequence probe workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from rlplasticity.core.aggregation import aggregate_snapshots, build_history_entry

from .plasticity import collect_plasticity_probe, collect_plasticity_snapshots


def collect_plasticity_window(
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
    """Run a multi-batch plasticity window probe with explicit window metadata."""

    snapshots = collect_plasticity_snapshots(
        model,
        batches,
        loss_fn=loss_fn,
        optimizer=optimizer,
        checkpoint=checkpoint,
        group_keywords=group_keywords,
        max_steps=max_steps,
        metadata=metadata,
    )
    return aggregate_snapshots(
        snapshots,
        history_label_prefix="window-step",
        metadata_updates={"aggregation_mode": "batch_window"},
        caveat=(
            f"This report averages {len(snapshots)} probe batches into a short plasticity window; "
            "inspect history metrics when the trend looks unstable."
        ),
    )


def _checkpoint_label(checkpoint: str | Mapping[str, Any], index: int) -> str:
    if isinstance(checkpoint, str):
        return Path(checkpoint).name
    source = checkpoint.get("__label__") if isinstance(checkpoint, Mapping) else None
    if isinstance(source, str) and source:
        return source
    return f"checkpoint-{index + 1}"


def collect_checkpoint_sequence_probe(
    model_builder,
    checkpoints: Iterable[str | Mapping[str, Any]],
    batches,
    *,
    loss_fn,
    optimizer_builder,
    group_keywords: dict[str, list[str]] | None = None,
    max_steps: int | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Probe multiple checkpoints and aggregate them into a trend-aware sequence snapshot."""

    concrete_checkpoints = list(checkpoints)
    if not concrete_checkpoints:
        raise ValueError("At least one checkpoint is required for checkpoint-sequence probing.")

    per_checkpoint_snapshots = []
    history = []
    for index, checkpoint in enumerate(concrete_checkpoints):
        model = model_builder()
        optimizer = optimizer_builder(model)
        checkpoint_metadata = dict(metadata or {})
        checkpoint_metadata["checkpoint_label"] = _checkpoint_label(checkpoint, index)
        snapshot = collect_plasticity_probe(
            model,
            batches,
            loss_fn=loss_fn,
            optimizer=optimizer,
            checkpoint=checkpoint,
            group_keywords=group_keywords,
            max_steps=max_steps,
            metadata=checkpoint_metadata,
        )
        per_checkpoint_snapshots.append(snapshot)
        history.append(build_history_entry(snapshot, label=checkpoint_metadata["checkpoint_label"]))

    sequence_snapshot = aggregate_snapshots(
        per_checkpoint_snapshots,
        history_label_prefix="checkpoint",
        metadata_updates={
            "aggregation_mode": "checkpoint_sequence",
            "sequence_length": len(per_checkpoint_snapshots),
            "checkpoint_labels": [entry["label"] for entry in history],
        },
        caveat=(
            f"This report averages {len(per_checkpoint_snapshots)} checkpoints. "
            "Use the history section to inspect how plasticity moves across the sequence."
        ),
    )
    sequence_snapshot.metadata["history"] = history
    return sequence_snapshot
