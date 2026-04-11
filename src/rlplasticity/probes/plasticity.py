"""Low-cost plasticity probe workflow."""

from __future__ import annotations

from typing import Any

from rlplasticity.adapters.pytorch import PlasticityMonitor
from rlplasticity.core.aggregation import aggregate_snapshots

from ._shared import maybe_load_model_checkpoint, normalize_batches


def _coerce_loss_result(result: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(result, tuple):
        loss = result[0]
        metadata = result[1] if len(result) > 1 and isinstance(result[1], dict) else {}
        return loss, metadata
    return result, {}


def collect_plasticity_snapshots(
    model,
    batches,
    *,
    loss_fn,
    optimizer,
    checkpoint: str | dict[str, Any] | None = None,
    group_keywords: dict[str, list[str]] | None = None,
    max_steps: int | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Run one or more low-cost update probes and return raw per-step snapshots."""

    maybe_load_model_checkpoint(model, checkpoint)
    concrete_batches = normalize_batches(batches)
    if max_steps is not None:
        concrete_batches = concrete_batches[:max_steps]
    if not concrete_batches:
        raise ValueError("At least one batch is required for a plasticity probe.")

    monitor = PlasticityMonitor(model, group_keywords=group_keywords)
    snapshots = []
    base_metadata = dict(metadata or {})
    try:
        for step_index, batch in enumerate(concrete_batches):
            step = step_index + 1
            monitor.begin_step(step=step, metadata=base_metadata)
            optimizer.zero_grad(set_to_none=True)
            loss_tensor, extra_metadata = _coerce_loss_result(loss_fn(model, batch))
            loss_tensor.backward()
            optimizer.step()
            merged_metadata = dict(base_metadata)
            merged_metadata.update(extra_metadata)
            snapshots.append(
                monitor.end_step(loss=float(loss_tensor.detach().item()), metadata=merged_metadata)
            )
    finally:
        monitor.close()

    return snapshots


def collect_plasticity_probe(
    model,
    batches,
    *,
    loss_fn,
    optimizer,
    checkpoint: str | dict[str, Any] | None = None,
    group_keywords: dict[str, list[str]] | None = None,
    max_steps: int | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Run one or more low-cost update probes and aggregate them."""

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

    return aggregate_snapshots(snapshots)
