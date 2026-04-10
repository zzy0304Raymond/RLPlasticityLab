"""Forward-only model probe workflow."""

from __future__ import annotations

from typing import Any

from rlplasticity.adapters.pytorch import PlasticityMonitor

from ._shared import maybe_load_model_checkpoint


def collect_model_probe(
    model,
    batch: Any,
    *,
    checkpoint: str | dict[str, Any] | None = None,
    forward_fn=None,
    group_keywords: dict[str, list[str]] | None = None,
    step: int | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Run a forward-only probe and return output plus a snapshot."""

    maybe_load_model_checkpoint(model, checkpoint)
    monitor = PlasticityMonitor(model, group_keywords=group_keywords)
    try:
        if forward_fn is None:
            forward_callable = lambda: model(batch)
        else:
            forward_callable = lambda: forward_fn(model, batch)
        output, snapshot = monitor.capture_forward(forward_callable, step=step, metadata=metadata)
        return output, snapshot
    finally:
        monitor.close()
