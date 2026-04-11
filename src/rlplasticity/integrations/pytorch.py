"""Helpers for integrating RLPlasticity into raw PyTorch training loops."""

from __future__ import annotations

from rlplasticity.api import probe_plasticity_window, probe_training_step


def probe_training_loop_step(
    model,
    batch,
    *,
    loss_fn,
    optimizer,
    checkpoint=None,
    metadata=None,
    group_keywords=None,
):
    """Run the single-step plasticity probe used in raw PyTorch loops."""

    return probe_training_step(
        model,
        batch,
        loss_fn=loss_fn,
        optimizer=optimizer,
        checkpoint=checkpoint,
        metadata=metadata,
        group_keywords=group_keywords,
    )


def probe_training_window(
    model,
    batches,
    *,
    loss_fn,
    optimizer,
    checkpoint=None,
    metadata=None,
    group_keywords=None,
    max_steps: int | None = None,
):
    """Run a short plasticity window over several PyTorch batches."""

    return probe_plasticity_window(
        model,
        batches,
        loss_fn=loss_fn,
        optimizer=optimizer,
        checkpoint=checkpoint,
        metadata=metadata,
        group_keywords=group_keywords,
        max_steps=max_steps,
    )
