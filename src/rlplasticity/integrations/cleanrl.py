"""Helpers for common CleanRL-style naming and invocation patterns."""

from __future__ import annotations

from rlplasticity.api import probe_plasticity, probe_plasticity_window


def cleanrl_group_keywords() -> dict[str, list[str]]:
    """Return heuristic group keywords that fit common CleanRL module naming."""

    return {
        "encoder": ["encoder", "feature", "features", "cnn", "conv"],
        "trunk": ["trunk", "backbone", "body", "shared", "network"],
        "policy": ["actor", "policy", "logits", "pi"],
        "value": ["critic", "value", "vf", "qf", "q_net"],
    }


def probe_cleanrl_agent(
    agent,
    batches,
    *,
    loss_fn,
    optimizer,
    checkpoint=None,
    metadata=None,
):
    """Run a plasticity probe with CleanRL-friendly default grouping."""

    return probe_plasticity(
        agent,
        batches,
        loss_fn=loss_fn,
        optimizer=optimizer,
        checkpoint=checkpoint,
        metadata=metadata,
        group_keywords=cleanrl_group_keywords(),
    )


def probe_cleanrl_window(
    agent,
    batches,
    *,
    loss_fn,
    optimizer,
    checkpoint=None,
    metadata=None,
    max_steps: int | None = None,
):
    """Run a short window probe with CleanRL-friendly default grouping."""

    return probe_plasticity_window(
        agent,
        batches,
        loss_fn=loss_fn,
        optimizer=optimizer,
        checkpoint=checkpoint,
        metadata=metadata,
        group_keywords=cleanrl_group_keywords(),
        max_steps=max_steps,
    )
