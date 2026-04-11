"""Helpers for Stable-Baselines3 style policies without importing SB3 itself."""

from __future__ import annotations

from rlplasticity.api import probe_plasticity


def sb3_group_keywords() -> dict[str, list[str]]:
    """Return heuristic group keywords that fit common SB3 policy naming."""

    return {
        "encoder": ["features_extractor", "cnn"],
        "trunk": ["mlp_extractor", "shared_net", "latent", "net"],
        "policy": ["action_net", "policy", "pi", "actor"],
        "value": ["value_net", "critic", "vf", "qf"],
    }


def probe_sb3_policy(
    policy,
    batches,
    *,
    loss_fn,
    optimizer,
    checkpoint=None,
    metadata=None,
):
    """Run a plasticity probe with SB3-friendly default grouping."""

    return probe_plasticity(
        policy,
        batches,
        loss_fn=loss_fn,
        optimizer=optimizer,
        checkpoint=checkpoint,
        metadata=metadata,
        group_keywords=sb3_group_keywords(),
    )
