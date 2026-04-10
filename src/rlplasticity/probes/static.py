"""Static checkpoint scan workflow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlplasticity.ingest.checkpoints import extract_state_dict, load_checkpoint, summarize_state_dict


def collect_checkpoint_scan(
    checkpoint: str | Mapping[str, Any],
    *,
    map_location: str = "cpu",
    group_keywords: dict[str, list[str]] | None = None,
):
    """Collect a static snapshot from a checkpoint path or state_dict."""

    payload = load_checkpoint(checkpoint, map_location=map_location) if isinstance(checkpoint, str) else checkpoint
    state_dict = extract_state_dict(payload)
    source = checkpoint if isinstance(checkpoint, str) else "in_memory_state_dict"
    return summarize_state_dict(state_dict, source=source, group_keywords=group_keywords)
