"""Naming and grouping helpers."""

from __future__ import annotations

from collections.abc import Iterable


DEFAULT_GROUP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "encoder": ("encoder", "backbone", "feature", "embed", "conv", "cnn"),
    "policy": ("policy", "actor", "action", "logits", "pi"),
    "value": ("value", "critic", "q", "vf"),
    "aux": ("aux", "predictor", "decoder"),
}


def infer_module_group(
    module_name: str,
    extra_keywords: dict[str, Iterable[str]] | None = None,
) -> str:
    """Infer a semantic module group from a module name."""

    lowered = module_name.lower()
    keyword_map = {key: tuple(values) for key, values in DEFAULT_GROUP_KEYWORDS.items()}
    if extra_keywords:
        for key, values in extra_keywords.items():
            keyword_map[key] = tuple(values)

    for group, keywords in keyword_map.items():
        if any(keyword in lowered for keyword in keywords):
            return group
    return "trunk"
