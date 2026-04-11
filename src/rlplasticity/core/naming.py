"""Naming and grouping helpers."""

from __future__ import annotations

from collections.abc import Iterable
import re


DEFAULT_GROUP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "encoder": ("encoder", "backbone", "feature", "features", "embed", "conv", "cnn"),
    "policy": ("policy", "actor", "action", "logits", "pi"),
    "value": ("value", "critic", "qf", "q_net", "q1", "q2", "vf"),
    "aux": ("aux", "predictor", "decoder"),
}


_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokenize_module_name(module_name: str) -> tuple[str, ...]:
    lowered = module_name.lower().replace("_", " ").replace("-", " ").replace(".", " ")
    tokens = [token for token in _TOKEN_SPLIT_RE.split(lowered) if token]
    return tuple(tokens)


def _keyword_matches(keyword: str, lowered: str, tokens: tuple[str, ...]) -> bool:
    if "_" in keyword:
        return keyword in lowered
    if len(keyword) <= 2:
        return any(token == keyword or token.startswith(keyword) for token in tokens)
    return any(token == keyword or token.startswith(keyword) for token in tokens)


def infer_module_group(
    module_name: str,
    extra_keywords: dict[str, Iterable[str]] | None = None,
) -> str:
    """Infer a semantic module group from a module name."""

    lowered = module_name.lower()
    tokens = _tokenize_module_name(module_name)
    keyword_map = {key: tuple(values) for key, values in DEFAULT_GROUP_KEYWORDS.items()}
    if extra_keywords:
        for key, values in extra_keywords.items():
            keyword_map[key] = tuple(values)

    for group, keywords in keyword_map.items():
        if any(_keyword_matches(keyword.lower(), lowered, tokens) for keyword in keywords):
            return group
    return "trunk"
