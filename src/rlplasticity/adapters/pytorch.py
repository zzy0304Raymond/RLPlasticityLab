"""PyTorch runtime helpers for forward and update probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from rlplasticity.core.enums import AnalysisKind, EvidenceLevel
from rlplasticity.core.naming import infer_module_group
from rlplasticity.core.types import LayerSnapshot, Snapshot


EPSILON = 1e-12


@dataclass(slots=True)
class _ActivationStats:
    mean: float
    std: float
    zero_fraction: float
    max_abs: float


def _module_has_direct_parameters(module: nn.Module) -> bool:
    return any(True for _ in module.parameters(recurse=False))


def _extract_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _combined_norm(tensors: list[torch.Tensor]) -> float:
    if not tensors:
        return 0.0
    squares = torch.stack([tensor.detach().float().norm() ** 2 for tensor in tensors])
    return torch.sqrt(squares.sum()).item()


def _mean_abs(tensors: list[torch.Tensor]) -> float:
    flat = []
    for tensor in tensors:
        flat.append(tensor.detach().float().abs().reshape(-1))
    if not flat:
        return 0.0
    merged = torch.cat(flat)
    return merged.mean().item()


def _zero_fraction(tensors: list[torch.Tensor], *, threshold: float = 1e-12) -> float | None:
    flat = []
    for tensor in tensors:
        flat.append(tensor.detach().float().abs().reshape(-1))
    if not flat:
        return None
    merged = torch.cat(flat)
    return (merged < threshold).float().mean().item()


def _max_abs(tensors: list[torch.Tensor]) -> float | None:
    flat = []
    for tensor in tensors:
        flat.append(tensor.detach().float().abs().reshape(-1))
    if not flat:
        return None
    merged = torch.cat(flat)
    return merged.max().item()


def _parameter_count(parameters: list[torch.Tensor]) -> int:
    return int(sum(parameter.numel() for parameter in parameters))


class PlasticityMonitor:
    """Collects normalized snapshots from a live PyTorch model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        group_keywords: dict[str, list[str]] | None = None,
        activation_sample_limit: int = 4096,
    ) -> None:
        self.model = model
        self.group_keywords = group_keywords or {}
        self.activation_sample_limit = activation_sample_limit
        self._modules = self._discover_modules()
        self._hooks = []
        self._current_activation_stats: dict[str, _ActivationStats] = {}
        self._previous_activation_stats: dict[str, _ActivationStats] = {}
        self._previous_parameters: dict[str, list[torch.Tensor]] = {}
        self._current_step: int | None = None
        self._current_metadata: dict[str, Any] = {}
        self._register_hooks()

    def _discover_modules(self) -> dict[str, nn.Module]:
        modules: dict[str, nn.Module] = {}
        for name, module in self.model.named_modules():
            if not name:
                continue
            if _module_has_direct_parameters(module):
                modules[name] = module
        return modules

    def _register_hooks(self) -> None:
        for name, module in self._modules.items():
            handle = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(handle)

    def close(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def begin_step(self, step: int | None = None, metadata: dict[str, Any] | None = None) -> None:
        """Capture pre-update parameters and clear transient activation state."""

        self._current_step = step
        self._current_metadata = dict(metadata or {})
        self._current_activation_stats.clear()
        self._previous_parameters = {
            name: [parameter.detach().clone() for parameter in module.parameters(recurse=False)]
            for name, module in self._modules.items()
        }

    def capture_forward(
        self,
        forward_callable,
        *,
        step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, Snapshot]:
        """Run a forward probe and return both model output and a snapshot."""

        self._current_step = step
        self._current_metadata = dict(metadata or {})
        self._current_activation_stats.clear()
        output = forward_callable()
        snapshot = self._build_snapshot(
            kind=AnalysisKind.MODEL_PROBE,
            evidence_level=EvidenceLevel.FORWARD,
            loss=None,
            metadata=self._current_metadata,
            include_gradients=False,
            include_updates=False,
            caveats=[
                "This is a forward-only probe. It can show activation health and local responsiveness, but not true plasticity under optimization.",
            ],
        )
        self._previous_activation_stats = dict(self._current_activation_stats)
        return output, snapshot

    def end_step(self, loss: float | None = None, metadata: dict[str, Any] | None = None) -> Snapshot:
        """Build an update-level snapshot after backward and optimizer step."""

        merged_metadata = dict(self._current_metadata)
        if metadata:
            merged_metadata.update(metadata)

        snapshot = self._build_snapshot(
            kind=AnalysisKind.PLASTICITY_PROBE,
            evidence_level=EvidenceLevel.UPDATE,
            loss=loss,
            metadata=merged_metadata,
            include_gradients=True,
            include_updates=True,
            caveats=[
                "This is a low-cost plasticity probe, not a full experiment. Treat the result as ranked evidence for where to investigate next.",
            ],
        )
        self._previous_activation_stats = dict(self._current_activation_stats)
        return snapshot

    def _build_snapshot(
        self,
        *,
        kind: AnalysisKind,
        evidence_level: EvidenceLevel,
        loss: float | None,
        metadata: dict[str, Any],
        include_gradients: bool,
        include_updates: bool,
        caveats: list[str],
    ) -> Snapshot:
        layers: list[LayerSnapshot] = []
        for name, module in self._modules.items():
            current_params = [parameter.detach() for parameter in module.parameters(recurse=False)]
            previous_params = self._previous_parameters.get(name, [])
            gradients = []
            if include_gradients:
                gradients = [
                    parameter.grad.detach()
                    for parameter in module.parameters(recurse=False)
                    if parameter.grad is not None
                ]

            update_tensors: list[torch.Tensor] = []
            if include_updates:
                for current, previous in zip(current_params, previous_params):
                    update_tensors.append((current - previous).detach())

            parameter_norm = _combined_norm(current_params)
            gradient_norm = _combined_norm(gradients) if include_gradients else None
            update_norm = _combined_norm(update_tensors) if include_updates else None
            relative_update = None
            grad_to_weight_ratio = None
            if update_norm is not None:
                relative_update = update_norm / max(parameter_norm, EPSILON)
            if gradient_norm is not None:
                grad_to_weight_ratio = gradient_norm / max(parameter_norm, EPSILON)

            current_activation = self._current_activation_stats.get(name)
            previous_activation = self._previous_activation_stats.get(name)
            activation_shift: float | None = None
            if current_activation and previous_activation:
                activation_shift = (
                    abs(current_activation.mean - previous_activation.mean)
                    + abs(current_activation.std - previous_activation.std)
                )

            group = infer_module_group(name, self.group_keywords)
            layers.append(
                LayerSnapshot(
                    name=name,
                    group=group,
                    module_type=module.__class__.__name__,
                    parameter_count=_parameter_count(current_params),
                    parameter_norm=parameter_norm,
                    parameter_mean_abs=_mean_abs(current_params),
                    parameter_zero_fraction=_zero_fraction(current_params),
                    parameter_max_abs=_max_abs(current_params),
                    gradient_norm=gradient_norm,
                    update_norm=update_norm,
                    relative_update=relative_update,
                    grad_to_weight_ratio=grad_to_weight_ratio,
                    activation_mean=current_activation.mean if current_activation else None,
                    activation_std=current_activation.std if current_activation else None,
                    activation_shift=activation_shift,
                    zero_activation_fraction=current_activation.zero_fraction if current_activation else None,
                    max_activation_abs=current_activation.max_abs if current_activation else None,
                )
            )

        return Snapshot(
            kind=kind,
            evidence_level=evidence_level,
            step=self._current_step,
            loss=loss,
            layers=layers,
            metadata=dict(metadata),
            caveats=list(caveats),
        )

    def _make_hook(self, module_name: str):
        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = _extract_tensor(output)
            if tensor is None:
                return
            flat = tensor.detach().float().reshape(-1)
            if flat.numel() == 0:
                return
            if flat.numel() > self.activation_sample_limit:
                flat = flat[: self.activation_sample_limit]
            zero_fraction = (flat.abs() < 1e-8).float().mean().item()
            self._current_activation_stats[module_name] = _ActivationStats(
                mean=flat.mean().item(),
                std=flat.std(unbiased=False).item(),
                zero_fraction=zero_fraction,
                max_abs=flat.abs().max().item(),
            )

        return _hook
