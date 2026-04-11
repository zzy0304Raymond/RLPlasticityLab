"""Command-line entry points for RLPlasticity."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from rlplasticity.api import (
    probe_checkpoint_sequence,
    probe_model,
    probe_plasticity,
    probe_plasticity_window,
    scan_checkpoint,
)


def _load_symbol(spec: str):
    module_name, _, attr_name = spec.partition(":")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid symbol spec '{spec}'. Use 'pkg.module:callable'.")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _load_samples(path: str):
    if path.endswith(".json"):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to load sample artifacts from .pt files.") from exc
    return torch.load(path, map_location="cpu")


def _render(report, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(report.to_dict(), indent=2)
    if output_format == "html":
        return report.to_html()
    return report.to_text()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlplasticity", description="Low-cost RL plasticity diagnostics.")
    parser.add_argument("--format", choices=("text", "json", "html"), default="text")

    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Run a static checkpoint scan.")
    scan.add_argument("--checkpoint", required=True)

    probe_forward = subparsers.add_parser("probe-model", help="Run a forward-only model probe.")
    probe_forward.add_argument("--builder", required=True, help="Model builder spec pkg.module:callable")
    probe_forward.add_argument("--samples", required=True, help="Sample artifact path (.pt or .json)")
    probe_forward.add_argument("--checkpoint")
    probe_forward.add_argument("--forward", help="Optional forward fn spec pkg.module:callable")

    probe_update = subparsers.add_parser("probe-plasticity", help="Run a low-cost update probe.")
    probe_update.add_argument("--builder", required=True, help="Model builder spec pkg.module:callable")
    probe_update.add_argument("--samples", required=True, help="Sample artifact path (.pt or .json)")
    probe_update.add_argument("--loss", required=True, help="Loss fn spec pkg.module:callable")
    probe_update.add_argument("--checkpoint")
    probe_update.add_argument("--optimizer", help="Optimizer factory spec pkg.module:callable")
    probe_update.add_argument("--max-steps", type=int, default=1)

    probe_window = subparsers.add_parser("probe-window", help="Run a multi-batch plasticity window probe.")
    probe_window.add_argument("--builder", required=True, help="Model builder spec pkg.module:callable")
    probe_window.add_argument("--samples", required=True, help="Sample artifact path (.pt or .json)")
    probe_window.add_argument("--loss", required=True, help="Loss fn spec pkg.module:callable")
    probe_window.add_argument("--checkpoint")
    probe_window.add_argument("--optimizer", help="Optimizer factory spec pkg.module:callable")
    probe_window.add_argument("--max-steps", type=int, default=8)

    probe_sequence = subparsers.add_parser(
        "probe-sequence",
        help="Run a plasticity probe across a sequence of checkpoints.",
    )
    probe_sequence.add_argument("--builder", required=True, help="Model builder spec pkg.module:callable")
    probe_sequence.add_argument("--samples", required=True, help="Sample artifact path (.pt or .json)")
    probe_sequence.add_argument("--loss", required=True, help="Loss fn spec pkg.module:callable")
    probe_sequence.add_argument("--optimizer", required=True, help="Optimizer factory spec pkg.module:callable")
    probe_sequence.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Ordered list of checkpoint paths to compare.",
    )
    probe_sequence.add_argument("--max-steps", type=int, default=4)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        report = scan_checkpoint(args.checkpoint)
    elif args.command == "probe-model":
        builder = _load_symbol(args.builder)
        model = builder()
        batch = _load_samples(args.samples)
        forward_fn = _load_symbol(args.forward) if args.forward else None
        report = probe_model(model, batch, checkpoint=args.checkpoint, forward_fn=forward_fn)
    elif args.command == "probe-plasticity":
        builder = _load_symbol(args.builder)
        model = builder()
        batch = _load_samples(args.samples)
        loss_fn = _load_symbol(args.loss)
        if args.optimizer:
            optimizer_factory = _load_symbol(args.optimizer)
            optimizer = optimizer_factory(model)
        else:
            try:
                import torch
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError(
                    "PyTorch is required to build the default optimizer for a plasticity probe."
                ) from exc
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        report = probe_plasticity(
            model,
            batch,
            loss_fn=loss_fn,
            optimizer=optimizer,
            checkpoint=args.checkpoint,
            max_steps=args.max_steps,
        )
    elif args.command == "probe-window":
        builder = _load_symbol(args.builder)
        model = builder()
        batch = _load_samples(args.samples)
        loss_fn = _load_symbol(args.loss)
        if args.optimizer:
            optimizer_factory = _load_symbol(args.optimizer)
            optimizer = optimizer_factory(model)
        else:
            try:
                import torch
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError(
                    "PyTorch is required to build the default optimizer for a window probe."
                ) from exc
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        report = probe_plasticity_window(
            model,
            batch,
            loss_fn=loss_fn,
            optimizer=optimizer,
            checkpoint=args.checkpoint,
            max_steps=args.max_steps,
        )
    elif args.command == "probe-sequence":
        builder = _load_symbol(args.builder)
        batch = _load_samples(args.samples)
        loss_fn = _load_symbol(args.loss)
        optimizer_factory = _load_symbol(args.optimizer)
        report = probe_checkpoint_sequence(
            builder,
            args.checkpoints,
            batch,
            loss_fn=loss_fn,
            optimizer_builder=optimizer_factory,
            max_steps=args.max_steps,
        )
    else:  # pragma: no cover
        parser.error(f"Unhandled command: {args.command}")

    print(_render(report, args.format))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
