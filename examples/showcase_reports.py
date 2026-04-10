"""Generate concrete showcase artifacts for the GitHub homepage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rlplasticity import probe_model, probe_plasticity, scan_checkpoint

from .rl_actor_case import (
    actor_loss,
    build_actor,
    build_optimizer,
    export_demo_artifacts,
    forward_batch,
)


def _write_report_triplet(base_path: Path, report) -> None:
    base_path.with_suffix(".txt").write_text(report.to_text(), encoding="utf-8")
    base_path.with_suffix(".json").write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
    base_path.with_suffix(".html").write_text(report.to_html(), encoding="utf-8")


def _build_frozen_encoder_model():
    model = build_actor()
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    return model


def generate_showcase(output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    artifacts_dir = output / "artifacts"
    artifacts = export_demo_artifacts(artifacts_dir)
    batch = torch.load(artifacts["batch"], map_location="cpu")

    static_report = scan_checkpoint(str(artifacts["actor"]))
    forward_report = probe_model(
        build_actor(),
        batch,
        checkpoint=str(artifacts["actor"]),
        forward_fn=forward_batch,
        metadata={"showcase": "healthy"},
    )

    healthy_model = build_actor()
    healthy_report = probe_plasticity(
        healthy_model,
        [batch],
        loss_fn=actor_loss,
        optimizer=build_optimizer(healthy_model),
        checkpoint=str(artifacts["actor"]),
        metadata={"showcase": "healthy"},
    )

    frozen_model = _build_frozen_encoder_model()
    frozen_report = probe_plasticity(
        frozen_model,
        [batch],
        loss_fn=actor_loss,
        optimizer=build_optimizer(frozen_model),
        metadata={"showcase": "frozen-encoder"},
    )

    _write_report_triplet(output / "healthy_static_scan", static_report)
    _write_report_triplet(output / "healthy_forward_probe", forward_report)
    _write_report_triplet(output / "healthy_plasticity_probe", healthy_report)
    _write_report_triplet(output / "frozen_encoder_plasticity_probe", frozen_report)

    summary = {
        "healthy": healthy_report.to_dict(),
        "frozen_encoder": frozen_report.to_dict(),
    }
    (output / "showcase_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "showcase_dir": output,
        "healthy_probe_json": output / "healthy_plasticity_probe.json",
        "frozen_probe_json": output / "frozen_encoder_plasticity_probe.json",
        "healthy_probe_html": output / "healthy_plasticity_probe.html",
        "frozen_probe_html": output / "frozen_encoder_plasticity_probe.html",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate README showcase artifacts.")
    parser.add_argument("--output-dir", default="docs/showcase")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    paths = generate_showcase(args.output_dir)
    for key, value in paths.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
