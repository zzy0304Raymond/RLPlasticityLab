"""Generate validation artifacts for the first public release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rlplasticity import probe_checkpoint_sequence, probe_plasticity

from .rl_actor_case import (
    actor_loss,
    build_actor,
    build_optimizer,
    export_demo_artifacts,
    export_training_sequence_artifacts,
    make_batch,
)


def _write_report_triplet(base_path: Path, report) -> None:
    base_path.with_suffix(".txt").write_text(report.to_text(), encoding="utf-8")
    base_path.with_suffix(".json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    base_path.with_suffix(".html").write_text(report.to_html(), encoding="utf-8")


def _freeze_encoder(model) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False


def _freeze_policy_head(model) -> None:
    for parameter in model.policy_head.parameters():
        parameter.requires_grad = False


def _freeze_trunk(model) -> None:
    for parameter in model.trunk.parameters():
        parameter.requires_grad = False


def _zero_signal_loss(model, batch):
    logits = model(batch["obs"])
    zero_loss = (logits * 0.0).sum()
    return zero_loss, {"note": "zero-signal synthetic stall"}


def _run_case(
    *,
    checkpoint: str | None,
    batch,
    loss_fn,
    model_builder,
    optimizer_builder,
    metadata: dict[str, str],
):
    model = model_builder()
    optimizer = optimizer_builder(model)
    return probe_plasticity(
        model,
        [batch],
        loss_fn=loss_fn,
        optimizer=optimizer,
        checkpoint=checkpoint,
        metadata=metadata,
    )


def generate_validation_suite(output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    artifacts = export_demo_artifacts(output / "artifacts")
    actor_path = str(artifacts["actor"])
    batch = torch.load(artifacts["batch"], map_location="cpu")

    healthy_report = _run_case(
        checkpoint=actor_path,
        batch=batch,
        loss_fn=actor_loss,
        model_builder=build_actor,
        optimizer_builder=build_optimizer,
        metadata={"validation_case": "healthy"},
    )

    frozen_encoder_report = _run_case(
        checkpoint=actor_path,
        batch=batch,
        loss_fn=actor_loss,
        model_builder=lambda: _build_variant(_freeze_encoder),
        optimizer_builder=build_optimizer,
        metadata={"validation_case": "frozen_encoder"},
    )

    frozen_policy_head_report = _run_case(
        checkpoint=actor_path,
        batch=batch,
        loss_fn=actor_loss,
        model_builder=lambda: _build_variant(_freeze_policy_head),
        optimizer_builder=build_optimizer,
        metadata={"validation_case": "frozen_policy_head"},
    )

    frozen_trunk_report = _run_case(
        checkpoint=actor_path,
        batch=batch,
        loss_fn=actor_loss,
        model_builder=lambda: _build_variant(_freeze_trunk),
        optimizer_builder=build_optimizer,
        metadata={"validation_case": "frozen_trunk"},
    )

    global_stall_report = _run_case(
        checkpoint=actor_path,
        batch=batch,
        loss_fn=_zero_signal_loss,
        model_builder=build_actor,
        optimizer_builder=build_optimizer,
        metadata={"validation_case": "global_stall"},
    )

    sequence_artifacts = export_training_sequence_artifacts(output / "sequence_artifacts", steps=3)
    sequence_batches = torch.load(sequence_artifacts["batches"], map_location="cpu")
    checkpoint_sequence_report = probe_checkpoint_sequence(
        build_actor,
        [str(path) for path in sequence_artifacts["checkpoints"]],
        sequence_batches,
        loss_fn=actor_loss,
        optimizer_builder=build_optimizer,
        max_steps=2,
        metadata={"validation_case": "checkpoint_sequence"},
    )

    reports = {
        "healthy": healthy_report,
        "frozen_encoder": frozen_encoder_report,
        "frozen_policy_head": frozen_policy_head_report,
        "frozen_trunk": frozen_trunk_report,
        "global_stall": global_stall_report,
        "checkpoint_sequence": checkpoint_sequence_report,
    }

    for name, report in reports.items():
        _write_report_triplet(output / name, report)

    summary_rows = []
    for name, report in reports.items():
        findings = ", ".join(finding.name for finding in report.findings) or "none"
        summary_rows.append(
            {
                "case": name,
                "summary": report.summary(),
                "findings": findings,
                "text": f"{name}.txt",
                "json": f"{name}.json",
                "html": f"{name}.html",
            }
        )
    (output / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    lines = [
        "# Validation Artifacts",
        "",
        "| Case | Expected pattern | Actual summary | Findings |",
        "| --- | --- | --- | --- |",
        "| healthy | no acute issue | "
        f"{healthy_report.summary()} | "
        f"{', '.join(f.name for f in healthy_report.findings) or 'none'} |",
        "| frozen_encoder | encoder-side bottleneck | "
        f"{frozen_encoder_report.summary()} | "
        f"{', '.join(f.name for f in frozen_encoder_report.findings) or 'none'} |",
        "| frozen_policy_head | head-side bottleneck | "
        f"{frozen_policy_head_report.summary()} | "
        f"{', '.join(f.name for f in frozen_policy_head_report.findings) or 'none'} |",
        "| frozen_trunk | trunk-side bottleneck | "
        f"{frozen_trunk_report.summary()} | "
        f"{', '.join(f.name for f in frozen_trunk_report.findings) or 'none'} |",
        "| global_stall | global plasticity stall | "
        f"{global_stall_report.summary()} | "
        f"{', '.join(f.name for f in global_stall_report.findings) or 'none'} |",
        "| checkpoint_sequence | history metadata and trend metrics available | "
        f"{checkpoint_sequence_report.summary()} | "
        f"{', '.join(f.name for f in checkpoint_sequence_report.findings) or 'none'} |",
        "",
        "Generated files:",
        "",
    ]
    for row in summary_rows:
        lines.append(f"- `{row['case']}`: `{row['text']}`, `{row['json']}`, `{row['html']}`")
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")

    return {"output_dir": output, "summary_json": output / "summary.json", "readme": output / "README.md"}


def _build_variant(mutation):
    model = build_actor()
    mutation(model)
    return model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate validation artifacts for the release.")
    parser.add_argument("--output-dir", default="docs/validation")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    outputs = generate_validation_suite(args.output_dir)
    for key, value in outputs.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
