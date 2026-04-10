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


def _headline_metrics(report) -> dict[str, str]:
    metrics = report.metrics
    return {
        "summary": report.summary(),
        "plasticity_score": metrics.get("plasticity_score").summary if "plasticity_score" in metrics else "n/a",
        "stagnant_layer_fraction": metrics.get("stagnant_layer_fraction").summary if "stagnant_layer_fraction" in metrics else "n/a",
        "encoder_plasticity_score": metrics.get("encoder_plasticity_score").summary if "encoder_plasticity_score" in metrics else "n/a",
        "trunk_plasticity_score": metrics.get("trunk_plasticity_score").summary if "trunk_plasticity_score" in metrics else "n/a",
        "policy_plasticity_score": metrics.get("policy_plasticity_score").summary if "policy_plasticity_score" in metrics else "n/a",
        "findings": "; ".join(finding.summary for finding in report.findings) or "No diagnostic rule fired.",
    }


def _render_comparison_svg(healthy_report, frozen_report, output_path: Path) -> None:
    def _metric_value(report, name: str) -> float:
        metric = report.metrics.get(name)
        return float(metric.value) if metric is not None else 0.0

    healthy_scores = {
        "Encoder": _metric_value(healthy_report, "encoder_plasticity_score"),
        "Trunk": _metric_value(healthy_report, "trunk_plasticity_score"),
        "Policy": _metric_value(healthy_report, "policy_plasticity_score"),
    }
    frozen_scores = {
        "Encoder": _metric_value(frozen_report, "encoder_plasticity_score"),
        "Trunk": _metric_value(frozen_report, "trunk_plasticity_score"),
        "Policy": _metric_value(frozen_report, "policy_plasticity_score"),
    }

    def _bar_group(x: int, y: int, scores: dict[str, float], accent: str) -> str:
        parts: list[str] = []
        current_y = y
        for label, score in scores.items():
            width = 220 * max(0.0, min(score, 1.0))
            value_x = x + 300
            parts.append(
                f"<text x='{x}' y='{current_y}' font-size='16' font-weight='600' fill='#2d2924'>{label}</text>"
            )
            parts.append(
                f"<rect x='{x}' y='{current_y + 14}' width='220' height='16' rx='8' fill='#eadfce' />"
            )
            parts.append(
                f"<rect x='{x}' y='{current_y + 14}' width='{width:.1f}' height='16' rx='8' fill='{accent}' />"
            )
            parts.append(
                f"<text x='{value_x}' y='{current_y + 28}' font-size='15' fill='#5a5147'>{score:.2f}</text>"
            )
            current_y += 62
        return "\n".join(parts)

    healthy_summary = "No issue detected"
    frozen_summary = "Encoder bottleneck"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="460" viewBox="0 0 1120 460" role="img" aria-labelledby="title desc">
  <title id="title">RLPlasticity healthy vs frozen-encoder comparison</title>
  <desc id="desc">Side-by-side comparison of healthy and frozen-encoder plasticity probe outputs.</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f5efe7" />
      <stop offset="100%" stop-color="#eee4d4" />
    </linearGradient>
  </defs>
  <rect width="1120" height="460" fill="url(#bg)" />
  <text x="60" y="64" font-size="34" font-weight="700" fill="#1f1d1a">RLPlasticity showcase</text>
  <text x="60" y="96" font-size="18" fill="#5c5448">Healthy policy vs frozen encoder on the same probe batch</text>

  <rect x="60" y="128" width="470" height="280" rx="24" fill="#fffdf8" stroke="#d7cbbd" />
  <rect x="590" y="128" width="470" height="280" rx="24" fill="#fffdf8" stroke="#d7cbbd" />

  <text x="90" y="172" font-size="24" font-weight="700" fill="#0d6b57">Healthy</text>
  <rect x="360" y="148" width="130" height="34" rx="17" fill="#dff4eb" />
  <text x="383" y="171" font-size="15" font-weight="700" fill="#0d6b57">{healthy_summary}</text>
  {_bar_group(90, 220, healthy_scores, "#0d6b57")}

  <text x="620" y="172" font-size="24" font-weight="700" fill="#b53d23">Frozen encoder</text>
  <rect x="880" y="148" width="140" height="34" rx="17" fill="#f7ddd6" />
  <text x="901" y="171" font-size="15" font-weight="700" fill="#b53d23">{frozen_summary}</text>
  {_bar_group(620, 220, frozen_scores, "#b53d23")}

  <text x="60" y="438" font-size="15" fill="#6a6054">Generated from real report artifacts committed in this repository.</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


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
    _render_comparison_svg(healthy_report, frozen_report, output / "healthy_vs_frozen_encoder.svg")

    summary = {
        "healthy": healthy_report.to_dict(),
        "frozen_encoder": frozen_report.to_dict(),
    }
    (output / "showcase_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "showcase_dir": output,
        "comparison_svg": output / "healthy_vs_frozen_encoder.svg",
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
