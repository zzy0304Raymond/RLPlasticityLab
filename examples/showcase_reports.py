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
    healthy = _headline_metrics(healthy_report)
    frozen = _headline_metrics(frozen_report)

    def _lines(x: int, y: int, title: str, values: list[str], accent: str) -> str:
        parts = [
            f"<text x='{x}' y='{y}' font-size='24' font-weight='700' fill='#1f1d1a'>{title}</text>",
        ]
        current_y = y + 36
        for value in values:
            safe = (
                value.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            parts.append(
                f"<text x='{x}' y='{current_y}' font-size='16' fill='#3f3a33'>{safe}</text>"
            )
            current_y += 28
        parts.append(
            f"<rect x='{x}' y='{y + 160}' width='360' height='6' rx='3' fill='{accent}' opacity='0.9' />"
        )
        return "\n".join(parts)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="520" viewBox="0 0 1120 520" role="img" aria-labelledby="title desc">
  <title id="title">RLPlasticity healthy vs frozen-encoder comparison</title>
  <desc id="desc">Side-by-side comparison of healthy and frozen-encoder plasticity probe outputs.</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f5efe7" />
      <stop offset="100%" stop-color="#eee4d4" />
    </linearGradient>
  </defs>
  <rect width="1120" height="520" fill="url(#bg)" />
  <text x="60" y="68" font-size="34" font-weight="700" fill="#1f1d1a">RLPlasticity showcase: healthy vs frozen encoder</text>
  <text x="60" y="104" font-size="18" fill="#5c5448">Both reports are generated from real demo artifacts in this repository.</text>

  <rect x="60" y="140" width="470" height="320" rx="24" fill="#fffdf8" stroke="#d7cbbd" />
  <rect x="590" y="140" width="470" height="320" rx="24" fill="#fffdf8" stroke="#d7cbbd" />

  <text x="88" y="182" font-size="18" font-weight="700" fill="#0d6b57">Healthy checkpoint</text>
  {_lines(88, 220, "Probe result", [
        healthy["summary"],
        healthy["plasticity_score"],
        healthy["stagnant_layer_fraction"],
        healthy["encoder_plasticity_score"],
        healthy["trunk_plasticity_score"],
        healthy["policy_plasticity_score"],
        healthy["findings"],
    ], "#0d6b57")}

  <text x="618" y="182" font-size="18" font-weight="700" fill="#b53d23">Frozen encoder</text>
  {_lines(618, 220, "Probe result", [
        frozen["summary"],
        frozen["plasticity_score"],
        frozen["stagnant_layer_fraction"],
        frozen["encoder_plasticity_score"],
        frozen["trunk_plasticity_score"],
        frozen["policy_plasticity_score"],
        frozen["findings"],
    ], "#b53d23")}
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
