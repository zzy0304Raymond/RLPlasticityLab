"""Human-readable report renderers."""

from __future__ import annotations

from html import escape

from rlplasticity.core.types import AnalysisReport, LayerSnapshot


def _format_optional(value: float | None, *, scientific: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6e}" if scientific else f"{value:.6f}"


def _format_layer_line(report: AnalysisReport, layer: LayerSnapshot) -> str:
    if report.snapshot.has_updates():
        return (
            f"{layer.name} [{layer.group}] "
            f"rel_update={_format_optional(layer.relative_update, scientific=True)} "
            f"grad/weight={_format_optional(layer.grad_to_weight_ratio, scientific=True)}"
        )
    if report.snapshot.has_activations():
        return (
            f"{layer.name} [{layer.group}] "
            f"activation_std={_format_optional(layer.activation_std, scientific=True)} "
            f"zero_frac={_format_optional(layer.zero_activation_fraction)}"
        )
    return (
        f"{layer.name} [{layer.group}] "
        f"param_norm={_format_optional(layer.parameter_norm, scientific=True)} "
        f"zero_frac={_format_optional(layer.parameter_zero_fraction)}"
    )


def _layer_section_title(report: AnalysisReport) -> str:
    if report.snapshot.has_updates():
        return "Lowest-Update Layers"
    if report.snapshot.has_activations():
        return "Lowest-Variation Layers"
    return "Smallest-Norm Layers"


def _choose_layers(report: AnalysisReport) -> list[LayerSnapshot]:
    if report.snapshot.has_updates():
        return report.top_layers_by("relative_update", reverse=False)
    if report.snapshot.has_activations():
        return report.top_layers_by("activation_std", reverse=False)
    return report.top_layers_by("parameter_norm", reverse=False)


def render_report_text(report: AnalysisReport) -> str:
    lines = []
    lines.append(
        "RLPlasticity Report | "
        f"analyzer={report.analyzer_name} | "
        f"kind={report.snapshot.kind.value} | "
        f"evidence={report.snapshot.evidence_level.value} | "
        f"layers={len(report.snapshot.layers)}"
    )
    if report.snapshot.loss is not None:
        lines.append(f"loss={report.snapshot.loss:.6f}")
    lines.append(f"summary={report.summary()}")

    lines.append("")
    lines.append("Metrics")
    for metric in report.metrics.values():
        lines.append(f"- {metric.summary}")

    lines.append("")
    lines.append("Findings")
    if report.findings:
        for finding in report.findings:
            lines.append(f"- [{finding.severity}/{finding.confidence}] {finding.summary}")
            for evidence in finding.evidence:
                lines.append(f"  evidence: {evidence}")
            for recommendation in finding.recommendations:
                lines.append(f"  next: {recommendation}")
    else:
        lines.append("- No diagnostic rule fired.")

    if report.caveats:
        lines.append("")
        lines.append("Caveats")
        for caveat in report.caveats:
            lines.append(f"- {caveat}")

    lines.append("")
    lines.append(_layer_section_title(report))
    for layer in _choose_layers(report):
        lines.append(f"- {_format_layer_line(report, layer)}")

    return "\n".join(lines)


def render_report_html(report: AnalysisReport) -> str:
    rows = []
    for layer in _choose_layers(report)[:10]:
        rows.append(
            "<tr>"
            f"<td>{escape(layer.name)}</td>"
            f"<td>{escape(layer.group)}</td>"
            f"<td>{escape(_format_optional(layer.relative_update, scientific=True))}</td>"
            f"<td>{escape(_format_optional(layer.activation_std, scientific=True))}</td>"
            f"<td>{escape(_format_optional(layer.parameter_norm, scientific=True))}</td>"
            "</tr>"
        )

    finding_items = []
    if report.findings:
        for finding in report.findings:
            evidence = "".join(f"<li>{escape(item)}</li>" for item in finding.evidence)
            recommendations = "".join(
                f"<li>{escape(item)}</li>" for item in finding.recommendations
            )
            finding_items.append(
                "<section class='finding'>"
                f"<h3>{escape(finding.severity.upper())}: {escape(finding.summary)}</h3>"
                f"<p><strong>Rule:</strong> {escape(finding.name)} | <strong>Confidence:</strong> {escape(finding.confidence)}</p>"
                f"<ul>{evidence}</ul>"
                f"<ul>{recommendations}</ul>"
                "</section>"
            )
    else:
        finding_items.append("<p>No diagnostic rule fired.</p>")

    metric_items = "".join(
        f"<li><strong>{escape(metric.name)}</strong>: {escape(metric.summary)}</li>"
        for metric in report.metrics.values()
    )
    caveat_items = "".join(f"<li>{escape(caveat)}</li>" for caveat in report.caveats)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RLPlasticity Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f0e8;
      --card: #fffdf8;
      --ink: #1f1d1a;
      --muted: #6b6257;
      --accent: #0d6b57;
      --line: #d7cbbd;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(13, 107, 87, 0.12), transparent 28%),
        linear-gradient(180deg, #f7f2ea, var(--bg));
      color: var(--ink);
    }}
    main {{
      max-width: 1040px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
      margin-bottom: 18px;
      box-shadow: 0 14px 34px rgba(31, 29, 26, 0.06);
    }}
    h1, h2, h3 {{
      margin-top: 0;
    }}
    ul {{
      padding-left: 20px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }}
    .finding {{
      border-left: 4px solid var(--accent);
      padding-left: 14px;
      margin-bottom: 14px;
    }}
    .meta {{
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main>
    <section class="card">
      <h1>RLPlasticity Report</h1>
      <p class="meta">analyzer={escape(report.analyzer_name)} | kind={escape(report.snapshot.kind.value)} | evidence={escape(report.snapshot.evidence_level.value)} | layers={len(report.snapshot.layers)} | loss={escape(str(report.snapshot.loss))}</p>
      <p>{escape(report.summary())}</p>
    </section>
    <section class="card">
      <h2>Metrics</h2>
      <ul>{metric_items}</ul>
    </section>
    <section class="card">
      <h2>Findings</h2>
      {''.join(finding_items)}
    </section>
    <section class="card">
      <h2>Caveats</h2>
      <ul>{caveat_items or '<li>No caveats recorded.</li>'}</ul>
    </section>
    <section class="card">
      <h2>{escape(_layer_section_title(report))}</h2>
      <table>
        <thead>
          <tr>
            <th>Layer</th>
            <th>Group</th>
            <th>Relative Update</th>
            <th>Activation Std</th>
            <th>Parameter Norm</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
