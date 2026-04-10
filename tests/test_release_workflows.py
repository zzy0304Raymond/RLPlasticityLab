import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from rlplasticity import scan_checkpoint
from rlplasticity.core.enums import AnalysisKind, EvidenceLevel
from rlplasticity.core.types import LayerSnapshot, Snapshot
from rlplasticity.plasticity.analyzer import (
    create_checkpoint_scan_analyzer,
    create_default_plasticity_analyzer,
    create_forward_probe_analyzer,
)


class AnalyzerTests(unittest.TestCase):
    def test_global_stall_is_detected(self) -> None:
        analyzer = create_default_plasticity_analyzer()
        snapshot = Snapshot(
            kind=AnalysisKind.PLASTICITY_PROBE,
            evidence_level=EvidenceLevel.UPDATE,
            step=10,
            loss=1.0,
            layers=[
                LayerSnapshot(
                    name="encoder.linear1",
                    group="encoder",
                    module_type="Linear",
                    parameter_count=100,
                    parameter_norm=10.0,
                    parameter_mean_abs=0.1,
                    parameter_zero_fraction=0.0,
                    gradient_norm=1e-8,
                    update_norm=1e-8,
                    relative_update=1e-9,
                    grad_to_weight_ratio=1e-9,
                    zero_activation_fraction=1.0,
                ),
                LayerSnapshot(
                    name="policy_head",
                    group="policy",
                    module_type="Linear",
                    parameter_count=20,
                    parameter_norm=5.0,
                    parameter_mean_abs=0.2,
                    parameter_zero_fraction=0.0,
                    gradient_norm=1e-8,
                    update_norm=1e-8,
                    relative_update=1e-9,
                    grad_to_weight_ratio=1e-9,
                    zero_activation_fraction=1.0,
                ),
            ],
        )

        report = analyzer.analyze(snapshot)

        self.assertTrue(any(finding.name == "global_plasticity_stall" for finding in report.findings))

    def test_encoder_bottleneck_is_detected(self) -> None:
        analyzer = create_default_plasticity_analyzer()
        snapshot = Snapshot(
            kind=AnalysisKind.PLASTICITY_PROBE,
            evidence_level=EvidenceLevel.UPDATE,
            step=20,
            loss=0.5,
            layers=[
                LayerSnapshot(
                    name="encoder.linear1",
                    group="encoder",
                    module_type="Linear",
                    parameter_count=100,
                    parameter_norm=10.0,
                    parameter_mean_abs=0.1,
                    parameter_zero_fraction=0.0,
                    gradient_norm=1e-4,
                    update_norm=1e-5,
                    relative_update=1e-6,
                    grad_to_weight_ratio=1e-6,
                    zero_activation_fraction=0.5,
                ),
                LayerSnapshot(
                    name="trunk.linear",
                    group="trunk",
                    module_type="Linear",
                    parameter_count=100,
                    parameter_norm=10.0,
                    parameter_mean_abs=0.1,
                    parameter_zero_fraction=0.0,
                    gradient_norm=1e-1,
                    update_norm=1e-2,
                    relative_update=1e-3,
                    grad_to_weight_ratio=1e-2,
                    zero_activation_fraction=0.1,
                ),
                LayerSnapshot(
                    name="policy_head",
                    group="policy",
                    module_type="Linear",
                    parameter_count=20,
                    parameter_norm=5.0,
                    parameter_mean_abs=0.2,
                    parameter_zero_fraction=0.0,
                    gradient_norm=1e-1,
                    update_norm=1e-2,
                    relative_update=1e-3,
                    grad_to_weight_ratio=1e-2,
                    zero_activation_fraction=0.1,
                ),
            ],
        )

        report = analyzer.analyze(snapshot)

        self.assertTrue(any(finding.name == "encoder_bottleneck" for finding in report.findings))

    def test_forward_analyzer_marks_low_response(self) -> None:
        analyzer = create_forward_probe_analyzer()
        snapshot = Snapshot(
            kind=AnalysisKind.MODEL_PROBE,
            evidence_level=EvidenceLevel.FORWARD,
            step=1,
            loss=None,
            layers=[
                LayerSnapshot(
                    name="encoder.block",
                    group="encoder",
                    module_type="Linear",
                    parameter_count=10,
                    parameter_norm=1.0,
                    parameter_mean_abs=0.1,
                    activation_mean=0.0,
                    activation_std=1e-6,
                    zero_activation_fraction=0.99,
                ),
                LayerSnapshot(
                    name="policy_head",
                    group="policy",
                    module_type="Linear",
                    parameter_count=10,
                    parameter_norm=1.0,
                    parameter_mean_abs=0.1,
                    activation_mean=0.1,
                    activation_std=0.5,
                    zero_activation_fraction=0.1,
                ),
            ],
        )

        report = analyzer.analyze(snapshot)
        self.assertTrue(any(finding.name == "encoder_low_response" for finding in report.findings))


class WorkflowTests(unittest.TestCase):
    def test_scan_checkpoint_uses_static_mode(self) -> None:
        report = scan_checkpoint(
            {
                "encoder.weight": [[0.0, 0.0], [0.0, 0.0]],
                "policy.bias": [1.0, -1.0],
            }
        )

        self.assertEqual(report.snapshot.kind, AnalysisKind.CHECKPOINT_SCAN)
        self.assertEqual(report.snapshot.evidence_level, EvidenceLevel.STATIC)
        self.assertIn("static checkpoint scan", " ".join(report.caveats).lower())

    def test_report_json_serializes(self) -> None:
        report = create_checkpoint_scan_analyzer().analyze(
            Snapshot(
                kind=AnalysisKind.CHECKPOINT_SCAN,
                evidence_level=EvidenceLevel.STATIC,
                step=None,
                loss=None,
                layers=[
                    LayerSnapshot(
                        name="encoder.weight",
                        group="encoder",
                        module_type="Parameter",
                        parameter_count=4,
                        parameter_norm=0.0,
                        parameter_mean_abs=0.0,
                        parameter_zero_fraction=1.0,
                    )
                ],
            )
        )

        payload = json.dumps(report.to_dict())
        self.assertIn("checkpoint_scan", payload)


class CliTests(unittest.TestCase):
    def test_cli_help_returns_zero(self) -> None:
        process = subprocess.run(
            [sys.executable, "-m", "rlplasticity.cli", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONPATH": "src"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 0)
        self.assertIn("probe-plasticity", process.stdout)


if __name__ == "__main__":
    unittest.main()
