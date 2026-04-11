import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rlplasticity import scan_checkpoint
from rlplasticity.core.aggregation import aggregate_snapshots
from rlplasticity.core.enums import AnalysisKind, EvidenceLevel
from rlplasticity.core.types import LayerSnapshot, Snapshot
from rlplasticity.ingest.checkpoints import extract_state_dict
from rlplasticity.plasticity.analyzer import create_checkpoint_scan_analyzer

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class RobustnessTests(unittest.TestCase):
    def test_extract_state_dict_prefers_nested_keys(self) -> None:
        payload = {
            "epoch": 3,
            "state_dict": {"encoder.weight": [1.0, 2.0]},
            "actor": {"policy.weight": [3.0]},
        }
        state_dict = extract_state_dict(payload)
        self.assertEqual(state_dict, {"encoder.weight": [1.0, 2.0]})

    def test_scan_checkpoint_accepts_wrapped_payload(self) -> None:
        report = scan_checkpoint(
            {
                "state_dict": {
                    "encoder.weight": [0.0, 1.0],
                    "policy.bias": [0.5],
                }
            }
        )
        self.assertEqual(report.snapshot.kind, AnalysisKind.CHECKPOINT_SCAN)
        self.assertEqual(len(report.snapshot.layers), 2)

    def test_empty_static_snapshot_still_renders(self) -> None:
        report = create_checkpoint_scan_analyzer().analyze(
            Snapshot(
                kind=AnalysisKind.CHECKPOINT_SCAN,
                evidence_level=EvidenceLevel.STATIC,
                step=None,
                loss=None,
                layers=[],
                caveats=["empty input"],
            )
        )
        text = report.to_text()
        html = report.to_html()
        payload = report.to_dict()
        self.assertIn("No acute issue detected", text)
        self.assertIn("RLPlasticity Report", html)
        self.assertEqual(payload["snapshot"]["kind"], "checkpoint_scan")

    def test_aggregate_snapshots_marks_window_evidence(self) -> None:
        snapshots = [
            Snapshot(
                kind=AnalysisKind.PLASTICITY_PROBE,
                evidence_level=EvidenceLevel.UPDATE,
                step=1,
                loss=1.0,
                layers=[
                    LayerSnapshot(
                        name="encoder.linear",
                        group="encoder",
                        module_type="Linear",
                        parameter_count=10,
                        parameter_norm=1.0,
                        parameter_mean_abs=0.1,
                        gradient_norm=0.2,
                        update_norm=0.01,
                        relative_update=0.01,
                        grad_to_weight_ratio=0.2,
                    )
                ],
            ),
            Snapshot(
                kind=AnalysisKind.PLASTICITY_PROBE,
                evidence_level=EvidenceLevel.UPDATE,
                step=2,
                loss=3.0,
                layers=[
                    LayerSnapshot(
                        name="encoder.linear",
                        group="encoder",
                        module_type="Linear",
                        parameter_count=10,
                        parameter_norm=3.0,
                        parameter_mean_abs=0.3,
                        gradient_norm=0.4,
                        update_norm=0.03,
                        relative_update=0.03,
                        grad_to_weight_ratio=0.4,
                    )
                ],
            ),
        ]
        aggregated = aggregate_snapshots(snapshots)
        self.assertEqual(aggregated.evidence_level, EvidenceLevel.WINDOW)
        self.assertEqual(aggregated.metadata["window_size"], 2)
        self.assertAlmostEqual(aggregated.loss, 2.0)

    def test_integration_keyword_helpers_cover_expected_groups(self) -> None:
        from rlplasticity.integrations.cleanrl import cleanrl_group_keywords
        from rlplasticity.integrations.sb3 import sb3_group_keywords

        self.assertEqual(set(cleanrl_group_keywords()), {"encoder", "trunk", "policy", "value"})
        self.assertEqual(set(sb3_group_keywords()), {"encoder", "trunk", "policy", "value"})


@unittest.skipUnless(torch is not None, "PyTorch is required for robustness integration tests.")
class PyTorchRobustnessTests(unittest.TestCase):
    def test_encoder_bottleneck_triggers_on_frozen_encoder(self) -> None:
        from examples.rl_actor_case import actor_loss, build_actor, build_optimizer, make_batch
        from rlplasticity import probe_plasticity

        model = build_actor()
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        optimizer = build_optimizer(model)
        batch = make_batch()

        report = probe_plasticity(
            model,
            [batch],
            loss_fn=actor_loss,
            optimizer=optimizer,
            metadata={"scenario": "frozen-encoder"},
        )

        finding_names = {finding.name for finding in report.findings}
        self.assertIn("encoder_bottleneck", finding_names)

    def test_head_saturation_triggers_on_frozen_policy_head(self) -> None:
        from examples.rl_actor_case import actor_loss, build_actor, build_optimizer, make_batch
        from rlplasticity import probe_plasticity

        model = build_actor()
        for parameter in model.policy_head.parameters():
            parameter.requires_grad = False
        optimizer = build_optimizer(model)
        batch = make_batch()

        report = probe_plasticity(
            model,
            [batch],
            loss_fn=actor_loss,
            optimizer=optimizer,
            metadata={"scenario": "frozen-policy-head"},
        )

        finding_names = {finding.name for finding in report.findings}
        self.assertIn("head_saturation", finding_names)

    def test_global_stall_triggers_on_zero_signal_loss(self) -> None:
        from examples.rl_actor_case import build_actor, build_optimizer, make_batch
        from rlplasticity import probe_plasticity

        def zero_signal_loss(model, batch):
            return (model(batch["obs"]) * 0.0).sum()

        model = build_actor()
        optimizer = build_optimizer(model)
        batch = make_batch()

        report = probe_plasticity(
            model,
            [batch],
            loss_fn=zero_signal_loss,
            optimizer=optimizer,
            metadata={"scenario": "global-stall"},
        )

        finding_names = {finding.name for finding in report.findings}
        self.assertIn("global_plasticity_stall", finding_names)

    def test_pytorch_integration_helpers_return_reports(self) -> None:
        from examples.rl_actor_case import actor_loss, build_actor, build_optimizer, make_batch
        from rlplasticity.integrations.pytorch import probe_training_loop_step, probe_training_window

        batch = make_batch()
        model = build_actor()
        optimizer = build_optimizer(model)

        step_report = probe_training_loop_step(
            model,
            batch,
            loss_fn=actor_loss,
            optimizer=optimizer,
            metadata={"integration": "pytorch-helper"},
        )
        self.assertEqual(step_report.snapshot.evidence_level.value, "update")

        window_model = build_actor()
        window_optimizer = build_optimizer(window_model)
        window_report = probe_training_window(
            window_model,
            [batch, batch],
            loss_fn=actor_loss,
            optimizer=window_optimizer,
            max_steps=2,
            metadata={"integration": "pytorch-window-helper"},
        )
        self.assertEqual(window_report.snapshot.evidence_level.value, "window")


@unittest.skipUnless(torch is not None, "PyTorch is required for UX integration tests.")
class UserExperienceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.pythonpath = os.pathsep.join(["src", "."])

    def test_cli_scan_json_output_is_parseable(self) -> None:
        from examples.rl_actor_case import export_demo_artifacts

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = export_demo_artifacts(temp_dir)
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rlplasticity.cli",
                    "--format",
                    "json",
                    "scan",
                    "--checkpoint",
                    str(artifacts["actor"]),
                ],
                cwd=self.repo_root,
                env={**os.environ, "PYTHONPATH": self.pythonpath},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            payload = json.loads(process.stdout)
            self.assertEqual(payload["snapshot"]["kind"], "checkpoint_scan")
            self.assertIn("metrics", payload)

    def test_cli_invalid_builder_is_actionable(self) -> None:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "rlplasticity.cli",
                "probe-model",
                "--builder",
                "bad-spec",
                "--samples",
                "missing.pt",
            ],
            cwd=self.repo_root,
            env={**os.environ, "PYTHONPATH": self.pythonpath},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("pkg.module:callable", process.stderr + process.stdout)


if __name__ == "__main__":
    unittest.main()
