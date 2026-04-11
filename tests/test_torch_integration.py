import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@unittest.skipUnless(torch is not None, "PyTorch is required for integration tests.")
class TorchIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.pythonpath = os.pathsep.join(["src", "."])

    def _prepare_artifacts(self):
        from examples.rl_actor_case import export_demo_artifacts

        temp_dir = tempfile.TemporaryDirectory()
        artifacts = export_demo_artifacts(temp_dir.name)
        return temp_dir, artifacts

    def test_python_api_with_real_checkpoint_and_batch(self) -> None:
        from examples.rl_actor_case import (
            actor_loss,
            build_actor,
            build_optimizer,
            export_training_sequence_artifacts,
            forward_batch,
        )
        from rlplasticity import (
            probe_checkpoint_sequence,
            probe_model,
            probe_plasticity,
            probe_plasticity_window,
            scan_checkpoint,
        )

        temp_dir, artifacts = self._prepare_artifacts()
        self.addCleanup(temp_dir.cleanup)

        actor_path = artifacts["actor"]
        batch = torch.load(artifacts["batch"], map_location="cpu")

        static_report = scan_checkpoint(str(actor_path))
        self.assertEqual(static_report.snapshot.kind.value, "checkpoint_scan")

        forward_report = probe_model(
            build_actor(),
            batch,
            checkpoint=str(actor_path),
            forward_fn=forward_batch,
            metadata={"integration": "python-api"},
        )
        self.assertEqual(forward_report.snapshot.kind.value, "model_probe")
        self.assertEqual(forward_report.snapshot.evidence_level.value, "forward")

        model = build_actor()
        optimizer = build_optimizer(model)
        plasticity_report = probe_plasticity(
            model,
            [batch],
            loss_fn=actor_loss,
            optimizer=optimizer,
            checkpoint=str(actor_path),
            metadata={"integration": "python-api"},
        )
        self.assertEqual(plasticity_report.snapshot.kind.value, "plasticity_probe")
        self.assertEqual(plasticity_report.snapshot.evidence_level.value, "update")
        self.assertIn("plasticity_score", plasticity_report.metrics)

        window_model = build_actor()
        window_optimizer = build_optimizer(window_model)
        window_report = probe_plasticity_window(
            window_model,
            [batch, batch],
            loss_fn=actor_loss,
            optimizer=window_optimizer,
            checkpoint=str(actor_path),
            max_steps=2,
            metadata={"integration": "window"},
        )
        self.assertEqual(window_report.snapshot.evidence_level.value, "window")
        self.assertGreaterEqual(len(window_report.snapshot.metadata.get("history", [])), 2)

        sequence_dir = tempfile.TemporaryDirectory()
        self.addCleanup(sequence_dir.cleanup)
        sequence_artifacts = export_training_sequence_artifacts(sequence_dir.name, steps=3)
        sequence_batches = torch.load(sequence_artifacts["batches"], map_location="cpu")
        sequence_report = probe_checkpoint_sequence(
            build_actor,
            [str(path) for path in sequence_artifacts["checkpoints"]],
            sequence_batches,
            loss_fn=actor_loss,
            optimizer_builder=build_optimizer,
            max_steps=2,
            metadata={"integration": "sequence"},
        )
        self.assertEqual(sequence_report.snapshot.evidence_level.value, "window")
        self.assertEqual(sequence_report.snapshot.metadata.get("sequence_length"), 3)
        self.assertGreaterEqual(len(sequence_report.snapshot.metadata.get("history", [])), 3)

    def test_cli_probe_model_and_probe_plasticity(self) -> None:
        from examples.rl_actor_case import export_training_sequence_artifacts

        temp_dir, artifacts = self._prepare_artifacts()
        self.addCleanup(temp_dir.cleanup)

        actor_path = str(artifacts["actor"])
        batch_path = str(artifacts["batch"])
        env = {**os.environ, "PYTHONPATH": self.pythonpath}

        probe_model_process = subprocess.run(
            [
                sys.executable,
                "-m",
                "rlplasticity.cli",
                "probe-model",
                "--builder",
                "examples.rl_actor_case:build_actor",
                "--samples",
                batch_path,
                "--checkpoint",
                actor_path,
                "--forward",
                "examples.rl_actor_case:forward_batch",
            ],
            cwd=self.repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(probe_model_process.returncode, 0, probe_model_process.stderr)
        self.assertIn("kind=model_probe", probe_model_process.stdout)

        probe_plasticity_process = subprocess.run(
            [
                sys.executable,
                "-m",
                "rlplasticity.cli",
                "probe-plasticity",
                "--builder",
                "examples.rl_actor_case:build_actor",
                "--samples",
                batch_path,
                "--loss",
                "examples.rl_actor_case:actor_loss",
                "--optimizer",
                "examples.rl_actor_case:build_optimizer",
                "--checkpoint",
                actor_path,
                "--max-steps",
                "1",
            ],
            cwd=self.repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(probe_plasticity_process.returncode, 0, probe_plasticity_process.stderr)
        self.assertIn("kind=plasticity_probe", probe_plasticity_process.stdout)
        self.assertIn("plasticity score", probe_plasticity_process.stdout.lower())

        with tempfile.TemporaryDirectory() as sequence_dir:
            sequence_artifacts = export_training_sequence_artifacts(sequence_dir, steps=3)
            probe_window_process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rlplasticity.cli",
                    "probe-window",
                    "--builder",
                    "examples.rl_actor_case:build_actor",
                    "--samples",
                    str(sequence_artifacts["batches"]),
                    "--loss",
                    "examples.rl_actor_case:actor_loss",
                    "--optimizer",
                    "examples.rl_actor_case:build_optimizer",
                    "--checkpoint",
                    actor_path,
                    "--max-steps",
                    "2",
                ],
                cwd=self.repo_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(probe_window_process.returncode, 0, probe_window_process.stderr)
            self.assertIn("History", probe_window_process.stdout)

            sequence_process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rlplasticity.cli",
                    "probe-sequence",
                    "--builder",
                    "examples.rl_actor_case:build_actor",
                    "--samples",
                    str(sequence_artifacts["batches"]),
                    "--loss",
                    "examples.rl_actor_case:actor_loss",
                    "--optimizer",
                    "examples.rl_actor_case:build_optimizer",
                    "--checkpoints",
                    *[str(path) for path in sequence_artifacts["checkpoints"]],
                    "--max-steps",
                    "2",
                ],
                cwd=self.repo_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(sequence_process.returncode, 0, sequence_process.stderr)
            self.assertIn("History", sequence_process.stdout)

    def test_showcase_generator_emits_demo_artifacts(self) -> None:
        from examples.showcase_reports import generate_showcase

        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = generate_showcase(temp_dir)
            self.assertTrue(outputs["healthy_probe_json"].exists())
            self.assertTrue(outputs["frozen_probe_json"].exists())
            self.assertTrue(outputs["healthy_probe_html"].exists())
            self.assertTrue(outputs["frozen_probe_html"].exists())

    def test_validation_suite_emits_expected_cases(self) -> None:
        from examples.validation_suite import generate_validation_suite

        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = generate_validation_suite(temp_dir)
            self.assertTrue(outputs["summary_json"].exists())
            self.assertTrue(outputs["readme"].exists())
            summary = outputs["summary_json"].read_text(encoding="utf-8")
            self.assertIn("frozen_encoder", summary)
            self.assertIn("frozen_policy_head", summary)
            self.assertIn("frozen_trunk", summary)
            self.assertIn("global_stall", summary)
            self.assertIn("checkpoint_sequence", summary)


if __name__ == "__main__":
    unittest.main()
