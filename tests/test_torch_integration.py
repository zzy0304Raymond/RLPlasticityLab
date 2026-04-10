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
        from examples.rl_actor_case import actor_loss, build_actor, build_optimizer, forward_batch
        from rlplasticity import probe_model, probe_plasticity, scan_checkpoint

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

    def test_cli_probe_model_and_probe_plasticity(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
