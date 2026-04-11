"""A realistic demo case for actor checkpoint probing and plasticity diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlplasticity import probe_model, probe_plasticity, scan_checkpoint


OBS_DIM = 8
ACTION_DIM = 4
HIDDEN_DIM = 32
DEFAULT_SEED = 7


class DemoActor(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(obs)
        hidden = self.trunk(features)
        return self.policy_head(hidden)


def build_actor() -> DemoActor:
    torch.manual_seed(DEFAULT_SEED)
    return DemoActor()


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=3e-4)


def make_batch(batch_size: int = 32, *, seed: int = DEFAULT_SEED) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    obs = torch.randn(batch_size, OBS_DIM, generator=generator)
    teacher = torch.tensor(
        [
            [0.8, -0.3, 0.2, 0.1],
            [-0.4, 0.9, 0.2, -0.5],
            [0.3, 0.1, 0.7, -0.2],
            [0.0, -0.2, 0.3, 0.8],
            [0.4, 0.2, -0.6, 0.1],
            [-0.5, 0.3, 0.1, 0.2],
            [0.2, 0.4, -0.3, 0.5],
            [0.1, -0.7, 0.6, 0.2],
        ],
        dtype=torch.float32,
    )
    logits_target = obs @ teacher
    action_target = logits_target.argmax(dim=-1)
    return {
        "obs": obs,
        "target_logits": logits_target,
        "target_actions": action_target,
    }


def forward_batch(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(batch["obs"])


def actor_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    logits = model(batch["obs"])
    cross_entropy = F.cross_entropy(logits, batch["target_actions"])
    imitation = F.mse_loss(logits, batch["target_logits"])
    loss = cross_entropy + 0.05 * imitation
    predicted_actions = logits.argmax(dim=-1)
    accuracy = (predicted_actions == batch["target_actions"]).float().mean().item()
    return loss, {
        "demo_accuracy": accuracy,
        "cross_entropy": float(cross_entropy.detach().item()),
    }


def export_demo_artifacts(
    output_dir: str | Path,
    *,
    batch_size: int = 32,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model = build_actor()
    batch = make_batch(batch_size=batch_size, seed=seed)

    actor_path = output / "actor.pt"
    batch_path = output / "batch.pt"
    torch.save(model.state_dict(), actor_path)
    torch.save(batch, batch_path)

    return {
        "actor": actor_path,
        "batch": batch_path,
    }


def export_training_sequence_artifacts(
    output_dir: str | Path,
    *,
    steps: int = 3,
    batch_size: int = 32,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Export a short sequence of checkpoints produced by light demo training."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model = build_actor()
    optimizer = build_optimizer(model)
    checkpoints: list[Path] = []
    batches: list[dict[str, torch.Tensor]] = []

    for step in range(steps):
        batch = make_batch(batch_size=batch_size, seed=seed + step)
        batches.append(batch)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = actor_loss(model, batch)
        loss.backward()
        optimizer.step()
        checkpoint_path = output / f"actor_step_{step + 1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        checkpoints.append(checkpoint_path)

    batch_path = output / "sequence_batches.pt"
    torch.save(batches, batch_path)
    return {
        "checkpoints": checkpoints,
        "batches": batch_path,
    }


def run_demo(output_dir: str | Path) -> None:
    artifacts = export_demo_artifacts(output_dir)
    actor_path = artifacts["actor"]
    batch_path = artifacts["batch"]

    batch = torch.load(batch_path, map_location="cpu")

    static_report = scan_checkpoint(str(actor_path))
    print("=== Static Scan ===")
    print(static_report.to_text())
    print("")

    forward_report = probe_model(
        build_actor(),
        batch,
        checkpoint=str(actor_path),
        forward_fn=forward_batch,
        metadata={"demo_case": "rl_actor_case"},
    )
    print("=== Forward Probe ===")
    print(forward_report.to_text())
    print("")

    plasticity_model = build_actor()
    plasticity_report = probe_plasticity(
        plasticity_model,
        [batch],
        loss_fn=actor_loss,
        optimizer=build_optimizer(plasticity_model),
        checkpoint=str(actor_path),
        metadata={"demo_case": "rl_actor_case"},
    )
    print("=== Plasticity Probe ===")
    print(plasticity_report.to_text())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate demo RLPlasticity artifacts and optional reports.")
    parser.add_argument("--output-dir", default="reports/demo_rl_actor_case")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run", action="store_true", help="Also run scan/probe/plasticity reports after export.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    artifacts = export_demo_artifacts(args.output_dir, batch_size=args.batch_size, seed=args.seed)
    print(f"actor={artifacts['actor']}")
    print(f"batch={artifacts['batch']}")
    if args.run:
        run_demo(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
