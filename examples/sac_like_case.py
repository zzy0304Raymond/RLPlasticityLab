"""SAC-style example using an SB3-like policy structure."""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlplasticity import probe_checkpoint_sequence
from rlplasticity.integrations.sb3 import probe_sb3_policy


OBS_DIM = 6
ACTION_DIM = 3


class SACLikePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features_extractor = nn.Sequential(
            nn.Linear(OBS_DIM, 64),
            nn.ReLU(),
        )
        self.mlp_extractor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.action_net = nn.Linear(64, ACTION_DIM)
        self.value_net = nn.Linear(64, 1)
        self.qf1 = nn.Linear(64, 1)
        self.qf2 = nn.Linear(64, 1)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.features_extractor(obs)
        latent = self.mlp_extractor(features)
        return {
            "action_logits": self.action_net(latent),
            "value": self.value_net(latent),
            "q1": self.qf1(latent),
            "q2": self.qf2(latent),
        }


def build_policy() -> SACLikePolicy:
    torch.manual_seed(31)
    return SACLikePolicy()


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=3e-4)


def make_batch(batch_size: int = 64, *, seed: int = 31) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    obs = torch.randn(batch_size, OBS_DIM, generator=generator)
    logits_target = obs @ torch.randn(OBS_DIM, ACTION_DIM, generator=generator) * 0.4
    target_actions = logits_target.argmax(dim=-1)
    q_target = logits_target.sum(dim=-1, keepdim=True) * 0.2
    return {
        "obs": obs,
        "target_actions": target_actions,
        "q_target": q_target,
    }


def sac_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = model(batch["obs"])
    action_logits = outputs["action_logits"]
    action_loss = F.cross_entropy(action_logits, batch["target_actions"])
    q1_loss = F.mse_loss(outputs["q1"], batch["q_target"])
    q2_loss = F.mse_loss(outputs["q2"], batch["q_target"])
    value_loss = F.mse_loss(outputs["value"], batch["q_target"])
    entropy = -(action_logits.softmax(dim=-1) * action_logits.log_softmax(dim=-1)).sum(dim=-1).mean()
    loss = action_loss + q1_loss + q2_loss + 0.5 * value_loss - 0.02 * entropy
    return loss, {
        "actor_loss": float(action_loss.detach().item()),
        "critic_loss": float((q1_loss + q2_loss).detach().item()),
        "entropy": float(entropy.detach().item()),
    }


def export_sequence_artifacts(output_dir: str, *, steps: int = 3) -> tuple[list[str], list[dict[str, torch.Tensor]]]:
    import os

    os.makedirs(output_dir, exist_ok=True)
    model = build_policy()
    optimizer = build_optimizer(model)
    checkpoints: list[str] = []
    batches: list[dict[str, torch.Tensor]] = []
    for offset in range(steps):
        batch = make_batch(seed=31 + offset)
        batches.append(batch)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = sac_loss(model, batch)
        loss.backward()
        optimizer.step()
        checkpoint_path = os.path.join(output_dir, f"sac_like_step_{offset + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        checkpoints.append(checkpoint_path)
    return checkpoints, batches


def run_demo(output_dir: str = "reports/demo_sac_sequence") -> None:
    policy = build_policy()
    optimizer = build_optimizer(policy)
    batch = make_batch()

    report = probe_sb3_policy(
        policy,
        [batch],
        loss_fn=sac_loss,
        optimizer=optimizer,
        metadata={"algo": "sac-like"},
    )
    print("=== SAC-Like Single Probe ===")
    print(report.to_text())
    print("")

    sequence_checkpoints, sequence_batches = export_sequence_artifacts(output_dir)
    sequence_report = probe_checkpoint_sequence(
        build_policy,
        sequence_checkpoints,
        sequence_batches,
        loss_fn=sac_loss,
        optimizer_builder=build_optimizer,
        max_steps=2,
        metadata={"algo": "sac-like", "view": "sequence"},
    )
    print("=== SAC-Like Checkpoint Sequence Probe ===")
    print(sequence_report.to_text())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a SAC-like RLPlasticity example.")
    parser.add_argument("--output-dir", default="reports/demo_sac_sequence")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_demo(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
