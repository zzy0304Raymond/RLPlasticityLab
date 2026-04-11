"""PPO-style example using the CleanRL-oriented helper."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlplasticity.integrations.cleanrl import probe_cleanrl_agent, probe_cleanrl_window


OBS_DIM = 8
ACTION_DIM = 4


class PPOAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(OBS_DIM, 64),
            nn.Tanh(),
        )
        self.network = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, ACTION_DIM)
        self.critic = nn.Linear(64, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(obs)
        hidden = self.network(features)
        return self.actor(hidden), self.critic(hidden)


def build_agent() -> PPOAgent:
    torch.manual_seed(11)
    return PPOAgent()


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=3e-4)


def make_batch(batch_size: int = 64, *, seed: int = 11) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    obs = torch.randn(batch_size, OBS_DIM, generator=generator)
    teacher = torch.randn(OBS_DIM, ACTION_DIM, generator=generator) * 0.5
    logits_target = obs @ teacher
    actions = logits_target.argmax(dim=-1)
    old_logits = logits_target + 0.05 * torch.randn(logits_target.shape, generator=generator)
    old_log_probs = old_logits.log_softmax(dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    advantages = torch.randn(batch_size, generator=generator).clamp(-2.0, 2.0)
    returns = logits_target.mean(dim=-1, keepdim=True)
    return {
        "obs": obs,
        "actions": actions,
        "advantages": advantages,
        "returns": returns,
        "old_log_probs": old_log_probs,
    }


def ppo_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    logits, values = model(batch["obs"])
    log_probs = logits.log_softmax(dim=-1).gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1)
    ratio = (log_probs - batch["old_log_probs"]).exp()
    clipped_ratio = ratio.clamp(0.8, 1.2)
    policy_loss = -torch.minimum(ratio * batch["advantages"], clipped_ratio * batch["advantages"]).mean()
    value_loss = F.mse_loss(values, batch["returns"])
    entropy = -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
    return loss, {
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
    }


def run_demo() -> None:
    agent = build_agent()
    optimizer = build_optimizer(agent)
    batch = make_batch()

    single_report = probe_cleanrl_agent(
        agent,
        [batch],
        loss_fn=ppo_loss,
        optimizer=optimizer,
        metadata={"algo": "ppo-like"},
    )
    print("=== PPO-Like Single Probe ===")
    print(single_report.to_text())
    print("")

    window_agent = build_agent()
    window_optimizer = build_optimizer(window_agent)
    window_batches = [make_batch(seed=11 + offset) for offset in range(4)]
    window_report = probe_cleanrl_window(
        window_agent,
        window_batches,
        loss_fn=ppo_loss,
        optimizer=window_optimizer,
        max_steps=4,
        metadata={"algo": "ppo-like", "view": "window"},
    )
    print("=== PPO-Like Window Probe ===")
    print(window_report.to_text())


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Run a PPO-like RLPlasticity example.")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    parser.parse_args(argv)
    run_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
