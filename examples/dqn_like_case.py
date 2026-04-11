"""DQN-style example using a value-only network."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlplasticity.integrations.cleanrl import probe_cleanrl_agent, probe_cleanrl_window


OBS_DIM = 10
ACTION_DIM = 5


class DQNQNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(OBS_DIM, 64),
            nn.ReLU(),
        )
        self.q_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_DIM),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.features(obs)
        return self.q_net(hidden)


def build_q_network() -> DQNQNetwork:
    torch.manual_seed(23)
    return DQNQNetwork()


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=1e-3)


def make_batch(batch_size: int = 64, *, seed: int = 23) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    obs = torch.randn(batch_size, OBS_DIM, generator=generator)
    next_obs = torch.randn(batch_size, OBS_DIM, generator=generator)
    actions = torch.randint(0, ACTION_DIM, (batch_size,), generator=generator)
    rewards = torch.randn(batch_size, generator=generator).clamp(-1.0, 1.0)
    dones = torch.randint(0, 2, (batch_size,), generator=generator).float()
    target_projection = torch.randn(OBS_DIM, ACTION_DIM, generator=generator) * 0.3
    bootstrap_q = (next_obs @ target_projection).max(dim=-1).values
    td_target = rewards + 0.99 * (1.0 - dones) * bootstrap_q
    return {
        "obs": obs,
        "actions": actions,
        "td_target": td_target,
    }


def dqn_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    q_values = model(batch["obs"])
    chosen_q = q_values.gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1)
    loss = F.smooth_l1_loss(chosen_q, batch["td_target"])
    return loss, {"td_error": float((chosen_q - batch["td_target"]).abs().mean().detach().item())}


def run_demo() -> None:
    q_network = build_q_network()
    optimizer = build_optimizer(q_network)
    batch = make_batch()

    report = probe_cleanrl_agent(
        q_network,
        [batch],
        loss_fn=dqn_loss,
        optimizer=optimizer,
        metadata={"algo": "dqn-like"},
    )
    print("=== DQN-Like Single Probe ===")
    print(report.to_text())
    print("")

    window_network = build_q_network()
    window_optimizer = build_optimizer(window_network)
    window_batches = [make_batch(seed=23 + offset) for offset in range(4)]
    window_report = probe_cleanrl_window(
        window_network,
        window_batches,
        loss_fn=dqn_loss,
        optimizer=window_optimizer,
        max_steps=4,
        metadata={"algo": "dqn-like", "view": "window"},
    )
    print("=== DQN-Like Window Probe ===")
    print(window_report.to_text())


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Run a DQN-like RLPlasticity example.")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    parser.parse_args(argv)
    run_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
