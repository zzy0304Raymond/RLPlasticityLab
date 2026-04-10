"""Minimal low-cost plasticity probe example."""

from __future__ import annotations

import torch
import torch.nn as nn

from rlplasticity import probe_model, probe_plasticity


class TinyActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(obs)
        return self.policy_head(features), self.value_head(features)


def main() -> None:
    model = TinyActorCritic(obs_dim=8, action_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    obs = torch.randn(32, 8)
    targets = torch.randn(32, 4)
    value_targets = torch.randn(32, 1)

    batch = {
        "obs": obs,
        "policy_target": targets,
        "value_target": value_targets,
    }

    def loss_fn(current_model: nn.Module, current_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits, values = current_model(current_batch["obs"])
        policy_loss = (logits - current_batch["policy_target"]).pow(2).mean()
        value_loss = (values - current_batch["value_target"]).pow(2).mean()
        return policy_loss + value_loss

    forward_report = probe_model(model, batch["obs"])
    print(forward_report.to_text())
    print("")

    plasticity_report = probe_plasticity(
        model,
        [batch],
        loss_fn=loss_fn,
        optimizer=optimizer,
        metadata={"algo": "toy-actor-critic"},
    )
    print(plasticity_report.to_text())


if __name__ == "__main__":
    main()
