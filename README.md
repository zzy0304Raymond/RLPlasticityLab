# RLPlasticity

`RLPlasticity` is a low-cost diagnostics toolkit for answering one question quickly:

**Has my RL model lost the capacity to adapt, and where does that loss seem to concentrate?**

This repository is not trying to replace full experiment validation. It is built for fast triage: use a checkpoint, a model, or a small amount of replay data to narrow the search space before you spend time on expensive ablations or full retraining runs.

## Project Boundary

`RLPlasticity` focuses on **model plasticity only**.

It does not try to decide whether a failure is caused by reward design, exploration, data distribution, or environment bugs. Those questions are valuable, but they require broader task context and would make early diagnoses too speculative.

The first public release answers a narrower set of questions:

- Does this checkpoint show suspicious structural signs?
- Does this model respond weakly on forward passes?
- Does this model still produce meaningful gradient flow and updates on a short probe window?
- Does the issue look global, encoder-side, or head-side?

## User Scenarios

The toolkit is organized around three evidence levels.

### 1. `scan_checkpoint`
Input:
- `.pt` checkpoint or in-memory `state_dict`

Service:
- Static structural scan
- Parameter norm and sparsity statistics
- Grouping into `encoder / trunk / policy / value`
- Low-confidence structural hints

What it cannot do:
- It cannot measure gradients, updates, or true plasticity loss

### 2. `probe_model`
Input:
- Loadable model
- One sample batch
- Optional checkpoint to load

Service:
- Forward-only probe
- Activation health and low-response detection
- Weak evidence about whether some modules look inactive

What it cannot do:
- It still cannot prove plasticity loss under optimization

### 3. `probe_plasticity`
Input:
- Loadable model
- One or more batches
- Loss function
- Optimizer
- Optional checkpoint to load

Service:
- Low-cost update probe
- Gradient reachability, update effectiveness, activation shift
- Rule-based plasticity findings
- Global, encoder-side, and head-side bottleneck hints

This is the main value path for the repository.

## Package Layout

```text
src/rlplasticity/
  core/          # shared schemas, enums, aggregation, base interfaces
  ingest/        # checkpoint loading and static summarization
  probes/        # evidence collection workflows
  plasticity/    # metrics, findings, analyzers
  reporting/     # text/html rendering
  cli.py         # command-line entry point
  api.py         # public Python workflows
```

## Install

Editable install:

```bash
pip install -e .
```

Install with PyTorch support:

```bash
pip install -e ".[torch]"
```

## Quick Start

### Static scan with only a checkpoint

```python
from rlplasticity import scan_checkpoint

state_dict = {
    "encoder.weight": [[0.0, 0.0], [0.0, 0.0]],
    "policy.bias": [1.0, -1.0],
}

report = scan_checkpoint(state_dict)
print(report.to_text())
```

### Low-cost plasticity probe

```python
import torch
import torch.nn as nn

from rlplasticity import probe_plasticity


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU())
        self.policy_head = nn.Linear(32, 4)

    def forward(self, obs):
        return self.policy_head(self.encoder(obs))


def loss_fn(model, batch):
    logits = model(batch["obs"])
    return (logits - batch["target"]).pow(2).mean()


model = TinyActor()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
batch = {"obs": torch.randn(32, 8), "target": torch.randn(32, 4)}

report = probe_plasticity(
    model,
    [batch],
    loss_fn=loss_fn,
    optimizer=optimizer,
)

print(report.to_text())
```

See [examples/checkpoint_scan.py](/C:/Users/22050/Desktop/RLPlasticityLab/examples/checkpoint_scan.py) and [examples/minimal_torch_integration.py](/C:/Users/22050/Desktop/RLPlasticityLab/examples/minimal_torch_integration.py).

### Realistic actor checkpoint demo

The repository also ships an importable demo case with:

- `build_actor`
- `build_optimizer`
- `forward_batch`
- `actor_loss`
- `export_demo_artifacts`

Generate a reusable `actor.pt` and `batch.pt` pair:

```bash
python -m examples.rl_actor_case --output-dir reports/demo_rl_actor_case
```

Run the full demo:

```bash
python -m examples.rl_actor_case --output-dir reports/demo_rl_actor_case --run
```

Then probe those artifacts through the CLI:

```bash
rlplasticity probe-model \
  --builder examples.rl_actor_case:build_actor \
  --samples reports/demo_rl_actor_case/batch.pt \
  --checkpoint reports/demo_rl_actor_case/actor.pt \
  --forward examples.rl_actor_case:forward_batch
```

```bash
rlplasticity probe-plasticity \
  --builder examples.rl_actor_case:build_actor \
  --samples reports/demo_rl_actor_case/batch.pt \
  --loss examples.rl_actor_case:actor_loss \
  --optimizer examples.rl_actor_case:build_optimizer \
  --checkpoint reports/demo_rl_actor_case/actor.pt \
  --max-steps 1
```

## CLI

Static scan:

```bash
rlplasticity scan --checkpoint actor.pt
```

Forward probe:

```bash
rlplasticity probe-model \
  --builder mypkg.models:build_actor \
  --samples batch.pt \
  --checkpoint actor.pt
```

Plasticity probe:

```bash
rlplasticity probe-plasticity \
  --builder mypkg.models:build_actor \
  --samples batch.pt \
  --loss mypkg.losses:actor_loss \
  --checkpoint actor.pt \
  --max-steps 8
```

## Reports

All workflows can render to:

- text
- HTML
- JSON via `report.to_dict()`

Each report includes:

- analysis mode
- evidence strength
- findings
- caveats
- a layer shortlist for follow-up inspection

## Maturity Of This Release

The first usable release is intentionally narrow:

- PyTorch-first
- Offline-first
- Short-window probes instead of long experiments
- Rule-based findings instead of learned diagnoses

That makes it cheap to run and easier to trust.

## Roadmap

Planned next:

- checkpoint sequence analysis
- stronger replay-window aggregation
- CleanRL / SB3 integration helpers
- richer encoder/trunk/head grouping configuration
- online periodic monitoring

## Development

Run tests with the standard library test runner:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The current test suite focuses on:

- analyzer logic
- report serialization
- CLI bootstrapping
- optional PyTorch API/CLI integration when `torch` is available

## Non-Goals For v0.1

- full RL training orchestration
- reward diagnosis
- environment debugging
- all-purpose explainability tooling
