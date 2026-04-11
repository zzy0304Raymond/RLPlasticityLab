# RLPlasticity

Low-cost plasticity diagnostics for reinforcement learning models.

`RLPlasticity` helps you answer a practical question before you spend time on expensive reruns or ablations:

**Does this RL model still look capable of adapting, and where does a plasticity problem seem to concentrate?**

This project is built for fast triage, not experiment replacement. It is most useful when training gets weird and you want a cheap first pass before touching reward design, environment code, or large training jobs.

Technical Report: [GitHub Pages report](docs/index.md)

## How Most Users Start

Clone the repository, create a virtual environment, install with the PyTorch extra, and run the bundled demo:

```bash
git clone https://github.com/zzy0304Raymond/RLPlasticityLab.git
cd RLPlasticityLab
python -m venv .venv
```

PowerShell:

```bash
.venv\Scripts\Activate.ps1
pip install -e ".[torch]"
python -m examples.rl_actor_case --output-dir reports/demo_rl_actor_case --run
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -e ".[torch]"
python -m examples.rl_actor_case --output-dir reports/demo_rl_actor_case --run
```

If you only want to scan a checkpoint:

```bash
rlplasticity scan --checkpoint actor.pt
```

## What It Looks Like In Practice

Healthy checkpoint vs a checkpoint with a frozen encoder:

Healthy plasticity probe output:

```text
RLPlasticity Report | analyzer=plasticity/default | kind=plasticity_probe | evidence=update | layers=3
loss=1.448104
summary=No acute issue detected. Plasticity score=1.000 (update-active=1.000, grad-active=1.000)

Findings
- No diagnostic rule fired.
```

Frozen-encoder plasticity probe output:

```text
RLPlasticity Report | analyzer=plasticity/default | kind=plasticity_probe | evidence=update | layers=3
loss=1.448104
summary=The encoder is adapting less than downstream layers.

Findings
- [medium/medium] The encoder is adapting less than downstream layers.
  evidence: encoder_plasticity_score=0.000
  evidence: reference_downstream_score=1.000
```

Real generated artifacts:

- Healthy plasticity text: [docs/showcase/healthy_plasticity_probe.txt](docs/showcase/healthy_plasticity_probe.txt)
- Healthy plasticity JSON: [docs/showcase/healthy_plasticity_probe.json](docs/showcase/healthy_plasticity_probe.json)
- Healthy plasticity HTML: [docs/showcase/healthy_plasticity_probe.html](docs/showcase/healthy_plasticity_probe.html)
- Frozen encoder text: [docs/showcase/frozen_encoder_plasticity_probe.txt](docs/showcase/frozen_encoder_plasticity_probe.txt)
- Frozen encoder JSON: [docs/showcase/frozen_encoder_plasticity_probe.json](docs/showcase/frozen_encoder_plasticity_probe.json)
- Frozen encoder HTML: [docs/showcase/frozen_encoder_plasticity_probe.html](docs/showcase/frozen_encoder_plasticity_probe.html)

Example JSON excerpt:

```json
{
  "analyzer_name": "plasticity/default",
  "snapshot": {
    "kind": "plasticity_probe",
    "evidence_level": "update",
    "loss": 1.448103904724121
  },
  "metrics": {
    "plasticity_score": {
      "value": 1.0,
      "summary": "Plasticity score=1.000 (update-active=1.000, grad-active=1.000)"
    },
    "encoder_plasticity_score": {
      "value": 1.0,
      "summary": "encoder plasticity score=1.000"
    }
  },
  "findings": []
}
```

## What This Project Is For

Use `RLPlasticity` when you want to inspect:

- a single `actor.pt` or `policy.pt`
- a checkpoint plus model code
- a checkpoint plus model plus a small batch of replay or environment samples

The toolkit is designed to help with questions like:

- Does this checkpoint look structurally suspicious?
- Is the model responding normally on real inputs?
- Are gradients and updates still reaching the encoder, trunk, and heads?
- Does the issue look global, encoder-side, or head-side?

## What This Project Is Not

This first release does **not** try to decide whether a failure comes from:

- reward design
- exploration
- data collection
- environment bugs
- training orchestration issues outside the model

It focuses only on **model plasticity** and reports evidence with explicit caveats.

## User Scenarios

`RLPlasticity` supports three progressively stronger workflows.

### 1. `scan_checkpoint`

You have:

- only a checkpoint or `state_dict`

You get:

- parameter norm and sparsity statistics
- `encoder / trunk / policy / value` grouping
- low-confidence structural hints

Best for:

- "I only have `actor.pt`; is anything obviously strange?"

### 2. `probe_model`

You have:

- a loadable model
- one batch of samples
- optionally a checkpoint to load

You get:

- forward-only activation health
- low-response hints
- a cheap sanity check before running update probes

Best for:

- "Does this model even respond normally on real observations?"

### 3. `probe_plasticity`

You have:

- a loadable model
- one or more batches
- a loss function
- an optimizer
- optionally a checkpoint to load

You get:

- gradient reachability
- relative update strength
- stagnant-layer statistics
- encoder/trunk/head plasticity hints

Best for:

- "Is this checkpoint still learning, or has part of the model gone stale?"

## Installation

Install the package:

```bash
pip install -e .
```

Install with PyTorch support:

```bash
pip install -e ".[torch]"
```

## Public Entry Points

The project currently exposes five main Python workflows:

- `scan_checkpoint(...)`
- `probe_model(...)`
- `probe_plasticity(...)`
- `probe_plasticity_window(...)`
- `probe_checkpoint_sequence(...)`

Convenience helpers are also available for:

- builder-based probing
  - `probe_model_from_builder(...)`
  - `probe_plasticity_from_builder(...)`
- single training-step probing
  - `probe_training_step(...)`
- light integration helpers
  - `rlplasticity.integrations.pytorch`
  - `rlplasticity.integrations.cleanrl`
  - `rlplasticity.integrations.sb3`

## Quick Start

### A. Analyze a checkpoint directly

```python
from rlplasticity import scan_checkpoint

report = scan_checkpoint("actor.pt")
print(report.to_text())
```

### D. Run a short multi-batch plasticity window

```python
from rlplasticity import probe_plasticity_window

report = probe_plasticity_window(
    model,
    batches,
    loss_fn=loss_fn,
    optimizer=optimizer,
    max_steps=8,
)

print(report.to_text())
```

### E. Compare a checkpoint sequence

```python
from rlplasticity import probe_checkpoint_sequence

report = probe_checkpoint_sequence(
    build_model,
    checkpoints,
    batches,
    loss_fn=loss_fn,
    optimizer_builder=build_optimizer,
    max_steps=4,
)

print(report.to_text())
```

### B. Run a forward-only probe

```python
from rlplasticity import probe_model

report = probe_model(model, batch_obs)
print(report.to_text())
```

### C. Run a low-cost plasticity probe

```python
from rlplasticity import probe_plasticity

report = probe_plasticity(
    model,
    [batch],
    loss_fn=loss_fn,
    optimizer=optimizer,
)

print(report.to_text())
```

## CLI Usage

Static checkpoint scan:

```bash
rlplasticity scan --checkpoint actor.pt
```

Forward-only probe:

```bash
rlplasticity probe-model \
  --builder mypkg.models:build_actor \
  --samples batch.pt \
  --checkpoint actor.pt \
  --forward mypkg.probes:forward_batch
```

Plasticity probe:

```bash
rlplasticity probe-plasticity \
  --builder mypkg.models:build_actor \
  --samples batch.pt \
  --loss mypkg.losses:actor_loss \
  --optimizer mypkg.optim:build_optimizer \
  --checkpoint actor.pt \
  --max-steps 8
```

Plasticity window:

```bash
rlplasticity probe-window \
  --builder mypkg.models:build_actor \
  --samples replay_window.pt \
  --loss mypkg.losses:actor_loss \
  --optimizer mypkg.optim:build_optimizer \
  --checkpoint actor.pt \
  --max-steps 8
```

Checkpoint sequence:

```bash
rlplasticity probe-sequence \
  --builder mypkg.models:build_actor \
  --samples replay_window.pt \
  --loss mypkg.losses:actor_loss \
  --optimizer mypkg.optim:build_optimizer \
  --checkpoints ckpt_10.pt ckpt_20.pt ckpt_30.pt \
  --max-steps 4
```

## Real Example In This Repo

This repository ships a demo actor case with:

- `build_actor`
- `build_optimizer`
- `forward_batch`
- `actor_loss`
- `export_demo_artifacts`
- `export_training_sequence_artifacts`

Generate a reusable demo checkpoint and batch:

```bash
python -m examples.rl_actor_case --output-dir reports/demo_rl_actor_case
```

Run the demo end-to-end:

```bash
python -m examples.rl_actor_case --output-dir reports/demo_rl_actor_case --run
```

Or probe the generated files through the CLI:

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

Generate the homepage showcase artifacts:

```bash
python -m examples.showcase_reports --output-dir docs/showcase
```

Generate the release validation suite:

```bash
python -m examples.validation_suite --output-dir docs/validation
```

Generate a short checkpoint sequence demo:

```bash
python - <<'PY'
from examples.rl_actor_case import export_training_sequence_artifacts
print(export_training_sequence_artifacts("reports/demo_sequence"))
PY
```

Common RL-shaped examples:

- PPO-like actor-critic: `python -m examples.ppo_like_case`
- DQN-like value network: `python -m examples.dqn_like_case`
- SAC-like actor/critic with checkpoint sequence: `python -m examples.sac_like_case --output-dir reports/demo_sac_sequence`

These examples are framework-independent PyTorch scripts, but they are structured to resemble real PPO / DQN / SAC training code and show how to plug in the helper APIs.

## What You Get Back

Each report includes:

- analysis mode
- evidence strength
- key metrics
- findings
- caveats
- optional history rows for windows and checkpoint sequences
- a shortlist of layers worth checking next

Reports can be rendered as:

- text
- HTML
- JSON via `report.to_dict()`

## Repository Layout

```text
src/rlplasticity/
  core/          # shared schemas, enums, aggregation, base interfaces
  ingest/        # checkpoint loading and static summarization
  integrations/  # lightweight helpers for raw PyTorch, CleanRL, and SB3-style setups
  probes/        # evidence collection workflows
  plasticity/    # metrics, rules, analyzers
  reporting/     # text/html rendering
  api.py         # public Python workflows
  cli.py         # command-line entry point
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the framework design.

Release-oriented docs:

- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Validation: [docs/VALIDATION.md](docs/VALIDATION.md)
- Report schema: [docs/REPORT_SCHEMA.md](docs/REPORT_SCHEMA.md)
- Troubleshooting: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- Roadmap: [docs/ROADMAP.md](docs/ROADMAP.md)
- Changelog: [docs/CHANGELOG.md](docs/CHANGELOG.md)
- Release checklist: [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md)
- Release report: [docs/RELEASE_REPORT_v0.1.0.md](docs/RELEASE_REPORT_v0.1.0.md)

## Open Source Standards

This repository currently uses:

- License: MIT, see [LICENSE](LICENSE)
- Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Security policy: [SECURITY.md](SECURITY.md)

## Security Notes

This project currently works with PyTorch checkpoints and `torch.load(...)`.

That means:

- only load checkpoints you trust
- treat untrusted model files as potentially unsafe
- prefer isolated environments when inspecting third-party artifacts

See [SECURITY.md](SECURITY.md) for the project policy.

## Development

Run the test suite in PowerShell:

```bash
$env:PYTHONPATH="src;."
python -m unittest discover -s tests -v
```

Run the test suite in bash:

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -v
```

Current coverage includes:

- analyzer logic
- CLI behavior
- report serialization
- robustness edge cases
- trend-aware window and checkpoint-sequence probing
- optional PyTorch API and CLI integration

## Roadmap

Planned next:

- more architecture families beyond the bundled demo actor
- stronger trainer-specific helper coverage
- lower-overhead online monitoring
