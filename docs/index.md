---
title: RLPlasticity Technical Report
---

# RLPlasticity Technical Report

Repository: [RLPlasticityLab](https://github.com/zzy0304Raymond/RLPlasticityLab)  
Project positioning: low-cost plasticity diagnostics for reinforcement learning models

## Motivation

When RL training degrades, the first question is often not "how do I fix everything?" but "where should I look first?" Running full ablations, retraining from scratch, or testing many reward and architecture variants is expensive. `RLPlasticity` is designed as a cheaper first-pass diagnostic layer for model-side plasticity issues.

The project is intentionally narrow. It does not try to replace experiments or explain every RL failure mode. Instead, it helps answer whether a checkpoint still appears capable of adapting, and whether the apparent problem concentrates in the encoder, the shared trunk, or the output heads.

## What Plasticity Means in RL

In this repository, plasticity refers to a model's ability to continue changing usefully under optimization.

For RL models, that often shows up through signals such as:

- gradients continuing to reach important modules
- parameter updates remaining non-trivial relative to parameter scale
- representations still changing under relevant data
- different parts of the model not becoming stale at different rates

This is not the same as final task performance. A policy can perform poorly while still being plastic, and a checkpoint can keep a decent score while already losing the ability to adapt further.

## Problem Statement

The project focuses on a practical debugging problem:

> Given a checkpoint, model code, and possibly a small amount of sample data, can we cheaply rank whether a model-side plasticity issue is worth investigating before running expensive experiments?

The first release is explicitly scoped to model plasticity only. It does **not** claim to identify reward-design failures, exploration failures, environment bugs, or orchestration issues outside the model.

## Framework Overview

`RLPlasticity` is organized around three progressively stronger core diagnostic workflows:

1. `scan_checkpoint`
2. `probe_model`
3. `probe_plasticity`

On top of these, the current release also adds two history-aware workflows:

- `probe_plasticity_window`
- `probe_checkpoint_sequence`

These workflows share a common report structure so that results remain comparable across different evidence levels:

- `kind`
- `evidence_level`
- per-layer snapshots
- aggregated metrics
- ranked findings
- caveats

The current implementation is PyTorch-first and offline-first. It is meant to sit next to a training workflow, not replace one.

## Diagnostic Level 1: `scan_checkpoint`

`scan_checkpoint` is the lowest-cost entry point. It works when you only have a `.pt` file or a `state_dict`.

Typical output includes:

- parameter norms
- sparsity and zero-fraction summaries
- heuristic grouping into `encoder`, `trunk`, `policy`, and `value`
- low-confidence structural hints

This level is useful when you want to ask:

- does anything look obviously malformed?
- do some layers have suspiciously tiny norms?
- is the checkpoint structure what I expect?

What it cannot do:

- measure gradient flow
- measure update effectiveness
- establish true plasticity loss

## Diagnostic Level 2: `probe_model`

`probe_model` uses a loadable model plus one batch of samples. It runs a forward-only probe.

Typical output includes:

- activation mean and standard deviation
- low-variation module hints
- forward-response health summaries

This level is useful when you want to ask:

- does the model respond normally on representative observations?
- are some modules nearly silent even before optimization?

What it still cannot do:

- determine whether optimization is still effective
- distinguish "responsive but no longer learning" from healthy adaptation

## Diagnostic Level 3: `probe_plasticity`

`probe_plasticity` is the main workflow in this release. It uses a model, one or more batches, a loss function, and an optimizer to run a low-cost update-level probe.

Typical output includes:

- gradient reachability
- relative update size
- stagnant-layer fraction
- group-level plasticity scores
- rule-based findings such as `encoder_bottleneck`, `head_saturation`, and `global_plasticity_stall`

This level is useful when you want to ask:

- is this checkpoint still changing meaningfully under optimization?
- does the issue look global or localized?
- should I inspect encoder, trunk, or output heads first?

### History-Aware Workflows

The repository also supports:

- `probe_plasticity_window`
  - averages several update probes across a short batch window and preserves compact history rows
- `probe_checkpoint_sequence`
  - compares several checkpoints under the same probe recipe and keeps checkpoint-by-checkpoint history metadata

These workflows are useful when you care about trend, not just a single point estimate.

## Example Workflow

### Python API

```python
from rlplasticity import probe_plasticity

report = probe_plasticity(
    model,
    [batch],
    loss_fn=loss_fn,
    optimizer=optimizer,
    checkpoint="actor.pt",
)

print(report.to_text())
```

### CLI

```bash
rlplasticity probe-plasticity \
  --builder examples.rl_actor_case:build_actor \
  --samples reports/demo_rl_actor_case/batch.pt \
  --loss examples.rl_actor_case:actor_loss \
  --optimizer examples.rl_actor_case:build_optimizer \
  --checkpoint reports/demo_rl_actor_case/actor.pt \
  --max-steps 1
```

The repository also includes ready-to-run demo and validation generators:

- Showcase artifacts: [docs/showcase](showcase)
- Validation suite: [docs/validation](validation)
- Sequence artifacts are generated by the bundled example utilities

## Example Outputs

The excerpts below are drawn from repository-generated demo artifacts. They are illustrative examples from bundled cases, not benchmark claims.

### Healthy demo case

```text
RLPlasticity Report | analyzer=plasticity/default | kind=plasticity_probe | evidence=update | layers=3
loss=1.448104
summary=No acute issue detected. Plasticity score=1.000 (update-active=1.000, grad-active=1.000)
```

Reference artifacts:

- [Healthy text report](showcase/healthy_plasticity_probe.txt)
- [Healthy JSON report](showcase/healthy_plasticity_probe.json)
- [Healthy HTML report](showcase/healthy_plasticity_probe.html)

### Frozen-encoder demo case

```text
RLPlasticity Report | analyzer=plasticity/default | kind=plasticity_probe | evidence=update | layers=3
loss=1.448104
summary=The encoder is adapting less than downstream layers.
```

Reference artifacts:

- [Frozen encoder text report](showcase/frozen_encoder_plasticity_probe.txt)
- [Frozen encoder JSON report](showcase/frozen_encoder_plasticity_probe.json)
- [Frozen encoder HTML report](showcase/frozen_encoder_plasticity_probe.html)

### Validation cases included in the repo

The release validation set currently checks:

- `healthy`
- `frozen_encoder`
- `frozen_trunk`
- `frozen_policy_head`
- `global_stall`
- `checkpoint_sequence`

Validation summary:

- [Validation overview](validation/README.md)
- [Validation summary JSON](validation/summary.json)
- [Validation notes](VALIDATION.md)

## Limitations

The current release has several deliberate limitations:

- PyTorch-first only
- offline-first only
- heuristic, rule-based findings rather than calibrated statistical diagnoses
- no reward-side diagnosis
- no online monitoring yet
- no claim of benchmark superiority or complete failure attribution

Results should be interpreted as ranked evidence for where to investigate next, not as a replacement for controlled experiments.

## Figures and Assets

If you want to add diagrams or screenshots later, place them under [docs/assets](assets/README.md).

Current asset placeholders:

- [docs/assets/README.md](assets/README.md)

## Related Project Documents

- [Architecture notes](ARCHITECTURE.md)
- [Report schema](REPORT_SCHEMA.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [Roadmap](ROADMAP.md)
- [Changelog](CHANGELOG.md)
- [Release checklist](RELEASE_CHECKLIST.md)
- [Release report for v0.1.0](RELEASE_REPORT_v0.1.0.md)

## GitHub Repository

- Source code: [github.com/zzy0304Raymond/RLPlasticityLab](https://github.com/zzy0304Raymond/RLPlasticityLab)
- Project README: [README.md on GitHub](https://github.com/zzy0304Raymond/RLPlasticityLab/blob/main/README.md)
