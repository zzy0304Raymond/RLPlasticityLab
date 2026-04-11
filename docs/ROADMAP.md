# Roadmap

This roadmap reflects the current positioning of `RLPlasticity` as a low-cost plasticity diagnostics toolkit for RL workflows.

## Current Release Focus

The current release candidate emphasizes:

- offline checkpoint and runtime probes
- PyTorch-first workflows
- low-cost plasticity triage rather than full experimental replacement

## Near-Term Next Steps

- broaden architecture coverage beyond the bundled demo actor
- add more synthetic and semi-realistic failure cases
- improve trainer-specific helper coverage for common RL codebases
- make history and trend views easier to compare across runs

## Medium-Term Direction

- lower-overhead periodic monitoring during training
- richer checkpoint-sequence comparison utilities
- better report export hooks for experiment dashboards
- additional documentation for integrating with replay buffers and trainer loops

## Explicit Non-Goals For Now

- reward diagnosis
- environment debugging
- claims of benchmark superiority
- replacing controlled experiments with automated root-cause attribution
