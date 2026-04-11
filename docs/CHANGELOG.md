# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog and the project follows semantic-version style tags once formal releases start.

## [0.1.0] - 2026-04-10

### Added

- public workflows for `scan_checkpoint`, `probe_model`, and `probe_plasticity`
- public workflows for `probe_plasticity_window` and `probe_checkpoint_sequence`
- convenience entry points for builder-based probing and single training-step probing
- static checkpoint scanning for bare `.pt` and `state_dict` inputs
- forward-only activation probes for loadable PyTorch models
- low-cost update-level plasticity probes for PyTorch models
- trend-aware history metadata for batch windows and checkpoint sequences
- rule-based findings for:
  - global plasticity stall
  - encoder bottleneck
  - trunk bottleneck
  - head saturation
  - plasticity decline trend
  - forward low-response conditions
- text, JSON, and HTML report rendering
- realistic demo actor case, sequence artifacts, and generated showcase artifacts
- PPO-like, DQN-like, and SAC-like example scripts for common RL wiring patterns
- validation suite covering healthy and intentionally broken cases, including trunk-side failure and checkpoint-sequence history
- lightweight integration helpers for:
  - raw PyTorch loops
  - CleanRL-style naming
  - SB3-style naming
- more robust module-group inference to avoid false positives such as `actor` matching `extractor`
- report schema, troubleshooting guide, and roadmap docs
- community files:
  - contributing guide
  - code of conduct
  - security policy
  - issue templates
  - pull request template
- baseline CI workflow for tests

### Notes

- The first release is PyTorch-first and offline-first.
- The project intentionally focuses on model plasticity only.
- Reward diagnosis and online monitoring are out of scope for `0.1.0`.
