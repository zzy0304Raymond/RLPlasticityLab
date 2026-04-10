# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog and the project follows semantic-version style tags once formal releases start.

## [0.1.0] - 2026-04-10

### Added

- public workflows for `scan_checkpoint`, `probe_model`, and `probe_plasticity`
- static checkpoint scanning for bare `.pt` and `state_dict` inputs
- forward-only activation probes for loadable PyTorch models
- low-cost update-level plasticity probes for PyTorch models
- rule-based findings for:
  - global plasticity stall
  - encoder bottleneck
  - head saturation
  - forward low-response conditions
- text, JSON, and HTML report rendering
- realistic demo actor case and generated showcase artifacts
- validation suite covering healthy and intentionally broken cases
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
