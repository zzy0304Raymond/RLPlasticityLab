# Release Report: v0.1.0

## Scope

`v0.1.0` is the first public release candidate of `RLPlasticity`.

The release focuses on one narrow but practical problem:

**low-cost offline diagnosis of model plasticity issues in RL workflows**

## Included Workflows

- `scan_checkpoint`
- `probe_model`
- `probe_plasticity`

## Intended User Value

This release aims to reduce the cost of answering:

- does this checkpoint still look trainable?
- does the issue seem global or localized?
- should I investigate encoder, trunk, or head first?

## Validation Summary

The current validation set covers:

- healthy baseline
- frozen encoder
- frozen policy head
- global stall

Artifacts:

- [docs/validation/README.md](validation/README.md)
- [docs/validation/summary.json](validation/summary.json)

## Engineering Readiness

The release includes:

- public Python APIs
- CLI entry points
- tests for logic, robustness, and PyTorch integration
- community files and issue templates
- changelog
- release checklist

## Known Limitations

- PyTorch-first only
- offline-first only
- heuristic findings rather than calibrated statistical diagnoses
- no reward-side diagnosis
- no online monitoring yet

## Next Steps After Release

- add more architectures and RL algorithm examples
- add checkpoint sequence and trend analysis
- add integration helpers for common training frameworks
