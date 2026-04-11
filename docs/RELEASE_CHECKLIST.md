# Release Checklist

## Pre-Release

- update version metadata if needed
- update [docs/CHANGELOG.md](CHANGELOG.md)
- regenerate showcase artifacts
- regenerate validation artifacts
- confirm README quick start still matches public APIs
- confirm README command examples cover `probe-window` and `probe-sequence`
- confirm report schema docs match actual JSON output
- run full test suite

## Validation Commands

PowerShell:

```bash
$env:PYTHONPATH="src;."
python -m unittest discover -s tests -v
```

bash:

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -v
```

```bash
python -m examples.rl_actor_case --output-dir reports/demo_rl_actor_case --run
```

```bash
python -m examples.showcase_reports --output-dir docs/showcase
```

```bash
python -m examples.validation_suite --output-dir docs/validation
```

```bash
python -m rlplasticity.cli probe-window --help
```

```bash
python -m rlplasticity.cli probe-sequence --help
```

## Release Criteria For 0.1.0

- all tests pass
- healthy case stays quiet
- frozen encoder triggers `encoder_bottleneck`
- frozen trunk triggers `trunk_bottleneck`
- frozen policy head triggers `head_saturation`
- global stall triggers `global_plasticity_stall`
- checkpoint sequence emits history metadata
- README links resolve
- package installs with and without the `torch` extra

## Post-Release

- create GitHub release notes
- link changelog and validation docs
- collect first user feedback on:
  - false positives
  - missing workflows
  - missing integrations
