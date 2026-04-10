# Contributing

Thanks for helping improve `RLPlasticity`.

## Before You Start

Please open an issue first for:

- new features
- large refactors
- changes to public APIs
- new analysis rules or report semantics

Small fixes, docs improvements, and tests are always welcome.

## Development Setup

```bash
pip install -e .
pip install -e ".[torch]"
```

Run tests:

```bash
PYTHONPATH=src;. python -m unittest discover -s tests -v
```

## What Good Contributions Look Like

We especially welcome contributions that improve:

- real RL use cases
- reporting clarity
- false-positive reduction
- framework integrations
- documentation and examples
- robustness tests for bad or partial inputs

## Pull Request Guidelines

Please try to keep pull requests:

- focused on one change area
- covered by tests when behavior changes
- documented when user-facing behavior changes

For analysis rules in particular, include:

- the intended user scenario
- the evidence level it relies on
- at least one positive test
- at least one non-trigger or false-positive test when practical

## Style Expectations

- Prefer explicit, typed Python.
- Keep runtime collection separate from analysis logic.
- Avoid changing public report semantics without discussion.
- Preserve clear caveats when a result is low-confidence.

## Community

By participating in this project, you agree to follow the [Code of Conduct](/C:/Users/22050/Desktop/RLPlasticityLab/CODE_OF_CONDUCT.md).
