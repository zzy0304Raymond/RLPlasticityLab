# Architecture

## Design Principle

The project is organized around **progressive evidence**.

Different users have different amounts of context:

- only a checkpoint
- checkpoint plus model
- checkpoint plus model plus replay samples

Instead of forcing every workflow into one heavy path, `RLPlasticity` offers progressively stronger evidence levels.

## Layers

### `core`

Responsibilities:

- shared enums such as analysis kind and evidence level
- normalized snapshot schema
- metric and rule interfaces
- report serialization
- aggregation helpers for short probe windows

The `core` layer is intentionally runtime-agnostic.

### `ingest`

Responsibilities:

- load checkpoint payloads
- extract likely `state_dict` mappings
- summarize parameter tensors without requiring a live model

This is where artifact variability gets normalized before analysis starts.

### `probes`

Responsibilities:

- collect a `Snapshot` from a concrete user scenario
- keep runtime concerns separate from analysis concerns

Current probe families:

- static checkpoint scan
- forward-only model probe
- update-based plasticity probe

### `plasticity`

Responsibilities:

- metrics
- diagnostic rules
- analyzer presets

The analyzers are deliberately separated by evidence level so weak evidence does not accidentally produce overly strong conclusions.

### `reporting`

Responsibilities:

- text and HTML output
- explicit caveats
- stable JSON serialization via `AnalysisReport.to_dict()`

## Public API

The intended public surface for v0.1 is:

- `scan_checkpoint(...)`
- `probe_model(...)`
- `probe_plasticity(...)`

Everything else should be treated as internal or advanced usage.

## Why The Architecture Looks This Way

The repository is optimized for low-cost triage, not experimental completeness.

That leads to a few concrete engineering choices:

- offline workflows first
- rule-based analysis before learned models
- short probe windows before long monitoring runs
- explicit evidence levels in every report

This keeps the first release small, honest, and extensible.
