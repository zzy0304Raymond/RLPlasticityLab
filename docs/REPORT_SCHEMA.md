# Report Schema

This document describes the JSON structure returned by `AnalysisReport.to_dict()`.

The schema is intentionally lightweight and human-readable. It is designed for:

- saving offline diagnostics
- comparing runs and checkpoints
- attaching structured evidence to issues or experiment logs

## Top-Level Shape

```json
{
  "analyzer_name": "plasticity/default",
  "snapshot": {},
  "metrics": {},
  "findings": [],
  "caveats": []
}
```

## `analyzer_name`

String identifying which analyzer generated the report.

Examples:

- `scan/checkpoint`
- `probe/forward`
- `plasticity/default`

## `snapshot`

Normalized evidence captured by the workflow.

Fields:

- `kind`
  - one of `checkpoint_scan`, `model_probe`, `plasticity_probe`
- `evidence_level`
  - one of `static`, `forward`, `update`, `window`
- `step`
  - integer step index when available
- `loss`
  - float loss when available
- `layers`
  - list of per-layer snapshots
- `metadata`
  - workflow-specific structured metadata
- `caveats`
  - list of strings carried from the collection stage

### `snapshot.layers[]`

Each layer snapshot may contain:

- `name`
- `group`
- `module_type`
- `parameter_count`
- `parameter_norm`
- `parameter_mean_abs`
- `parameter_zero_fraction`
- `parameter_max_abs`
- `gradient_norm`
- `update_norm`
- `relative_update`
- `grad_to_weight_ratio`
- `activation_mean`
- `activation_std`
- `activation_shift`
- `zero_activation_fraction`
- `max_activation_abs`
- `metadata`

Not every field is populated in every workflow. For example:

- `scan_checkpoint` mostly uses parameter statistics
- `probe_model` uses activation statistics
- `probe_plasticity` adds gradients and update statistics

### `snapshot.metadata.history`

Window and checkpoint-sequence probes record compact history rows under `snapshot.metadata["history"]`.

Each history row can include:

- `label`
- `step`
- `loss`
- `mean_relative_update`
- `mean_grad_to_weight_ratio`
- `group_relative_update`
- `group_grad_to_weight_ratio`
- `layer_count`

This allows trend analysis without storing every full per-layer snapshot separately.

## `metrics`

Dictionary from metric name to metric result.

Each metric result contains:

- `name`
- `value`
- `summary`
- `metadata`

Examples:

- `plasticity_score`
- `stagnant_layer_fraction`
- `encoder_plasticity_score`
- `plasticity_trend_delta`
- `plasticity_first_decline`
- `encoder_plasticity_first_decline`

## `findings`

Ordered list of human-readable rule outputs.

Each finding contains:

- `name`
- `severity`
- `summary`
- `evidence`
- `recommendations`
- `confidence`
- `metadata`

Examples:

- `encoder_bottleneck`
- `head_saturation`
- `trunk_bottleneck`
- `global_plasticity_stall`
- `plasticity_decline_trend`

## `caveats`

List of high-level warnings about evidence strength or scope.

Examples:

- static scans cannot measure gradient flow
- low-cost probes are not full experiments
- window reports average several steps and may hide instability

## Stability Notes

The project treats this JSON format as a public interface for the first release candidate, but still reserves the right to add new fields in minor releases.

Compatibility expectations:

- existing keys should remain stable where practical
- new metrics and findings may appear over time
- consumers should ignore unknown keys rather than failing hard
