# Validation Artifacts

| Case | Expected pattern | Actual summary | Findings |
| --- | --- | --- | --- |
| healthy | no acute issue | No acute issue detected. Plasticity score=1.000 (update-active=1.000, grad-active=1.000) | none |
| frozen_encoder | encoder-side bottleneck | The encoder is adapting less than downstream layers. | encoder_bottleneck |
| frozen_policy_head | head-side bottleneck | The output heads are adapting less than the trunk. | head_saturation |
| frozen_trunk | trunk-side bottleneck | The shared trunk is adapting less than surrounding modules. | trunk_bottleneck |
| global_stall | global plasticity stall | The model looks globally plasticity-limited. | global_plasticity_stall |
| checkpoint_sequence | history metadata and trend metrics available | No acute issue detected. Plasticity score=1.000 (update-active=1.000, grad-active=1.000) | none |

Generated files:

- `healthy`: `healthy.txt`, `healthy.json`, `healthy.html`
- `frozen_encoder`: `frozen_encoder.txt`, `frozen_encoder.json`, `frozen_encoder.html`
- `frozen_policy_head`: `frozen_policy_head.txt`, `frozen_policy_head.json`, `frozen_policy_head.html`
- `frozen_trunk`: `frozen_trunk.txt`, `frozen_trunk.json`, `frozen_trunk.html`
- `global_stall`: `global_stall.txt`, `global_stall.json`, `global_stall.html`
- `checkpoint_sequence`: `checkpoint_sequence.txt`, `checkpoint_sequence.json`, `checkpoint_sequence.html`