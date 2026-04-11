# Troubleshooting

## `rlplasticity` command is not found

The editable install may place scripts in a user-level Python scripts directory that is not on `PATH`.

Workarounds:

- activate your virtual environment before installation
- or run the CLI as a module:

```bash
python -m rlplasticity.cli --help
```

## `torch.load(...)` warnings or checkpoint load failures

Make sure you:

- trust the checkpoint source
- install the PyTorch extra with `pip install -e ".[torch]"`
- use a model builder that matches the checkpoint structure

## `probe-model` fails even though the checkpoint loads

Typical causes:

- the model builder does not match the checkpoint
- the sample batch is not in the shape expected by your model
- your forward helper is missing and `model(batch)` is not valid

Try:

- checking your `pkg.module:callable` symbol spec
- providing a `--forward` helper for structured batches
- first running `scan_checkpoint` to confirm the checkpoint structure

## `probe-plasticity` fails or gives a trivial report

Typical causes:

- the loss function does not return a differentiable scalar
- the optimizer is not built for the loaded model
- the provided batch has no useful learning signal

Try:

- verifying that `loss_fn(model, batch)` returns a scalar tensor
- checking whether your loss helper should return `(loss, metadata)`
- comparing one-step and window probes to see if the result is stable

## Window or sequence probes show little history

Make sure you pass:

- multiple batches to `probe_plasticity_window`
- multiple checkpoints to `probe_checkpoint_sequence`

Single-batch inputs still work, but they do not create meaningful trend history.

## The report says "no acute issue" but training is still bad

This tool only diagnoses model-side plasticity evidence. It does not diagnose:

- reward design
- exploration failures
- environment bugs
- logging or trainer orchestration issues

Treat the report as a ranked model-side triage signal, not a full root-cause oracle.
