# Security Policy

## Supported Versions

Security fixes will be applied on the latest development state of the project.

## Reporting a Vulnerability

Please do **not** report potential vulnerabilities in a public GitHub issue.

Instead, contact the repository owner privately through GitHub and include:

- a clear description of the issue
- the affected code path
- reproduction steps if available
- any proof-of-concept or sample artifact needed to reproduce

## Safe Usage Notes

`RLPlasticity` works with model checkpoints and Python model code.

Please assume the following:

- untrusted model files may be unsafe
- loading unknown PyTorch checkpoints can execute unsafe deserialization paths
- custom model-builder code should be treated as code execution, not passive data loading

Recommendations:

- only inspect checkpoints from trusted sources
- use isolated virtual environments
- avoid running third-party model code on sensitive machines
- review custom builder and loss functions before running CLI probe workflows

## Scope

This policy covers vulnerabilities in the repository's own code and packaging.

It does not treat every unsafe third-party checkpoint as a vulnerability in `RLPlasticity`; part of this tool's threat model is that model files and model code can be untrusted.
