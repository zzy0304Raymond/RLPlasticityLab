"""Static scan example that works without a live PyTorch model."""

from rlplasticity import scan_checkpoint


def main() -> None:
    state_dict = {
        "encoder.weight": [[0.0, 0.0], [0.0, 0.0]],
        "policy.bias": [1.0, -1.0],
        "value.bias": [0.1],
    }
    report = scan_checkpoint(state_dict)
    print(report.to_text())


if __name__ == "__main__":
    main()
