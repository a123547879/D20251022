# GhostNet Ablation Matrix and Results

The paper originally compared original, modified, and simple GhostNet variants. Reviewers correctly noted that those variants change multiple factors at once. The revised experiment separates SE modules and residual shortcuts through a controlled GhostNet ablation on Sudoku-Digits and USPS test.

## Executed Matrix

| Variant | SE | Residual shortcut | Bottleneck expansion/compression | Direct Ghost stack | Purpose |
|---|---:|---:|---:|---:|---|
| `ghostnet_original` | yes | yes | yes | no | Original reference. |
| `ghostnet_no_se` | no | yes | yes | no | Isolate SE effect. |
| `ghostnet_no_shortcut` | yes | no | yes | no | Isolate shortcut effect. |
| `ghostnet_no_se_no_shortcut` | no | no | yes | no | Test combined SE and shortcut removal. |
| `ghostnet_change` | no | no | partially removed | no | Existing modified variant. |
| `ghostnet_simple` | no | no | no | yes | Existing direct-stack variant. |

## Implemented Local Models

The single-factor variants are implemented in:

```text
revision/experiments/ghostnet_ablation_models.py
```

They reuse the original GhostNet stage configuration and change only the following switches:

- `ghostnet_no_se`: disables SE modules while keeping shortcut paths.
- `ghostnet_no_shortcut`: disables shortcut addition while keeping SE modules.
- `ghostnet_no_se_no_shortcut`: disables both SE modules and shortcut addition.

The variants are exposed through `revision/experiments/digit_shift_pipeline.py`.

## Completed Outputs

Each variant was trained with the same MNIST protocol and evaluated on Sudoku-Digits and USPS test:

| Variant | Sudoku acc. | USPS acc. | Params | MACs (M) | FLOPs (M) |
|---|---:|---:|---:|---:|---:|
| `ghostnet_original` | 0.8187 +/- 0.0356 | 0.4215 +/- 0.0426 | 3914030 | 145.80 | 291.59 |
| `ghostnet_no_se` | 0.8908 +/- 0.0122 | 0.1829 +/- 0.0113 | 2410162 | 144.29 | 288.58 |
| `ghostnet_no_shortcut` | 0.9146 +/- 0.0019 | 0.6587 +/- 0.1518 | 3914030 | 138.81 | 277.61 |
| `ghostnet_no_se_no_shortcut` | 0.9010 +/- 0.0115 | 0.6765 +/- 0.0637 | 2410162 | 137.30 | 274.60 |
| `ghostnet_change` | 0.9056 +/- 0.0314 | 0.5165 +/- 0.0517 | 2575594 | 247.46 | 494.93 |
| `ghostnet_simple` | 0.8743 +/- 0.0256 | 0.6494 +/- 0.1556 | 1984234 | 171.08 | 342.16 |

Result files:

- `revision/experiments/results/ghost_ablation/ghost_ablation_summary.csv`
- `revision/experiments/results/ghost_ablation/ghost_ablation_summary.md`
- `revision/experiments/results/ghost_ablation_usps/ghost_ablation_usps_summary.csv`
- `revision/experiments/results/ghost_ablation_usps/ghost_ablation_usps_summary.md`
- `revision/experiments/results/ghost_ablation_combined/ghost_ablation_target_complexity_summary.csv`
- `revision/experiments/results/model_complexity.csv`
- `revision/experiments/checkpoints/ghost_ablation/*.pth`

## Interpretation Rule

Do not attribute the improvement to generic "structural simplification." The completed result shows that variants without shortcut addition are competitive on the two tested digit targets, while the direct-stack variant is not consistently best and SE removal is target-dependent. Broader claims still require latency measurements, pruning baselines, and natural-image domain generalization benchmarks.
