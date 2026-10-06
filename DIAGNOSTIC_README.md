# Mixed-GGNAS Diagnostic Codebase

This repository is a **diagnostic experiment**, not the production NAS pipeline. Its purpose is to decide, from measurements on the real BUSI/CVC/IDRiD data, whether the current ceiling is caused primarily by training, the three-scale design, the architecture search space, or the zero-cost proxies.

## Scientific rules

1. **No synthetic images or masks are generated.** Every proxy image/mask comes from the real validation split of the selected dataset.
2. Proxy correlation is measured against **validation Dice and validation mIoU** because the test split must remain untouched during architecture/proxy development.
3. The test split is used only for fixed, pre-declared baseline sanity checks. It is never used to choose a proxy or architecture.
4. Every architecture is trained with the **same training recipe and seed construction**, so proxy comparisons are not confounded by different initialization.
5. The diagnostic includes both the project's current proxies and paper-derived controls.
6. The three-scale mixture is retained by default. A matched scale ablation tests whether hard collapse is actually beneficial.

## Proxies measured

- `synflow_log10`: current SynFlow.
- `current_sa_swap`: current project Spatial-SWAP heuristic.
- `current_sa_jaccov`: current segmentation-aware Jacobian covariance heuristic.
- `original_swap`: paper-faithful SWAP construction: unique sample-wise binary ReLU activation patterns across the real proxy batch.
- `regularized_swap`: SWAP with the paper's Gaussian model-size regularization form; μ and σ are calibrated from the diagnostic architecture sample.
- `original_jacobian_covariance`: Mellor-style Jacobian covariance across real samples, adapted to the single-output segmentation setting. This is explicitly labeled an adaptation because the original paper's benchmark output is classification-oriented.
- `naswot_relu_logdet`: standard ReLU activation-kernel logdet control.
- `grad_norm`, `snip`: standard data-dependent zero-cost controls.
- `parameters`: simple architecture-size control.

The report computes Pearson, Spearman and Kendall rank correlations, p-values, and bootstrap 95% confidence intervals.

## Main command

From the repository root:

```bash
python diagnosis/run_diagnosis.py --dataset BUSI --data-root /path/to/Datasets --architectures 24 --epochs 30 --proxy-images 32 --proxy-repeats 3
```

For Kaggle:

```python
%cd /kaggle/working/NSGA2_NAS_v3
!python diagnosis/run_diagnosis.py --dataset BUSI --data-root /kaggle/input/datasets/maak01/e-nas-datasets/Datasets --architectures 24 --epochs 30 --proxy-images 32 --proxy-repeats 3
```

For CVC:

```python
!python diagnosis/run_diagnosis.py --dataset CVC --data-root /kaggle/input/datasets/maak01/e-nas-datasets/Datasets --architectures 24 --epochs 30 --proxy-images 32 --proxy-repeats 3
```

For IDRiD:

```python
!python diagnosis/run_diagnosis.py --dataset IDRID --data-root /kaggle/input/datasets/maak01/e-nas-datasets/Datasets --architectures 24 --epochs 30 --proxy-images 32 --proxy-repeats 3
```

## Outputs

`diagnostic_runs/<DATASET>/` contains:

- `verification_report.json`: source/data integrity and split verification.
- `calibration_architectures.csv`: exact architectures evaluated.
- `real_proxy_batch.pt`: the real images/masks used for the proxy study.
- `proxy_scores.csv`: all zero-cost proxy measurements.
- `architecture_diagnostic.csv`: proxy scores joined with trained validation Dice/mIoU.
- `correlations_val_dice.csv`: correlation table with bootstrap confidence intervals.
- `correlations_val_miou.csv`: same for mIoU.
- `plots/`: one scatter plot per proxy and target.
- `scale_ablation.csv`: matched all-three-scale versus hard-scale comparisons.
- `baseline_test_results.csv`: fixed baseline test sanity check; never used for selection.
- `histories/`: epoch-by-epoch validation curves.
- `calibration_checkpoints/`: best validation checkpoints for calibration architectures.

## How to interpret the diagnosis

### Search/proxy problem
If the training baselines are strong but the best proxies have weak/unstable Spearman and Kendall correlation with Dice/mIoU, the search is proxy-limited. Do not run another NSGA-II with the same proxy.

### Training/model problem
If fixed canonical architectures also plateau at low Dice/mIoU, especially on CVC, the search is not the first problem. Fix training, preprocessing, loss, capacity or architecture construction before changing the proxy.

### Scale-collapse problem
If `all_three_scales` consistently beats every hard-scale condition for the same chromosome, keep the mixture throughout training. If a hard scale is consistently better, retain hard selection but use a statistically justified selection stage rather than a near-tied 10-epoch argmax.

### Proxy-selection rule
A proxy is not promoted merely because its mean correlation is positive. Prefer a proxy that:

1. has materially positive Spearman and Kendall correlation with both Dice and mIoU;
2. has a confidence interval that does not sit around zero;
3. is stable across repeated real-data proxy batches;
4. beats parameter count/FLOPs as simple controls;
5. gives useful top-k retrieval of high-Dice architectures, not just global correlation.

### Important limitation
The paper-faithful JacCov implementation here is a **segmentation adaptation**, not the original classification output formulation. The diagnostic explicitly records this so that a positive/negative result is not misreported as a reproduction of the original paper.
