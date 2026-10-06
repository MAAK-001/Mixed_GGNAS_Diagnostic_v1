# Mixed-GGNAS — Two-Path Training-Free NSGA-II

This version keeps the existing eight-gene Mixed-UNet search space and the original SynFlow/parameter objectives, and adds two independent third-objective experiments.

## Search paths

### 1. Spatial-SWAP
`nsga2/nsga2_spatial_swap.py`

Objectives passed to Pymoo NSGA-II:

1. maximize `log10(SynFlow)`
2. minimize trainable parameters
3. maximize the Spatially-Aware Sample-Wise Activation Proxy

### 2. SA-JacCov
`nsga2/nsga2_seg_jacobian.py`

Objectives passed to Pymoo NSGA-II:

1. maximize `log10(SynFlow)`
2. minimize trainable parameters
3. maximize Segmentation-Aware Jacobian Covariance

Both branches use the exact same:

- chromosome/search space
- population size and generations
- random seed
- dataset split
- validation calibration inputs
- reduced proxy resolution
- base channels
- crossover probability
- mutation probability

The branches are sequential, not simultaneous.

## Discrete NSGA-II operators

The algorithm itself is imported directly from Pymoo. No custom NSGA-II implementation is used.

- Sampling: `IntegerRandomSampling`
- Crossover: Pymoo `UniformCrossover`
- Mutation: Pymoo `PolynomialMutation` with `RoundingRepair`
- Problem variables: integer genes in `[0, 4]`

Thus every chromosome remains exactly eight discrete block IDs from `{0,1,2,3,4}`.

## TOPSIS

Each branch applies TOPSIS independently to its own final nondominated front:

- 50% — third objective
- 30% — trainable parameters (cost)
- 20% — SynFlow (benefit)

## Three-scale handling

The three scale variants inside every manual block are handled in two stages:

1. **NSGA-II search:** all three scales remain present and participate in the forward pass. The zero-cost objectives therefore evaluate the complete three-scale candidate architecture. The chromosome does not contain scale genes.
2. **Final training:** after TOPSIS selects the chromosome, a short supervised scale-selection stage trains all three scale branches and their softmax weights. The highest-weight scale is selected independently for every manual block. The losing two branches are then removed, a fresh optimizer is created, and the remaining full-training epochs continue using only the selected scale in each block.
3. **Final testing:** the saved selected-scale manifest is used to reconstruct the exact collapsed architecture, so testing uses one selected scale per manual block.

The total final-training epoch budget remains `full_epochs`; `scale_selection_epochs` is the portion of that budget used to learn the scale weights.

## Final training loss

Final retraining uses:

`L = a1 * L_BCE + a2 * L_mIoU`

with:

- `a1 >= 0`
- `a2 >= 0`
- `a1 + a2 = 1`

The coefficients are selected by a bounded one-dimensional SciPy optimizer over `a1`. Each trial performs a short deterministic calibration training with all three scales active, and the selected coefficients are then used for the fresh full-budget training run described above.

There is no Dice loss in final training.

## Pipeline

`main.py --mode all` performs:

1. Search with Spatial-SWAP NSGA-II.
2. Search with SA-JacCov NSGA-II.
3. TOPSIS selection for each branch.
4. Full training of each selected architecture.
5. Untouched test evaluation of each selected architecture.
6. Final side-by-side comparison.
7. One common 3-D Pareto-front comparison plot.

Outputs are separated under:

```text
experiments/<DATASET>/spatial_swap/
experiments/<DATASET>/seg_jacobian/
experiments/<DATASET>/comparison/
```

## Removing one branch later

To remove Spatial-SWAP:

1. Delete `nsga2/nsga2_spatial_swap.py`.
2. Remove its import from `main.py`.
3. Remove the `SPATIAL_BRANCH` constant and the Spatial-SWAP block in `run_search_branch` / `run_pipeline`.
4. Remove its final-training/test block.
5. Keep the SA-JacCov branch unchanged.

The same process works in reverse for SA-JacCov.


```bash
python main.py --dataset BUSI --mode all
```

Search only:

```bash
python main.py --dataset BUSI --mode search
```

Final training after search:

```bash
python main.py --dataset BUSI --mode train-final
```

Testing after training:

```bash
python main.py --dataset BUSI --mode test
```