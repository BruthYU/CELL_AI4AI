# Gene-specific Cross-Cell Perturbation Prior

Canonical branch name: `D_same_gene_other_cellline`.

This document fixes the exact implementation contract used by the Replogle
other-cell delta baseline and by the generic reproducibility script:

```text
tools/gene_specific_cross_cell_prior.py
```

## Symbols

- `p`: perturbation condition label. In Replogle this is often a gene symbol
  such as `ACTR2` or `MRPS31`, but here it is a perturbation label.
- `c`: cell context / cell line.
- `g`: expression feature in the expression matrix.
- `G`: expression feature set, for example the 2000 HVGs used by Replogle.
- `c*`: target cell line to predict.
- `T_c`: test perturbation labels for cell line `c`.

The target is the response vector for `(p, c*)` over all `g in G`.

## Train Delta Memory

For train rows only:

```text
mu_train(p, c, g) = mean expression of train cells with perturbation=p, cell=c
mu_ctrl(c, g)     = mean expression of train control cells with cell=c

delta_train(p, c, g) = mu_train(p, c, g) - mu_ctrl(c, g)
```

For target `(p, c*)`, only source contexts different from the target are used:

```text
S(p, c*) = { c_source : c_source != c*, and train contains (p, c_source) }
```

Default Replogle-compatible averaging is equal weight per source cell line:

```text
D_same_gene_other_cellline(p, c*, g)
  = mean_{c_source in S(p,c*)} delta_train(p, c_source, g)
```

If `S(p, c*)` is empty, the prediction delta is the all-zero vector. The
resulting delta Pearson is NaN because the predicted vector has zero variance;
the nan-as-zero aggregate counts that target as zero.

## Prediction and Scoring

The memory-only predicted expression is:

```text
y_hat(p, c*, g) = mu_ctrl(c*, g) + D_same_gene_other_cellline(p, c*, g)
```

The direct delta Pearson used for the reported score is:

```text
delta_real(p, c*, g) = mu_real(p, c*, g) - mu_real_ctrl(c*, g)
delta_pred(p, c*, g) = y_hat(p, c*, g) - mu_ctrl(c*, g)
                     = D_same_gene_other_cellline(p, c*, g)

rho(p, c*) = Pearson_g(delta_pred(p, c*, g), delta_real(p, c*, g))
```

Per-cell-line nan-as-zero score:

```text
score(c) = mean_{p in T_c} z(rho(p,c))
z(rho) = rho if finite else 0
```

The main Replogle number is the simple macro average over the four cell lines,
not a target-count weighted average.

## Full-Cell Output Rule

The prior may average source-context deltas, but the final full-cell prediction
file must not collapse each perturbation to a single target cell. For Cell-Eval
and DE-style metrics, each test perturbation should retain multiple predicted
cells, normally by anchoring the same predicted delta on sampled or matched
target-control cells. A compact one-row-per-perturbation file is only valid for
direct delta-PCC diagnostics.

## Generic Reproducibility Script

Script:

```bash
python /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/tools/gene_specific_cross_cell_prior.py --help
```

Modes:

- `make-example`: writes a deterministic toy CSV dataset.
- `csv`: scores row-level train/eval CSV files.
- `h5ad`: scores a single H5AD using either split labels or explicit
  train/eval `(context, perturbation)` pair CSV files. It reads H5AD with h5py
  directly and does not require `anndata`.

Toy smoke test:

```bash
python /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/tools/gene_specific_cross_cell_prior.py \
  make-example \
  --out-dir /tmp/gene_cross_cell_prior_example

python /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/tools/gene_specific_cross_cell_prior.py \
  csv \
  --train-csv /tmp/gene_cross_cell_prior_example/train.csv \
  --eval-csv /tmp/gene_cross_cell_prior_example/eval.csv \
  --out-dir /tmp/gene_cross_cell_prior_example/score \
  --gene-cols g0,g1,g2,g3,g4,g5,g6,g7 \
  --expected-mean-nan-as-zero 0.999 \
  --expected-tol 0.01
```

The current smoke-test result is:

```text
mean_pearson_delta_nan_as_zero = 0.9998830432273297
```

CSV data contract:

- One row is one cell.
- Required metadata columns default to `celltype` and `gene`.
- Control perturbation defaults to `non-targeting`.
- Expression columns are numeric and are selected by `--gene-cols`,
  `--gene-cols-file`, or inferred after excluding metadata columns.
- Use `--average-mode equal_context` for Replogle-compatible source-cell-line
  averaging. `--average-mode cell_weighted` is available for ablations only.

H5AD split example:

```bash
python /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/tools/gene_specific_cross_cell_prior.py \
  h5ad \
  --input-h5ad DATA.h5ad \
  --context-col celltype \
  --pert-col gene \
  --control-pert non-targeting \
  --split-col split \
  --train-splits train \
  --eval-splits test \
  --gene-cols-file hvg_2000.txt \
  --out-dir OUT_DIR
```

H5AD pair-list example:

```bash
python /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/tools/gene_specific_cross_cell_prior.py \
  h5ad \
  --input-h5ad DATA.h5ad \
  --context-col celltype \
  --pert-col gene \
  --control-pert non-targeting \
  --train-pairs-csv train_pairs.csv \
  --eval-pairs-csv test_pairs.csv \
  --gene-cols-file hvg_2000.txt \
  --out-dir OUT_DIR
```

In `--split-col` mode, control rows are filtered by `--train-splits` or
`--eval-splits` before computing train deltas or real eval deltas. In pair-list
mode there is no separate control split field, so all control rows in the H5AD
are used; for strict train/test control separation, prefer `--split-col`.

Main outputs:

- `direct_delta_pearson.json`
- `per_target_delta_pearson.csv`
- optional `predicted_deltas.csv.gz` and `predicted_means.csv.gz` with
  `--write-predicted`

## Replogle Alignment

Existing exact memory-only artifact:

```text
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/benchmark/workspace/replogle_reproduce_othercell_delta_20260622/direct_delta_pearson.json
```

Known memory-only result:

| cell line | total | finite | NaN | finite-only | nan-as-zero |
| --- | ---: | ---: | ---: | ---: | ---: |
| rpe1 | 1061 | 987 | 74 | 0.8566646037 | 0.7969160828 |
| hepg2 | 945 | 888 | 57 | 0.7745280030 | 0.7278104410 |
| jurkat | 1084 | 995 | 89 | 0.7924884736 | 0.7274225381 |
| k562 | 968 | 898 | 70 | 0.7196643557 | 0.6676225118 |

Macro mean:

```text
mean_pearson_delta_finite_only  = 0.7858363590187023
mean_pearson_delta_nan_as_zero  = 0.7299428934044634
target-weighted nan-as-zero     = 0.7314178051342977
```

The known model/memory fusion diagnostic artifact is:

```text
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/benchmark/workspace/replogle_othercell_model010_memory090_fullcell_eval_20260622/direct_delta_pearson.json
```

Known model010 + memory090 result:

```text
mean_pearson_delta_nan_as_zero = 0.7456459752638236
target-weighted nan-as-zero    = 0.747323325605701
```

The prediction workspace recorded in that JSON is compact:

```text
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/benchmark/workspace/replogle_othercell_model010_memory090_20260622/predictions
```

An h5py multiplicity check shows every test perturbation has one predicted row
in that workspace. It is valid for direct mean-level delta-PCC diagnostics, but
not valid for full-cell Cell-Eval/DE metrics.

The intended full-cell generator is:

```text
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/tools/replogle_othercell_model_memory_fullcell.py
```

It uses:

```text
pred_cell = target_control_cell
          + 0.1 * (model_pred_cell - mean_model_control)
          + 0.9 * D_same_gene_other_cellline(p, c*)
```

It expects the model prediction workspace to already contain replicated
per-cell rows. The default model workspace passes this check with at least 128
rows per perturbation target:

```text
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/remote-chen/nemo_cellflow/benchmark/workspace/aivc-llama-jit-replogle-v3-statealign-resid-gdelta-neg0235-set512-dit36-2gpu-acc4-lr8e5-w6_ep129_none_normalgpu1_mem24g_scale0528_full_eval/predictions
```

Use this h5py-only guard before full-cell eval:

```bash
/usr/bin/python /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/tools/check_replogle_h5ad_multiplicity.py \
  --workspace PREDICTIONS_DIR \
  --min-cells-per-target 2
```

## Four Separate Replogle Cell-Line Checkpoints

The separate old cell-line checkpoints found on disk are:

| cell line | checkpoint weights |
| --- | --- |
| rpe1 | `/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/outputs/checkpoints/aivc-replogle-llama-detach-rpe1/finetune_best-step=959/weights` |
| hepg2 | `/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/outputs/checkpoints/aivc-replogle-llama-detach-hepg2/finetune_best-step=2039/weights` |
| jurkat | `/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/outputs/checkpoints/aivc-replogle-llama-detach-jurkat/finetune_best-step=939/weights` |
| k562 | `/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/outputs/checkpoints/aivc-replogle-llama-detach-k562/finetune_best-step=959/weights` |

Note: `config/prior_llm_replogle_hepg2.yaml` and
`config/prior_llm_replogle_jurkat.yaml` still point to
`finetune_best-step=959/weights`, but those directories were not found for
hepg2/jurkat. The existing directories are `2039` for hepg2 and `939` for
jurkat.

## Checks Before Trusting a New Result

- Memory is built from train split only.
- Target context `c*` is excluded from sources.
- `p` is a perturbation label; expression features are always `g in G`.
- Source deltas use equal-context averaging for Replogle reproduction.
- Missing source support produces zero predicted delta and NaN per-target
  Pearson, which contributes zero under nan-as-zero.
- Train/eval feature order is identical.
- Full-cell H5AD output keeps multiple predicted cells per perturbation.
