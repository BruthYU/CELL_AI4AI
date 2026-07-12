---
name: prior-delta-centering
description: Apply a prior-delta center to full-cell perturbation predictions while preserving the model's per-cell prediction distribution. Use for Replogle-style and other perturbation benchmarks when comparing raw model centers, prior-delta centers, and weighted center interpolation without collapsing predictions to one row per perturbation.
---

# Prior Delta Centering

## Method Name

Canonical method name: `Prior Delta Centering`, abbreviated as `PDC`.

Use the following variants when reporting results:

- `PDC-lambda`: weighted center interpolation with `0 <= lambda <= 1`.
- `PDC-1`: full prior-delta center replacement, equal to `lambda = 1`.
- `PDC-external-center`: replay a supplied compact/external center. This is only for reproducing an old artifact, not the preferred train-only method.

Do not call this method `model_plus_D_same_gene_other_cellline`. That name is reserved for the strict formula
`control + a * delta_model + b * D_same_gene_other_cellline`.

## Purpose

`PDC` answers this question:

Can we keep the model's many predicted cells for each target perturbated gene, but replace or blend the mean center of those cells with a train-only empirical delta center?

This is useful when:

- the raw model has full-cell predictions, but the mean perturbation effect is weak;
- a train-only same-gene other-cellline delta has a stronger mean effect;
- evaluation needs many cells per target perturbated gene, so one-row compact predictions are invalid;
- we want a generic post-processing or export-time method that can be checked across datasets.

## Symbols

Use these symbols consistently:

- `c`: context, for example cell line, cell type, donor, dose context, or dataset-defined condition.
- `g`: target perturbated gene or perturbation key.
- `i`: predicted cell index within `(c, g)`.
- `pred[c,g,i]`: raw model predicted expression for cell `i`.
- `ctrl[c]`: control center used for context `c`.
- `mu_model[c,g]`: raw model center, defined as the mean of all `pred[c,g,i]`.
- `D_train[c,g]`: train-only delta memory for `(c, g)`.
- `mu_train[c,g]`: prior-delta center, defined as `ctrl[c] + D_train[c,g]`.
- `lambda`: center interpolation weight. `lambda = 0` gives raw model centers; `lambda = 1` gives prior-delta centers.
- `out[c,g,i]`: final full-cell prediction.

All vectors are gene-expression vectors with the same gene dimension.

## Train-Only Delta

Default branch for Replogle-style CRISPR data:

```text
D_train[c,g] =
  average over train contexts c' != c of
    mean(train_pert[c',g]) - mean(train_ctrl[c'])
```

This is the canonical `D_same_gene_other_cellline` branch.

Rules:

- Build `D_train` from the active train split only.
- Do not use validation or test real cells when constructing `D_train`.
- Exclude the exact target combo `(c, g)` whenever the benchmark is context/combo generalization.
- If `g` is not literally a gene, keep the same formula but define `g` as the dataset perturbation key.
- Record support count, fallback type, and missing count for every `(c, g)`.

Optional fallbacks, in order:

```text
D_same_cell_other_pert[c,g] =
  average over train perturbations g' != g of
    mean(train_pert[c,g']) - mean(train_ctrl[c])

D_global_train_delta =
  average over all eligible train perturbation/context deltas
```

Fallback use must be explicit in the output manifest.

## Core Formula

First compute the raw model center:

```text
mu_model[c,g] = mean_i pred[c,g,i]
```

Compute the prior-delta center:

```text
mu_train[c,g] = ctrl[c] + D_train[c,g]
```

Interpolate the two centers:

```text
mu_PDC[c,g; lambda] =
  (1 - lambda) * mu_model[c,g] + lambda * mu_train[c,g]
```

Shift every raw predicted cell by the same center correction:

```text
out[c,g,i] =
  pred[c,g,i] + (mu_PDC[c,g; lambda] - mu_model[c,g])
```

Equivalent shorter form:

```text
out[c,g,i] =
  pred[c,g,i] + lambda * (mu_train[c,g] - mu_model[c,g])
```

This guarantees, before any clipping:

```text
mean_i out[c,g,i] = mu_PDC[c,g; lambda]
```

Interpretation:

- `lambda = 0`: unchanged raw model prediction.
- `lambda = 1`: keep model cell-to-cell differences, but set the mean center to the prior-delta center.
- `0 < lambda < 1`: weighted sum of the raw model center and prior-delta center.

## Difference From Strict Model Plus Train Delta

Strict delta blending is:

```text
delta_model[c,g,i] = pred[c,g,i] - ctrl[c]

out_strict[c,g,i] =
  ctrl[c]
  + model_weight * delta_model[c,g,i]
  + prior_weight * D_train[c,g]
```

`PDC` is different:

```text
out_PDC[c,g,i] =
  pred[c,g,i] + lambda * (ctrl[c] + D_train[c,g] - mu_model[c,g])
```

Main difference:

- strict blending scales the whole model delta for every cell;
- `PDC` only shifts the group mean center and preserves the model's per-cell residual around that center.

Use strict blending when the experiment asks for `test_control + a * model_delta + b * train_delta`.
Use `PDC` when the experiment asks whether weighted centers can improve full-cell h5ad predictions without collapsing cells.

## Inputs

Required inputs:

- raw full-cell model prediction h5ad files;
- train perturbation data or precomputed train-only memory h5ad/pkl;
- context column name, for example `cell_type` or `cell_line`;
- perturbation column name, for example `target_gene`, `gene`, `condition`, or dataset-specific key;
- control label and control rows for each context;
- chosen `lambda` value;
- output directory.

Optional inputs:

- real/test h5ad for evaluation only;
- validation split for selecting `lambda`;
- fallback branch definitions;
- clipping mode, for example no clipping or nonnegative clipping.

## Generic Python Interface

The generic implementation should be one CLI script that can run on any dataset
with h5ad predictions and enough metadata columns to define context, perturbation,
and control.

Suggested script name:

```text
tools/apply_prior_delta_centering.py
```

Required CLI arguments:

```text
--pred-h5ad PATH                 raw full-cell model prediction h5ad
--out-h5ad PATH                  output full-cell h5ad
--context-col NAME               column for c
--pert-col NAME                  column for g
--control-label VALUE            label identifying control rows
--lambda FLOAT                   center interpolation weight
--center-source MODE             train_delta, external_center, or table
```

One of these center sources must be provided:

```text
--train-h5ad PATH                build D_train from train cells
--memory-h5ad PATH               read train-only memory predictions/centers
--center-table PATH              read precomputed centers or deltas from a table
```

Optional CLI arguments:

```text
--control-h5ad PATH              separate control source, if pred h5ad lacks controls
--same-pert-other-context        enable D_same_gene_other_cellline
--same-context-other-pert        enable D_same_cell_other_pert fallback
--global-fallback                enable D_global_train_delta fallback
--clip-min FLOAT                 optional lower clipping after shifting
--real-h5ad PATH                 evaluation only, never used for D_train
--manifest-json PATH             write method/provenance manifest
--sanity-json PATH               write row-count and support sanity checks
--score-json PATH                optional direct delta PCC output
```

The implementation must separate construction and evaluation:

- `train-h5ad`, `memory-h5ad`, and `center-table` may contribute to `D_train`.
- `real-h5ad` may only be used after prediction is written, for scoring.
- If `real-h5ad` is accidentally used while constructing centers, the run should fail.

## Outputs

For each context:

- full-cell predicted h5ad with the same perturbation row multiplicity as the raw model prediction;
- unchanged or explicitly regenerated control rows;
- manifest JSON with paths, weights, formulas, train-memory provenance, fallback counts, and missing counts;
- sanity JSON/CSV with per-context row counts and per-target minimum cell count;
- optional direct delta PCC JSON/CSV;
- optional lambda sweep CSV for diagnostics.

The output h5ad must not become one row per target perturbated gene unless the user explicitly asks for a compact diagnostic file.

## Manifest Contract

Every run must write a manifest with at least:

```text
method_name
method_variant
lambda
formula
pred_h5ad
out_h5ad
context_col
pert_col
control_label
center_source
train_or_memory_source
real_h5ad_eval_only
fallback_policy
clip_min
num_rows_in
num_rows_out
num_contexts
num_perturbations
target_cell_count_min
target_cell_count_median
target_cell_count_max
support_counts_by_branch
missing_center_count
created_at
```

For Replogle, the manifest must also record the four cell-line outputs and the
minimum target-cell count for each cell line.

## Required Checks

Before scoring:

1. Verify each `(c, g)` has more than one predicted cell unless the dataset itself is compact by design.
2. Verify no validation/test real cells were used to build `D_train`.
3. Verify `lambda = 0` reproduces raw model mean-level scores.
4. Verify `lambda = 1` reproduces the prior-delta-center endpoint.
5. If a lambda sweep is run on test, label it diagnostic only.
6. For a benchmark claim, choose `lambda` on validation or train-only-dev, then evaluate held-out test once.
7. Save a manifest that names the method exactly as `PDC-lambda` and records the numeric `lambda`.
8. Compare row counts and target-cell multiplicity before and after writing the output h5ad.
9. If `center_source = external_center`, label the run as artifact replay, not train-only evidence.

## Replogle Acceptance Gates

For current Replogle work:

- `PDC-external-center` should reproduce the old compact-center replay result when given the old compact centers as `mu_train`.
- `PDC-1` with train-only same-gene other-cellline delta should reproduce the current corrected full-cell artifact:
  `benchmark/workspace/replogle_main_corrected_fullcell_remote_ep129_20260623`.
- Valid full-cell Replogle output should have many cells per target perturbated gene. In the current checked artifact the minimum target count is 128 for each of the four cell lines.
- Direct delta PCC should be around the known corrected value, not near zero:
  macro approximately `0.751330` for `PDC-1` on the checked Replogle artifact.

For exact reproduction, compare:

```text
target_cell_count_min per cell line
direct delta PCC per cell line
macro direct delta PCC
weighted direct delta PCC
manifest source paths and lambda
```

Small floating-point differences are acceptable only at normal h5ad/float32
precision. A different method, different memory source, or one-row-per-target
output does not count as reproduction.

Known diagnostic Replogle center sweep from the checked artifact:

```text
lambda = 0.00  macro delta PCC = 0.345860
lambda = 0.90  macro delta PCC = 0.745679
lambda = 0.95  macro delta PCC = 0.751564
lambda = 1.00  macro delta PCC = 0.751330
```

The `lambda = 0.95` improvement over `lambda = 1.00` is very small and was measured on the test artifact, so it is diagnostic unless reselected on validation.

## Generalization Plan

For a new dataset:

1. Identify context key `c`, perturbation key `g`, control rows, and train/test split.
2. Load raw full-cell model predictions.
3. Compute `mu_model[c,g]` from raw predictions.
4. Build `D_train[c,g]` using train-only data.
5. Compute `mu_train[c,g] = ctrl[c] + D_train[c,g]`.
6. Write `out[c,g,i] = pred[c,g,i] + lambda * (mu_train[c,g] - mu_model[c,g])`.
7. Preserve row-level metadata from raw predictions.
8. Write manifest and sanity tables.
9. Run endpoint checks before any sweep.
10. Compare against raw model and strict `control + a * delta_model + b * train_delta`.

## Naming Rules

Use these exact names:

- `PDC-lambda` for the center interpolation method.
- `lambda` for the center interpolation weight.
- `D_same_gene_other_cellline` for Replogle same-gene other-cellline train delta.
- `delta_model` for `pred - control`.
- `model_plus_D_same_gene_other_cellline` only for strict control-plus-delta blending.

Avoid vague aliases such as `correction`, `memory correction`, `full-cell correction`, or `center trick` in final reports unless they are explicitly mapped to `PDC`.
