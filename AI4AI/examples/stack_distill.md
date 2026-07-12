# stack Distillation

## 0. Metadata

- Input path: `/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/AI4AI/examples/stack`
- Repo root: `/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/AI4AI/examples/stack`
- Output file: `/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/AI4AI/examples/stack_distill.md`
- Git remote: `https://github.com/ArcInstitute/stack`
- Git commit: `cacc2e4b09435c3e536d46237d10b50f222dd144`
- Inspection date: 2026-07-09
- Distiller: Codex
- Scope: core model, training/fine-tuning configs, loss functions, in-context inference/generation, dataset construction, and smoke tests. Notebooks were not fully executed.

## 1. Repo Summary

- Task: single-cell foundation modeling with in-context prediction/generation of unseen cell profiles and fine-tuning for human/drug perturbation-like replacement tasks.
- Data modality: cell-by-gene count matrices in AnnData/H5AD/H5, aligned to a target gene list.
- Prediction target: masked gene counts and in-context generated target cell profiles.
- Training objective: masked gene reconstruction with Negative Binomial NLL plus Sliced Wasserstein latent regularization.
- Fine-tuning objective: frozen-teacher student model with replacement/context construction, NB reconstruction, energy-distance MMD on generated distributions and embeddings, latent Sliced Wasserstein, and query/tail classification.
- Inference mode: construct mixed base/test batches, inject prompt/context ratios, decode NB means/samples, optionally iterate generation with a mask/context schedule.
- Main dependencies: PyTorch, PyTorch Lightning, scvi-tools NegativeBinomial, geomloss SamplesLoss, AnnData/H5PY/Scipy.
- Relevant entrypoints:
  - `src/stack/cli/launch_training.py`
  - `src/stack/cli/launch_finetuning.py`
  - `src/stack/cli/generation.py`
  - `src/stack/model.py`
  - `src/stack/models/core/base.py`
  - `src/stack/models/core/losses.py`
  - `src/stack/models/core/inference.py`
  - `src/stack/models/finetune/model.py`
  - `src/stack/models/finetune/mixins.py`
  - `src/stack/data/finetuning/datasets.py`

## 2. Minimal Call Graph

Pretraining:

```text
configs/training/bc_large.yaml
-> stack.cli.launch_training.main
-> parse_dataset_configs + MultiDatasetDataModule
-> build_model_config
-> LegacyLightningGeneModel
-> StateICLModel.forward
-> apply_mask -> _reduce_and_tokenize -> TabularAttentionLayer stack -> NB head
-> NegativeBinomial masked NLL + SlicedWasserstein latent regularization
-> Lightning train/val/test metrics
```

Fine-tuning:

```text
configs/finetuning/ft_parsecg.yaml
-> stack.cli.launch_finetuning.main
-> FinetuneDataModule
-> MultiDatasetSplittableDataset replacement sampling
-> LightningFinetunedModel
-> frozen teacher produces target embeddings from ground_truth_features
-> ICL_FinetunedModel.forward
-> query-position tokens + attention mask + NB head
-> reconstruction + MMD distribution/embedding + SW + classification losses
```

In-context generation:

```text
stack.cli.generation.generate
-> load checkpoint
-> per split base/test AnnData alignment
-> model.get_incontext_generation
-> iterative get_incontext_prediction
-> mixed base/test cells -> model.get_prediction
-> NB count sampling and optional query-logit masking schedule
-> generated AnnData outputs
```

## 3. Atomic Design Cards

### A01. Cell-by-Gene Tabular Attention

- 主类别: Encoder
- 辅助类别: Condition Fusion; Latent / Dynamics Modeling
- 可信度: high
- 代码位置:
  - `src/stack/models/core/base.py:45`: `gene_reduction`
  - `src/stack/models/core/base.py:51`: `gene_pos_embedding`
  - `src/stack/models/core/base.py:53`: repeated `TabularAttentionLayer`
  - `src/stack/modules/attention.py:61`: `TabularAttentionLayer`
  - `src/stack/modules/attention.py:112`: per-cell gene-token attention
  - `src/stack/modules/attention.py:118`: per-gene/cell flattened cell attention
- 设计摘要: Treat each input as a cell-by-gene matrix chunk. Project each cell's whole gene vector into `n_hidden * token_dim`, reshape to `[batch, cells, hidden_gene_tokens, token_dim]`, then alternate attention over gene tokens within each cell and attention over cells with flattened gene-token features.
- 解决的问题: Perturbation prediction needs both intra-cell gene program reasoning and inter-cell population/context reasoning. Pure per-cell MLPs miss cell population structure; pure cell embeddings can discard gene-local signals.
- 机制细节:
  - 输入: `features` shaped `[B, n_cells, n_genes]`.
  - 输出: contextualized tokens reshaped to cell embeddings `[B, n_cells, n_hidden * token_dim]`.
  - 关键张量/分布: reduced tokens `[B, n_cells, n_hidden, token_dim]`; gene positional embedding `[n_hidden, token_dim]`.
  - 训练时行为: masked features pass through all tabular attention layers before NB decoding.
  - 推理时行为: mixed base/test cells share the same attention window, enabling in-context information transfer.
- 优势: Separates gene-program attention and cell-context attention while keeping the matrix structure explicit. This is more directly aligned with single-cell perturbation than flattening all genes into one vector or independently processing cells.
- 为什么适合细胞扰动预测: Perturbation effects are often population-relative and gene-program-specific. Alternating attention can let control/treated/context cells inform target cell embeddings while preserving gene program tokens.
- 迁移到 `nemo_cellflow`:
  - 建议落点: `models/<MODEL_NAME>/networks/_nemo_vf.py` or a new `models/<MODEL_NAME>/networks/_tabular_attention.py`; optional replacement/augmentation for JIT `cellmap_encoder_kwargs`.
  - 需要新增的 `prepare_model` 配置: `n_hidden`, `token_dim`, `n_tabular_layers`, `n_heads`, `mlp_ratio`, `use_tabular_attention`.
  - 是否需要修改 `dsets/`: no. JIT already has `ctrl_cell_emb` / `pert_cell_emb` matrices that can be reshaped as cell sets.
  - 是否兼容 NeMo/BioNeMo 接口: yes if wrapped inside existing model module and called from `loss_fn` / `eval_fn`.
- 风险与限制: Memory scales with `n_cells`, `n_hidden`, and attention over cells. Need reconcile JIT's 2000-d cell embeddings with Stack's gene-count input assumption.
- 最小 ablation: compare JIT cellmap encoder vs tabular attention encoder with same downstream flow/loss; optionally disable cell attention while keeping gene attention.
- 迁移优先级: P0

### A02. Rectangular Gene Masking Across Cell Sets

- 主类别: Task / Data Formulation
- 辅助类别: Loss / Regularization
- 可信度: high
- 代码位置:
  - `src/stack/models/core/base.py:124`: `apply_mask`
  - `src/stack/models/core/base.py:128`: uniform mask rate between `mask_rate_min` and `mask_rate_max`
  - `src/stack/models/core/base.py:133`: one random gene subset used across all cells
  - `src/stack/models/core/base.py:155`: masked features enter the model
  - `src/stack/models/core/losses.py:21`: masked NB reconstruction loss
- 设计摘要: Pretrain by masking a rectangular block: the same subset of genes is removed across all cells in a cell set. The model reconstructs masked genes from observed genes plus other cells in the same set.
- 解决的问题: Single-cell perturbation prediction needs learning conditional gene dependencies and population-level context. Random individual entries are too local; whole-gene masking encourages cross-gene and cross-cell imputation.
- 机制细节:
  - 输入: log1p transformed gene counts.
  - 输出: NB parameters for all genes, supervised only on masked genes.
  - 关键张量/分布: boolean mask `[B, n_cells, n_genes]` where selected gene columns are masked for every cell.
  - 训练时行为: `masked_features[mask] = 0.0`; loss averages only masked entries.
  - 推理时行为: prediction can be run without random loss masking or with controlled evaluation masks.
- 优势: Forces the model to infer missing gene programs from context cells and remaining genes, matching the information pattern of perturbation extrapolation.
- 为什么适合细胞扰动预测: Perturbation response often manifests as coordinated gene modules. Masking entire gene columns across cells pushes model to learn module-level dependencies, not just local noise denoising.
- 迁移到 `nemo_cellflow`:
  - 建议落点: `models/<MODEL_NAME>/nemo_model.py::loss_fn` as an auxiliary masked reconstruction objective on `ctrl_cell_emb` / `pert_cell_emb` or decoder output.
  - 需要新增的 `prepare_model` 配置: `mask_rate_min`, `mask_rate_max`, `masked_recon_weight`, `mask_mode: gene_column`.
  - 是否需要修改 `dsets/`: no.
  - 是否兼容 NeMo/BioNeMo 接口: yes; all masking can be done in model forward/loss.
- 风险与限制: JIT currently works on 2000-gene embeddings/count-like vectors; if values are not raw counts, NB loss may not apply directly, but masking itself still applies.
- 最小 ablation: train with no masked reconstruction, per-entry masking, and gene-column masking.
- 迁移优先级: P0

### A03. Library-Size-Aware Negative Binomial Distribution Head

- 主类别: Decoder / Distribution Head
- 辅助类别: Loss / Regularization
- 可信度: high
- 代码位置:
  - `src/stack/models/core/base.py:67`: `output_mlp`
  - `src/stack/models/core/base.py:108`: `_compute_nb_parameters`
  - `src/stack/models/core/base.py:118`: `px_scale_logits`
  - `src/stack/models/core/base.py:119`: `nb_dispersion = softplus(...)`
  - `src/stack/models/core/base.py:120`: gene proportions via `softmax`
  - `src/stack/models/core/base.py:121`: `nb_mean = px_scale * observed_lib_size`
  - `src/stack/models/core/losses.py:21`: `NegativeBinomial(mu=..., theta=...)`
- 设计摘要: Decode each cell embedding to gene-wise NB mean and dispersion. Mean is constrained by gene probability simplex times observed library size, separating relative expression composition from sequencing depth.
- 解决的问题: Point MSE treats gene expression as homoscedastic continuous values and ignores count overdispersion and library-size variation.
- 机制细节:
  - 输入: final cell embeddings `[B, n_cells, n_hidden * token_dim]` and library sizes `[B, n_cells, 1]`.
  - 输出: `nb_mean`, `nb_dispersion`, `px_scale`.
  - 关键张量/分布: Negative Binomial with mean `mu` and dispersion `theta`.
  - 训练时行为: masked negative log likelihood on raw target counts.
  - 推理时行为: use `nb_mean`; optionally sample counts from NB in generation.
- 优势: Captures count overdispersion and preserves per-cell library size, which is biologically and technically important for scRNA-seq.
- 为什么适合细胞扰动预测: Perturbation targets are distributions of counts, not just deterministic vectors. NB head can model uncertainty and gene-dependent dispersion.
- 迁移到 `nemo_cellflow`:
  - 建议落点: `models/<MODEL_NAME>/nemo_model.py::loss_fn` and decoder module under `models/<MODEL_NAME>/networks/`.
  - 需要新增的 `prepare_model` 配置: `decoder_distribution: nb`, `nb_loss_weight`, `library_size_source`.
  - 是否需要修改 `dsets/`: risky/no depending on whether current `ctrl_cell_emb` and `pert_cell_emb` are raw/log counts. If they are embeddings, use Gaussian/energy loss instead.
  - 是否兼容 NeMo/BioNeMo 接口: yes if implemented inside current loss reduction.
- 风险与限制: Requires nonnegative count targets and meaningful library size. Existing JIT configs describe `cell_dim: 2000` but not necessarily raw count semantics.
- 最小 ablation: MSE decoder vs NB NLL decoder on same architecture; NB mean-only prediction vs NB sampling during inference.
- 迁移优先级: P1

### A04. Sliced Wasserstein Regularization of Cell Embedding Distribution

- 主类别: Loss / Regularization
- 辅助类别: Latent / Dynamics Modeling
- 可信度: high
- 代码位置:
  - `src/stack/models/core/base.py:43`: `self.sw_distance = SlicedWassersteinDistance`
  - `src/stack/models/core/losses.py:33`: `_compute_sw_loss`
  - `src/stack/models/core/losses.py:57`: random subsampling of cells
  - `src/stack/models/core/losses.py:59`: Gaussian prior samples
  - `src/stack/models/core/losses.py:60`: centered embeddings
  - `src/stack/modules/regularizers.py:10`: `SlicedWassersteinDistance`
  - `src/stack/modules/regularizers.py:22`: random projection directions
  - `src/stack/modules/regularizers.py:28`: sorted one-dimensional projections
- 设计摘要: Match the distribution of centered cell embeddings to Gaussian samples through sliced Wasserstein distance over random projections.
- 解决的问题: Latent spaces can drift, collapse, or become poorly calibrated across cell sets. A distributional prior regularizes latent geometry without requiring labels.
- 机制细节:
  - 输入: final cell embeddings `[B, n_cells, latent_dim]`.
  - 输出: scalar SW loss.
  - 关键张量/分布: centered embeddings vs `torch.randn_like` prior samples.
  - 训练时行为: total loss adds `sw_weight * sw_loss`.
  - 推理时行为: no direct operation, but latent geometry is more regular.
- 优势: Cheaper than full OT, differentiable, distribution-level, and independent of pair labels.
- 为什么适合细胞扰动预测: Full-cell perturbation prediction benefits from a smooth, comparable latent space across control and perturbed populations.
- 迁移到 `nemo_cellflow`:
  - 建议落点: `models/<MODEL_NAME>/nemo_model.py::loss_fn`, applied to encoded `src_encoded_map`, `tgt_encoded_map`, or predicted latent maps.
  - 需要新增的 `prepare_model` 配置: `sw_weight`, `sw_n_proj`, `sw_subsample_size`, `sw_apply_to`.
  - 是否需要修改 `dsets/`: no.
  - 是否兼容 NeMo/BioNeMo 接口: yes.
- 风险与限制: Matching to Gaussian may conflict with biologically structured manifolds if overweighted. It should regularize, not dominate.
- 最小 ablation: `sw_weight=0` vs small values; apply to source-only, target-only, and predicted latent maps.
- 迁移优先级: P0

### A05. Dataset-Type-Aware Replacement as Perturbation-Style Pseudo-Pair Construction

- 主类别: Task / Data Formulation
- 辅助类别: Inference / Calibration
- 可信度: high
- 代码位置:
  - `src/stack/data/finetuning/datasets.py:35`: `DatasetConfig` supports `human` and `drug`
  - `src/stack/data/finetuning/datasets.py:63`: group column differs by dataset type
  - `src/stack/data/finetuning/datasets.py:68`: identity column differs by dataset type
  - `src/stack/data/finetuning/datasets.py:756`: `find_replacement_cells`
  - `src/stack/data/finetuning/datasets.py:838`: human uses `DIFFERENT_GROUP`
  - `src/stack/data/finetuning/datasets.py:839`: drug uses `CONTROL_GROUP`
  - `src/stack/data/finetuning/datasets.py:961`: control condition filtering
  - `src/stack/data/finetuning/datasets.py:1392`: returns ground truth tensor
  - `src/stack/data/finetuning/datasets.py:1393`: returns observed/replacement tensor
- 设计摘要: Build pseudo-paired fine-tuning samples by replacing a fraction of cells with biologically constrained alternatives: for human data use same identity from another group/donor; for drug data use same identity from control condition.
- 解决的问题: Real paired control/treated cells are rarely available. The model needs a training signal for translating between observed context and target distribution without exact one-to-one cell pairs.
- 机制细节:
  - 输入: metadata with group, identity, condition, dataset type.
  - 输出: `(ground_truth_tensor, observed_tensor, cell_type_ids, position_mask, metadata)`.
  - 关键张量/分布: replacement map from original cells to control/different-group cells; `position_mask` marking first occurrence after balancing.
  - 训练时行为: student observes replacement/context data and is supervised toward ground truth cells.
  - 推理时行为: analogous context construction is used for base/test in-context prediction.
- 优势: Turns unpaired population data into a structured translation problem while preserving cell identity constraints.
- 为什么适合细胞扰动预测: Perturbation prediction commonly needs same-cell-type control-to-treated mapping. Control-condition replacement is directly aligned with drug/perturbation setups.
- 迁移到 `nemo_cellflow`:
  - 建议落点: do not change `dsets/` by default; mimic the idea inside model loss by treating existing `ctrl_cell_emb` as observed replacement/context and `pert_cell_emb` as ground truth.
  - 需要新增的 `prepare_model` 配置: `use_replacement_style_loss`, `replacement_context_ratio`, `identity_condition_weight`.
  - 是否需要修改 `dsets/`: no for conceptual transfer; yes only if reproducing Stack's sampler exactly, which is not recommended by default.
  - 是否兼容 NeMo/BioNeMo 接口: yes if implemented as loss logic over existing batch.
- 风险与限制: Exact Stack sampler relies on metadata not guaranteed in JIT batches. Avoid importing its data pipeline unless user explicitly asks.
- 最小 ablation: existing paired JIT loss vs paired loss plus replacement-style context masking over current batch.
- 迁移优先级: P1

### A06. Query Positional Embedding for Unknown Target Cells

- 主类别: Condition Fusion
- 辅助类别: Inference / Calibration
- 可信度: high
- 代码位置:
  - `src/stack/models/finetune/mixins.py:24`: `query_pos_embedding`
  - `src/stack/models/finetune/mixins.py:29`: gradient scaling hook
  - `src/stack/models/finetune/model.py:55`: query mask expansion
  - `src/stack/models/finetune/model.py:58`: add query embedding to target tail cells
  - `src/stack/models/core/inference.py:178`: query embedding during prediction
  - `src/stack/models/core/inference.py:181`: applies to cells after context boundary
- 设计摘要: Add a learned query token/positional embedding to cells that should be generated or predicted, distinguishing them from prompt/context cells inside the same attention window.
- 解决的问题: In a mixed context batch, model must know which cells are observed anchors and which cells are targets to infer.
- 机制细节:
  - 输入: token tensor `[B, n_cells, n_hidden, token_dim]`, context boundary `n_context_cell`.
  - 输出: target-tail tokens shifted by learned query embedding.
  - 关键张量/分布: `query_pos_embedding` broadcast over target cells and gene tokens.
  - 训练时行为: target tail cells receive query embedding and are supervised by distribution losses.
  - 推理时行为: test cells after context receive the same query marker.
- 优势: Cleanly encodes "this cell is a query to generate" without changing input feature dimensions or requiring new dataloader fields.
- 为什么适合细胞扰动预测: Control-to-perturbed prediction can frame perturbed cells as query cells conditioned on control/context cells.
- 迁移到 `nemo_cellflow`:
  - 建议落点: `models/<MODEL_NAME>/networks/_condition_encoders.py` or `_nemo_vf.py`, adding a learned query/context token to latent cell maps.
  - 需要新增的 `prepare_model` 配置: `use_query_embedding`, `query_embedding_scale`, `context_ratio`.
  - 是否需要修改 `dsets/`: no if context/query split is created inside `loss_fn`.
  - 是否兼容 NeMo/BioNeMo 接口: yes.
- 风险与限制: Needs a reliable convention for which cells in the set are context vs target. JIT batches may not currently order cells by role.
- 最小 ablation: with vs without query embedding under identical context masking.
- 迁移优先级: P0

### A07. Frozen-Teacher Fine-Tuning with Embedding Distillation

- 主类别: Loss / Regularization
- 辅助类别: Latent / Dynamics Modeling
- 可信度: high
- 代码位置:
  - `src/stack/finetune/lightning.py:16`: `LightningFinetunedModel`
  - `src/stack/finetune/lightning.py:31`: student model
  - `src/stack/finetune/lightning.py:32`: teacher model
  - `src/stack/finetune/lightning.py:34`: freeze teacher
  - `src/stack/finetune/lightning.py:52`: sync teacher weights at fit start
  - `src/stack/finetune/lightning.py:112`: teacher forward on ground truth
  - `src/stack/finetune/lightning.py:122`: detached target embeddings
  - `src/stack/finetune/lightning.py:124`: student forward with target embeddings
  - `src/stack/finetune/lightning.py:276`: EMA update helper
- 设计摘要: Maintain a frozen teacher initialized from the student/pretrained checkpoint. Teacher embeds ground-truth profiles; student predicts from observed/replaced inputs and matches teacher embeddings through downstream losses.
- 解决的问题: Fine-tuning on noisy pseudo-pairs can destabilize representation. Teacher embeddings provide a stable target space.
- 机制细节:
  - 输入: `ground_truth_features`, `observed_features`, metadata masks.
  - 输出: student prediction and loss terms; teacher target embeddings are detached.
  - 关键张量/分布: `target_embeddings = teacher_output['final_cell_embeddings'].detach()`.
  - 训练时行为: teacher is frozen; optional EMA update exists.
  - 推理时行为: only student is needed.
- 优势: Stabilizes fine-tuning while preserving pretrained representation geometry.
- 为什么适合细胞扰动预测: When adapting to PBMC/Replogle/Tahoe, teacher targets can prevent catastrophic drift while learning perturbation-specific generation.
- 迁移到 `nemo_cellflow`:
  - 建议落点: `models/<MODEL_NAME>/nemo_model.py`, creating an optional frozen teacher copy of the encoder/decoder or loading a baseline JIT checkpoint as teacher.
  - 需要新增的 `prepare_model` 配置: `use_frozen_teacher`, `teacher_ckpt`, `teacher_loss_weight`, `teacher_ema_decay`.
  - 是否需要修改 `dsets/`: no.
  - 是否兼容 NeMo/BioNeMo 接口: risky but feasible. Need careful state dict loading under Megatron/NeMo and avoid duplicating huge model memory.
- 风险与限制: Memory overhead can be high in NeMo/DDP. Teacher checkpoint format must be compatible.
- 最小 ablation: no teacher vs frozen teacher vs EMA teacher; compare embedding alignment and downstream perturbation metrics.
- 迁移优先级: P1

### A08. Energy-Distance MMD on Generated Count Distributions and Embeddings

- 主类别: Loss / Regularization
- 辅助类别: Decoder / Distribution Head; Evaluation / Diagnostics
- 可信度: high
- 代码位置:
  - `src/stack/models/finetune/mixins.py:22`: `SamplesLoss(loss="energy")`
  - `src/stack/models/finetune/mixins.py:102`: `_compute_mmd_loss`
  - `src/stack/models/finetune/mixins.py:127`: select top residual-variance genes if many genes
  - `src/stack/models/finetune/mixins.py:147`: predicted NB mean for replaced cells
  - `src/stack/models/finetune/mixins.py:152`: reparameterized NB log sampler
  - `src/stack/models/finetune/mixins.py:157`: true normalized log counts
  - `src/stack/models/finetune/mixins.py:185`: cell-type-stratified matching
  - `src/stack/models/finetune/mixins.py:218`: time-scaled combined distribution/embedding loss
- 设计摘要: Compare predicted and true populations using energy-distance MMD, both in expression space and embedding space, optionally stratified by cell type and position mask.
- 解决的问题: Cell perturbation outputs are distributions over cells. Per-cell MSE can be misleading under unpaired or pseudo-paired data.
- 机制细节:
  - 输入: NB parameters, ground truth counts, library sizes, predicted embeddings, teacher embeddings, cell type IDs.
  - 输出: scalar MMD loss plus predicted/true distributions for diagnostics.
  - 关键张量/分布: sampled log-normalized NB counts vs true log-normalized counts; predicted vs teacher embeddings.
  - 训练时行为: loss applies to cells after `n_context_cell`; if `position_mask` exists, match within cell types.
  - 推理时行为: validation can compute `sw_predict` between predicted and true distributions.
- 优势: Directly optimizes population-level similarity and handles unpaired target cells better than pointwise losses.
- 为什么适合细胞扰动预测: Benchmarks often evaluate perturbation-level distributions and DE behavior. MMD/energy losses align predictions with target population distributions.
- 迁移到 `nemo_cellflow`:
  - 建议落点: `models/<MODEL_NAME>/nemo_model.py::loss_fn`, supplementing current MSE/MMD metrics.
  - 需要新增的 `prepare_model` 配置: `mmd_weight`, `mmd_kernel: energy`, `mmd_space: expression|latent|both`, `mmd_celltype_stratified`.
  - 是否需要修改 `dsets/`: no for unstratified expression/latent MMD; yes/risky for cell-type stratification if IDs absent.
  - 是否兼容 NeMo/BioNeMo 接口: yes for unstratified loss.
- 风险与限制: `geomloss` may not be installed in current environment; using existing `_metrics.MMDLoss` in nemo_cellflow may be safer. Stratification requires metadata.
- 最小 ablation: pointwise loss only vs expression MMD vs expression+latent MMD.
- 迁移优先级: P0

### A09. Query/Tail Classifier for Confidence-Guided Generation

- 主类别: Inference / Calibration
- 辅助类别: Loss / Regularization
- 可信度: medium
- 代码位置:
  - `src/stack/models/finetune/mixins.py:35`: `self.cls`
  - `src/stack/models/finetune/mixins.py:66`: `_compute_cls_loss`
  - `src/stack/models/finetune/mixins.py:74`: context mean expanded to mid/tail cells
  - `src/stack/models/finetune/mixins.py:83`: concatenate mid and tail logits
  - `src/stack/models/core/inference.py:189`: logits computed for tail cells
  - `src/stack/models/core/inference.py:695`: use logits in generation mode
  - `src/stack/models/core/inference.py:699`: quantile threshold for keeping generated cells
- 设计摘要: Train a binary classifier to distinguish mid/context-like cells from tail/query cells using context mean plus cell embedding. During generation, logits drive which generated cells remain masked or accepted.
- 解决的问题: Iterative generation needs a confidence/selection mechanism for deciding which cells are sufficiently explained by the current context.
- 机制细节:
  - 输入: mean context embedding, mid cell embeddings, tail cell embeddings.
  - 输出: logits and BCE loss; generation-time logit scores.
  - 关键张量/分布: concatenated `[context_mean, cell_embedding]` features.
  - 训练时行为: BCE with class-balancing `pos_weight`.
  - 推理时行为: quantile threshold and positive logits update `is_masked`.
- 优势: Adds a learned calibration signal without requiring external labels beyond constructed context/tail positions.
- 为什么适合细胞扰动预测: Could serve as uncertainty/confidence for generated perturbed cells or conditions.
- 迁移到 `nemo_cellflow`:
  - 建议落点: optional confidence head in `models/<MODEL_NAME>/networks/_nemo_vf.py` and inference loop.
  - 需要新增的 `prepare_model` 配置: `use_confidence_head`, `confidence_loss_weight`, `confidence_threshold`.
  - 是否需要修改 `dsets/`: no if labels are synthetic from context/query positions.
  - 是否兼容 NeMo/BioNeMo 接口: yes, but inference scripts would need to consume logits if used at evaluation time.
- 风险与限制: The semantic meaning of mid vs tail labels is construction-dependent; may not directly map to perturbation confidence.
- 最小 ablation: iterative generation with no confidence filter vs confidence quantile filter.
- 迁移优先级: P2

### A10. Iterative In-Context Generation Schedule

- 主类别: Inference / Calibration
- 辅助类别: Latent / Dynamics Modeling
- 可信度: high
- 代码位置:
  - `src/stack/models/core/inference.py:841`: `get_incontext_generation`
  - `src/stack/models/core/inference.py:866`: derive `num_steps` from `mask_rate`
  - `src/stack/models/core/inference.py:869`: step schedule `t`
  - `src/stack/models/core/inference.py:873`: context ratio schedule
  - `src/stack/models/core/inference.py:874`: mask ratio schedule
  - `src/stack/models/core/inference.py:885`: vanilla repeated prediction
  - `src/stack/models/core/inference.py:912`: non-vanilla confidence-guided loop
  - `src/stack/models/core/inference.py:932`: update `test_adata.X` or `raw.X` each step
- 设计摘要: Generate profiles iteratively. Each step predicts target cells using current base/test state, updates test expression, and follows a decreasing mask-rate plus increasing context-ratio schedule.
- 解决的问题: One-shot generation can be brittle for large distribution shifts. Iterative refinement can progressively fill or denoise target populations.
- 机制细节:
  - 输入: base AnnData, test AnnData, prompt/context ratios, mask rate, number of steps.
  - 输出: generated expression matrix and optional logits.
  - 关键张量/分布: `mr_list = 1 - t`, `cr_list = linspace(context_ratio_min, context_ratio, num_steps)`.
  - 训练时行为: no direct training, but relies on query/context training.
  - 推理时行为: replace test matrix with generated result after each step.
- 优势: Simple schedule-based refinement analogous to denoising without requiring a full diffusion model.
- 为什么适合细胞扰动预测: Control-to-perturbed prediction could benefit from gradual movement from control-like state toward target-like state, especially for difficult OOD perturbations.
- 迁移到 `nemo_cellflow`:
  - 建议落点: `main_inference_*` or model `eval_fn` only after training support exists; for training, add consistency across multiple inference steps.
  - 需要新增的 `prepare_model` 配置: `iterative_inference_steps`, `context_ratio_min`, `context_ratio_max`, `mask_schedule`.
  - 是否需要修改 `dsets/`: no.
  - 是否兼容 NeMo/BioNeMo 接口: yes for inference, but project inference wrappers must be updated if this becomes a new behavior.
- 风险与限制: Updating generated test expression in-place can accumulate errors. Must not use test target information for scheduling.
- 最小 ablation: one-step inference vs 3/5-step iterative inference; fixed context ratio vs increasing context ratio.
- 迁移优先级: P1

## 4. Transfer Ranking

| Rank | Atom | Priority | Expected value for nemo_cellflow | Implementation cost | Risk |
| --- | --- | --- | --- | --- | --- |
| 1 | A08 Energy-Distance MMD on Generated Count Distributions and Embeddings | P0 | Aligns with perturbation population metrics and can reuse existing MMD-style utilities | Low-Medium | Metadata stratification optional; dependency on geomloss avoidable |
| 2 | A04 Sliced Wasserstein Regularization of Cell Embedding Distribution | P0 | Regularizes latent maps for stable source/target population geometry | Low | Overweighting may wash out biological structure |
| 3 | A06 Query Positional Embedding for Unknown Target Cells | P0 | Clear context/query conditioning mechanism for control-to-perturbed generation | Medium | Needs deterministic role split inside batch |
| 4 | A01 Cell-by-Gene Tabular Attention | P0 | Strong architecture idea for cell-set and gene-program interaction | Medium-High | Memory and input-semantics adaptation |
| 5 | A02 Rectangular Gene Masking Across Cell Sets | P0 | Adds self-supervised gene-program reconstruction regularizer | Medium | NB loss depends on target scale |
| 6 | A07 Frozen-Teacher Fine-Tuning with Embedding Distillation | P1 | Stabilizes adaptation from baseline checkpoint | Medium-High | Duplicate model memory under NeMo/Megatron |
| 7 | A03 Library-Size-Aware NB Distribution Head | P1 | Better count likelihood and uncertainty | Medium | Requires count-like targets |
| 8 | A10 Iterative In-Context Generation Schedule | P1 | Practical inference refinement idea | Medium | Needs inference wrapper changes and error-control |
| 9 | A05 Dataset-Type-Aware Replacement Pseudo-Pairs | P1 | Useful conceptual data formulation for unpaired perturbation | High if copied literally | Should not alter `dsets/` by default |
| 10 | A09 Query/Tail Classifier for Confidence-Guided Generation | P2 | Optional confidence signal for iterative generation | Medium | Semantics may not transfer cleanly |

## 5. Recommended Combinations

### Combo 1. Distribution-Aligned JIT

- Atoms: A04 + A08
- Hypothesis: JIT latent flow will improve perturbation-level distribution fidelity if target/predicted latent or expression populations are regularized with SW/MMD losses in addition to pointwise flow loss.
- Where to implement:
  - `models/<MODEL_NAME>/nemo_model.py::loss_fn`
  - optionally `models/<MODEL_NAME>/networks/_nemo_vf.py` for latent outputs
- Required config knobs:
  - `sw_weight`
  - `sw_n_proj`
  - `mmd_weight`
  - `mmd_space: latent|expression|both`
- Minimal experiment: PBMC split0 or Replogle rpe1 with same JIT config, comparing baseline JIT vs SW only vs MMD only vs SW+MMD.

### Combo 2. Context-Query Perturbation Generator

- Atoms: A01 + A06 + A08
- Hypothesis: Treat control cells as context and perturbed cells as query targets. A tabular attention encoder with query embedding and distribution loss can better model control-conditioned perturbation populations.
- Where to implement:
  - `models/<MODEL_NAME>/networks/_tabular_attention.py`
  - `models/<MODEL_NAME>/networks/_condition_encoders.py`
  - `models/<MODEL_NAME>/nemo_model.py::loss_fn`
- Required config knobs:
  - `use_tabular_attention`
  - `use_query_embedding`
  - `context_ratio`
  - `mmd_weight`
- Minimal experiment: Replogle rpe1 with `cell_sentence_len=32`, because Stack-style set attention is cheaper there than Tahoe.

### Combo 3. Masked Reconstruction Auxiliary Pretraining

- Atoms: A02 + A03 or A02 + Gaussian/MSE alternative
- Hypothesis: Masked gene-column reconstruction on control/perturbed batches can improve gene module understanding before or during flow matching.
- Where to implement:
  - `models/<MODEL_NAME>/nemo_model.py::loss_fn`
  - decoder under `models/<MODEL_NAME>/networks/`
- Required config knobs:
  - `masked_recon_weight`
  - `mask_rate_min`
  - `mask_rate_max`
  - `decoder_distribution`
- Minimal experiment: run with NB only if data are raw/count-like; otherwise use Gaussian/MSE masked reconstruction.

### Combo 4. Teacher-Stabilized Adaptation

- Atoms: A07 + A08
- Hypothesis: A frozen JIT teacher can preserve baseline latent geometry while a student learns stronger distribution matching.
- Where to implement:
  - `models/<MODEL_NAME>/nemo_model.py::configure_model`
  - `models/<MODEL_NAME>/nemo_model.py::loss_fn`
- Required config knobs:
  - `use_frozen_teacher`
  - `teacher_ckpt`
  - `teacher_embedding_weight`
  - `mmd_weight`
- Minimal experiment: single dataset/cell line first due to memory risk; compare strict no-teacher baseline against frozen-teacher student.

## 6. Do Not Transfer Blindly

- Do not copy Stack's full dataloader/replacement system into `nemo_cellflow` by default. The project protocol says dataset code should remain unchanged unless explicitly requested.
- Do not assume `ctrl_cell_emb` and `pert_cell_emb` are raw UMI counts. NB loss requires count-like nonnegative targets and meaningful library sizes.
- Do not treat CLI/config/logging/checkpoint helper code as algorithm atoms.
- Do not import PyTorch Lightning training loops into `nemo_cellflow`; new models must remain compatible with NeMo/BioNeMo/Megatron entrypoints.
- Do not use test target expression when adapting iterative generation schedules. Stack updates generated test expression, but a benchmark-safe implementation must avoid target leakage.
- Do not enable teacher duplication on large models without a memory audit.

## 7. Evidence Gaps

- Notebooks were not executed; generation behavior is inferred from `src/stack/models/core/inference.py` and `src/stack/cli/generation.py`.
- No Stack paper text was read beyond README; atom claims are code-based.
- Actual pretrained checkpoints and dataset files were not available, so runtime behavior and reported biological performance were not verified.
- The repo is a shallow clone at commit `cacc2e4`; history and alternate branches were not inspected.
- `nemo_cellflow` batch semantics may differ from Stack's raw count matrix assumption; NB-head transfer requires target-scale verification.
