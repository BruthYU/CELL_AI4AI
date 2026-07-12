# RJOB Submission Rules

This file defines the project-local RJOB submission rules for Codex before handling vague training, inference, or evaluation submission requests.

Project root:

```bash
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow
```

## Scope

When the user asks Codex to submit, rerun, launch, or execute a training/inference/evaluation RJOB in this project:

- Read this file first.
- Do not add new files.
- Do not modify existing code, configs, or shell wrappers.
- Only adapt the command line to the existing submission scripts listed below.
- Do not invent default config paths or input directories.
- Do not submit a live RJOB unless the user explicitly asks to submit.

If the requested job cannot be mapped to the existing wrappers, ask for clarification instead of editing files.

## Training

Use the existing training submit wrapper:

```bash
cd /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow
bash submit_train_rjob.sh <job-name> <config-path>
```

Argument order:

```text
1. job-name
2. config-path
```

Example:

```bash
bash submit_train_rjob.sh prior-llm-pbmc-split2 config/prior_llm_pbmc_split2.yaml
```

Rules:

- Use `submit_train_rjob.sh` only.
- Do not use old script names such as `run_success.sh` or `run_multi_node.sh`.
- Do not edit `main_train.py` or config files for a submission-only request.
- If `config-path` is missing, ask the user for it.

## Inference

Use one of the existing inference submit wrappers:

```bash
cd /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow
bash submit_inference_pbmc_rjob.sh <job-name> <config-path>
bash submit_inference_replogle_rjob.sh <job-name> <config-path>
bash submit_inference_tahoe_rjob.sh <job-name> <config-path>
```

Argument order:

```text
1. job-name
2. config-path
```

Examples:

```bash
bash submit_inference_pbmc_rjob.sh infer-pbmc-split2 config/prior_llm_pbmc_split2.yaml
bash submit_inference_replogle_rjob.sh infer-replogle-rpe1 config/prior_llm_replogle_rpe1.yaml
bash submit_inference_tahoe_rjob.sh infer-tahoe config/jit_fp_tahoe_film_fingerprint.yaml
```

Rules:

- Choose the wrapper by dataset: `pbmc`, `replogle`, or `tahoe`.
- Replogle cell line is inferred by `main_inference_replogle.py` from the config unless the user explicitly asks for an override.
- Tahoe config is passed through `--config` inside the wrapper.
- Do not edit inference scripts or configs for a submission-only request.
- If `config-path` is missing, ask the user for it.

## Evaluation

Use one of the existing evaluation submit wrappers:

```bash
cd /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow
bash submit_evaluate_pbmc_rjob.sh <input-dir> <job-name>
bash submit_evaluate_replogle_rjob.sh <input-dir> <job-name>
bash submit_evaluate_tahoe_rjob.sh <input-dir> <job-name>
```

Argument order:

```text
1. input-dir
2. job-name
```

Examples:

```bash
bash submit_evaluate_pbmc_rjob.sh /path/to/pbmc_workspace eval-pbmc-split2
bash submit_evaluate_replogle_rjob.sh /path/to/replogle_workspace eval-replogle-rpe1
bash submit_evaluate_tahoe_rjob.sh /path/to/tahoe_workspace eval-tahoe
```

Rules:

- Choose the wrapper by dataset: `pbmc`, `replogle`, or `tahoe`.
- PBMC split is inferred by `benchmark/evaluate_pbmc.py` from `pbmc_real_<split>.h5ad` and `pbmc_pred_<split>.h5ad`.
- Replogle cell line is inferred by `benchmark/evaluate_replogle.py` from `replogle_real_<cell>.h5ad` and `replogle_pred_<cell>.h5ad`.
- Tahoe input directory is passed explicitly.
- Evaluation argument order is different from training and inference.
- Do not edit evaluation scripts or configs for a submission-only request.
- If `input-dir` is missing, ask the user for it.

## Ambiguous Requests

For vague instructions, map intent as follows:

```text
"提交训练" / "跑训练"     -> submit_train_rjob.sh
"提交推理" / "跑推理"     -> submit_inference_<dataset>_rjob.sh
"提交评估" / "跑评估"     -> submit_evaluate_<dataset>_rjob.sh
```

Required information:

```text
training:   dataset optional, job-name, config-path
inference:  dataset, job-name, config-path
evaluation: dataset, input-dir, job-name
```

If `job-name` is missing, Codex may propose one from dataset, task, config name, split, or cell line. The name must not contain spaces.

If `dataset`, `config-path`, or `input-dir` is missing and cannot be uniquely inferred from the user's exact message, ask for clarification.

## Do Not Change Files

For submission-only requests, Codex must not:

- create new scripts,
- rename scripts,
- modify configs,
- modify Python files,
- modify existing wrappers,
- add defaults,
- change hard-coded resources,
- change checkpoint paths,
- change input/output directory logic.

The only allowed action is adapting and executing one of the existing `bash submit_*_rjob.sh ...` commands, after confirming the required arguments.

File changes are allowed only when the user explicitly asks for implementation or script changes.

## Sandbox Execution

When executing `rjob submit` through the existing project wrappers:

- Run the selected `bash submit_*_rjob.sh ...` command from the project root with escalated execution approval directly, outside the sandbox.
- Do not make an initial sandbox attempt for live RJOB submissions, because the sandbox may not reach the RJOB API host such as `h.pjlab.org.cn`.
- Do not edit wrappers, configs, or Python files to work around sandbox network restrictions.
- Keep the submitted job name and arguments exactly as determined from the request, unless the user explicitly requests a change.
