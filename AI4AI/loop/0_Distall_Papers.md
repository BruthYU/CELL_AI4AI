# 0. Distill Repository Protocol

本文档定义 Codex 如何从一个优秀代码 repo 中蒸馏“细胞扰动预测方法”的可迁移算法原子设计。输出不是普通论文摘要，也不是 README 复述，更不是工程实现风格总结，而是面向 `nemo_cellflow` 新算法设计的算法原子库：encoder、decoder、condition fusion、分布建模、distribution loss、latent dynamics、inference calibration 等。

本 protocol 的输出会作为 `AI4AI/loop/1_Design_Method.md` 的上游输入，用于把外部 repo 的优秀机制转化为本项目可实现的新算法。

## 0. 核心目标

给定一个 repo 路径或 repo 内文件路径，Codex 需要：

1. 识别该 repo 的任务设定、模型结构、训练目标、推理流程和评估逻辑。
2. 从代码中蒸馏若干独立的优秀算法原子设计。
3. 每个算法原子设计必须有代码证据、优势分析、迁移价值和风险判断。
4. 输出到 repo 同级目录的 `<repo>_distill.md` 文件。

禁止把蒸馏结果写成泛泛的“模型很强”“用了 transformer”“loss 很多”这类描述，也禁止把代码组织、CLI、日志、配置管理等工程设计当成核心原子。每条算法原子设计必须回答：它解决什么建模问题、代码在哪里、算法机制是什么、为什么值得迁移、如何迁移到 `nemo_cellflow`。

## 1. 输入与输出约定

### 输入

用户提供以下任一形式：

```text
/path/to/repo
/path/to/repo/some/file.py
/path/to/repo/submodule_or_package
```

Codex 需要先解析 repo root：

1. 如果输入路径在 Git repo 内，使用 `git -C <input> rev-parse --show-toplevel` 得到 repo root。
2. 如果不是 Git repo，向上查找 `pyproject.toml`、`setup.py`、`setup.cfg`、`README.md`、`configs/`、`src/`、`tests/` 等项目根目录信号。
3. 如果仍不能确定，把输入目录本身当作 repo root，并在输出中标记“repo root inferred with low confidence”。

### 输出

输出文件放在 repo root 的同级目录：

```text
<parent_dir>/<repo_basename>_distill.md
```

示例：

```text
输入: /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/AI4AI/examples/stack
输出: /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/AI4AI/examples/stack_distill.md
```

如果用户输入的是 repo 内某个文件：

```text
输入: /path/to/stack/src/stack/model.py
输出: /path/to/stack_distill.md
```

注意：`<repo_basename>` 取 repo root 的目录名，`distill` 固定小写。除非用户另行指定，不要把输出写进 repo 内部，避免污染外部 repo。

## 2. 不可变约束

1. 蒸馏过程默认只读 repo。
   - 不修改被蒸馏 repo 的代码、配置、notebook、README。
   - 只新增或覆盖同级目录的 `<repo>_distill.md`。

2. 必须基于代码证据。
   - 每条算法原子设计至少给出一个代码位置。
   - 优先使用 `path:line`、class/function 名、config key 作为证据。
   - 如果某条设计来自 README 或论文描述但代码中没有找到实现，必须标注为“文档声称，代码证据不足”。

3. 必须面向细胞扰动预测。
   - 蒸馏重点是 control-to-perturbed prediction、gene/cell representation、perturbation/covariate conditioning、distribution modeling、distribution matching、generative modeling、OOD generalization、uncertainty、inference calibration。
   - 与建模机制无关的通用工程技巧只放在“工程附录”，不要混入核心算法原子。工程技巧不能替代算法原子设计。

4. 必须判断能否迁移到 `nemo_cellflow`。
   - 默认不修改 `dsets/`。
   - 默认保持 `nemo_cellflow` 的 NeMo/BioNeMo/Megatron 接口。
   - 如果算法原子设计需要新增数据字段、外部数据库、预训练权重或复杂依赖，要在风险中明确标注。

## 3. Repo 读取流程

### Step 1: 建立文件地图

先用轻量命令建立 repo 结构：

```bash
rg --files <repo_root>
```

重点记录：

```text
README / docs:
pyproject / setup / requirements:
configs:
src / package:
models:
losses:
datasets / dataloaders:
train scripts:
inference scripts:
evaluation:
tests:
notebooks:
```

忽略或谨慎读取：

```text
.git/
checkpoints/
weights/
data/
wandb/
outputs/
__pycache__/
large generated files
```

### Step 2: 理解任务入口

找到训练、推理、评估入口：

```text
train.py / main.py / scripts/train*
predict.py / infer.py / evaluate.py
config files
model construction factory
dataset construction factory
loss construction code
```

必须画出最小调用链：

```text
config -> dataset -> model -> loss -> train loop -> inference -> metrics
```

如果调用链不完整，继续用 `rg` 搜索 class/function 名，直到能说明模型如何被训练和调用。

### Step 3: 定位核心代码

推荐搜索关键词：

```text
encoder decoder embed embedding condition perturb drug gene cell covariate control treated
vae latent prior posterior distribution gaussian poisson negative_binomial zinb nb
flow diffusion ode sde score matching velocity denoise interpolate
loss nll mse mae kl mmd ot wasserstein energy contrastive consistency correlation pearson
adapter film attention cross_attention transformer graph gnn pathway
predict infer sample evaluate metric de deg delta
```

对于每个命中模块，判断它属于：

```text
数据/任务构造
编码器
条件融合
latent 表征
分布建模
decoder / prediction head
loss / regularization
inference / calibration
evaluation
配置与 ablation
```

### Step 4: 提取候选算法原子设计

候选算法原子设计必须满足至少 3 条：

- 是一个可独立描述的建模机制，而非单纯超参数或工程实现习惯。
- 有明确代码位置。
- 能解释为什么对细胞扰动预测有帮助。
- 能映射到 `nemo_cellflow/models/<MODEL_NAME>/` 的某个实现位置。
- 可以通过 config 开关或小规模 ablation 验证。
- 默认不要求改变 `nemo_cellflow/dsets/`。

不合格例子：

```text
隐藏层更宽
head 数更多
训练 epoch 更多
用了 Adam
用了 dropout
代码写得整洁
```

合格例子：

```text
perturbation token 与 cell state token 用 cross-attention 融合
decoder 输出 NB/ZINB 分布参数而不是点预测
用 MMD/energy distance 约束预测细胞群体分布
用 control delta prior 做 endpoint correction
用 gene embedding + low-rank adapter 做 gene-wise decoder
用 time-conditioned velocity field 进行 latent flow matching
用 perturbation-aware contrastive loss 拉近同扰动细胞状态
```

## 4. 算法原子设计分类法

Codex 蒸馏时按以下类别归档。一个算法原子设计可以属于多个类别，但主类别只能有一个。核心分类必须围绕模型结构、概率分布、训练目标、推理校准或评估诊断，不以工程模块拆分作为主类别。

### 4.1 Task / Data Formulation

关注 repo 如何组织扰动预测任务：

- control / treated pair 构造。
- perturbation、dose、time、cell type、batch 等 covariate 表示。
- few-shot、OOD、unseen perturbation、unseen cell type 的 split 方式。
- 是否使用 pseudo-pair、population-level matching、single-cell distribution target。

### 4.2 Encoder

关注输入如何编码：

- cell encoder: MLP、Transformer、set encoder、DeepSets、GNN、VAE encoder。
- gene encoder: gene embedding、gene graph、pathway prior、gene tokenization。
- perturbation encoder: one-hot、drug/gene embedding、text/LLM embedding、fingerprint、dose embedding。
- covariate encoder: batch、cell type、time、condition、platform。
- time/noise encoder: diffusion/flow matching 中的 time embedding。

### 4.3 Condition Fusion

关注扰动条件如何注入模型：

- concatenation。
- additive bias / residual injection。
- FiLM / modulation。
- cross-attention。
- token fusion。
- adapter / LoRA / low-rank conditioning。
- gating / mixture-of-experts。

### 4.4 Latent / Dynamics Modeling

关注预测过程的结构假设：

- endpoint delta prediction。
- residual prediction。
- latent VAE。
- flow matching / velocity field。
- diffusion / score model。
- ODE/SDE solver。
- optimal transport coupling。
- source-to-target interpolation。
- population-level generative model。

### 4.5 Decoder / Distribution Head

关注输出如何从 latent 转成 gene expression：

- deterministic point decoder。
- gene-wise decoder。
- factorized low-rank decoder。
- NB / ZINB / Poisson / Gaussian 参数头。
- mean + variance uncertainty head。
- quantile / mixture density head。
- denoising decoder。

### 4.6 Loss / Regularization

关注监督信号：

- MSE / MAE / Huber。
- negative log likelihood: NB、ZINB、Poisson、Gaussian。
- KL divergence。
- MMD / energy distance。
- Wasserstein / OT loss。
- contrastive / triplet / supervised contrastive。
- Pearson / correlation / delta-correlation loss。
- DE gene weighted loss。
- consistency / cycle consistency。
- reconstruction + perturbation prediction multi-task loss。
- adversarial / batch-invariance regularization。

### 4.7 Inference / Calibration

关注训练后如何生成和校准预测：

- deterministic endpoint。
- sampling / ensemble。
- ODE/SDE steps。
- prior delta correction。
- control baseline blending。
- train-only perturbation prior。
- uncertainty filtering。
- test-time normalization，但必须避免使用 test target 信息。

### 4.8 Evaluation / Diagnostics

关注 repo 如何判断方法有效：

- whole-gene MSE/MAE/correlation。
- delta Pearson。
- DE gene metrics。
- distribution metrics。
- perturbation-level aggregation。
- OOD split reports。
- ablation table generation。

## 5. 算法原子设计卡片模板

每条算法原子设计都用卡片记录。推荐编号：

```text
A01, A02, A03...
```

模板：

```markdown
### A01. <算法原子设计名称>

- 主类别:
- 辅助类别:
- 可信度: high / medium / low
- 代码位置:
  - `<relative/path.py:line>`: `<class_or_function_or_config_key>`
  - `<relative/path.yaml:line>`: `<config_key>`
- 设计摘要:
- 解决的问题:
- 机制细节:
  - 输入:
  - 输出:
  - 关键张量/分布:
  - 训练时行为:
  - 推理时行为:
- 优势:
- 为什么适合细胞扰动预测:
- 迁移到 `nemo_cellflow`:
  - 建议落点:
  - 需要新增的 `prepare_model` 配置:
  - 是否需要修改 `dsets/`: no / yes, reason
  - 是否兼容 NeMo/BioNeMo 接口: yes / risky, reason
- 风险与限制:
- 最小 ablation:
- 迁移优先级: P0 / P1 / P2
```

填写要求：

- `代码位置` 不能空。
- `机制细节` 必须讲清楚 encoder/decoder/loss/distribution 中至少一个具体机制。
- `优势` 要从机制推导，不要写“效果好”但没有原因。
- `迁移到 nemo_cellflow` 要具体到文件级别，例如：

```text
models/<MODEL_NAME>/networks/_condition_encoders.py
models/<MODEL_NAME>/networks/_nemo_vf.py
models/<MODEL_NAME>/flow/interpolant.py
models/<MODEL_NAME>/nemo_model.py::loss_fn
models/<MODEL_NAME>/nemo_model.py::eval_fn
config/<model>_<dataset>.yaml::prepare_model
```

## 6. 输出文件结构

`<repo>_distill.md` 必须采用以下结构：

````markdown
# <repo_basename> Distillation

## 0. Metadata

- Input path:
- Repo root:
- Output file:
- Git remote:
- Git commit:
- Inspection date:
- Distiller:
- Scope:

## 1. Repo Summary

- Task:
- Data modality:
- Prediction target:
- Training objective:
- Inference mode:
- Main dependencies:
- Relevant entrypoints:

## 2. Minimal Call Graph

```text
config -> dataset -> model -> loss -> train -> inference -> evaluation
```

## 3. Atomic Design Cards

### A01. ...

## 4. Transfer Ranking

| Rank | Atom | Priority | Expected value for nemo_cellflow | Implementation cost | Risk |
| --- | --- | --- | --- | --- | --- |

## 5. Recommended Combinations

### Combo 1. <name>

- Atoms:
- Hypothesis:
- Where to implement:
- Required config knobs:
- Minimal experiment:

## 6. Do Not Transfer Blindly

- ...

## 7. Evidence Gaps

- ...
````

### Metadata 细节

如果是 Git repo，记录：

```bash
git -C <repo_root> remote -v
git -C <repo_root> rev-parse HEAD
```

如果不是 Git repo：

```text
Git remote: unknown
Git commit: unknown
```

### Transfer Ranking 标准

优先级定义：

```text
P0: 强建议迁移。机制明确、代码证据强、默认不改数据、适合 NeMo/BioNeMo 接口。
P1: 值得尝试。可能需要中等代码改动或有一定风险。
P2: 仅作为灵感。依赖额外数据、外部权重、复杂依赖，或与 nemo_cellflow 结构差距较大。
```

## 7. 蒸馏执行模板

当用户要求蒸馏某个 repo 时，Codex 按下面流程执行：

```text
1. 读取 AI4AI/loop/0_Distall_Papers.md。
2. 解析用户输入路径，确定 repo root。
3. 计算输出路径: <repo_parent>/<repo_basename>_distill.md。
4. 用 rg --files 建立文件地图。
5. 读取 README、配置、训练入口、模型实现、loss、dataset、inference、evaluation。
6. 画出 minimal call graph。
7. 抽取候选算法原子设计。
8. 过滤掉无代码证据、纯超参、纯工程实现、难以迁移或与扰动预测无关的候选。
9. 用算法原子设计卡片写入 `<repo>_distill.md`。
10. 写 transfer ranking、recommended combinations、do-not-transfer、evidence gaps。
11. 最后检查输出文件是否在 repo 同级目录，且每个 atom 都有代码位置。
```

## 8. 推荐 Codex Prompt

```text
阅读 /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/AI4AI/loop/0_Distall_Papers.md，
蒸馏以下 repo 的细胞扰动预测优秀算法原子设计，并把结果写到 repo 同级目录的 <repo>_distill.md：

[repo路径或repo内文件路径]
```

示例：

```text
阅读 /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/AI4AI/loop/0_Distall_Papers.md，
蒸馏以下 repo 的细胞扰动预测优秀算法原子设计，并把结果写到 repo 同级目录的 <repo>_distill.md：

/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/AI4AI/examples/stack
```

预期输出：

```text
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/AI4AI/examples/stack_distill.md
```

## 9. 质量检查表

输出前必须检查：

- [ ] 输出文件路径是否为 repo 同级目录的 `<repo_basename>_distill.md`？
- [ ] 是否记录 input path、repo root、git remote、git commit？
- [ ] 是否画出 config -> dataset -> model -> loss -> train -> inference -> evaluation 的调用链？
- [ ] 是否至少覆盖 encoder、decoder、distribution/loss、inference/calibration 中的关键机制？
- [ ] 每个算法原子设计是否有代码位置？
- [ ] 每个算法原子设计是否说明优势，而不是只描述代码存在？
- [ ] 每个算法原子设计是否判断能否迁移到 `nemo_cellflow`？
- [ ] 是否标注需要修改 `dsets/` 的风险？
- [ ] 是否标注 NeMo/BioNeMo 接口兼容性？
- [ ] 是否给出最小 ablation？
- [ ] 是否给出 transfer ranking 和 recommended combinations？
- [ ] 是否列出证据不足或未读完的部分？

## 10. 禁止事项

- 禁止无代码证据地编造算法原子设计。
- 禁止把 README 摘要当成蒸馏结果。
- 禁止把纯超参数变化或纯工程设计写成优秀算法原子设计。
- 禁止忽略输出路径约定，把 `<repo>_distill.md` 写到错误目录。
- 禁止默认修改被蒸馏 repo。
- 禁止默认修改 `nemo_cellflow/dsets/` 才能迁移。
- 禁止把与细胞扰动预测建模无关的通用技巧放进核心算法原子列表。
