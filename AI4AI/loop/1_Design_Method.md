# 1. Design Method Protocol

本文档定义 Codex 在本项目中自动设计新算法时必须遵循的 protocol。`nemo_cellflow` 是一个基于 NVIDIA NeMo / BioNeMo 训练栈的项目，新算法必须保持与默认 repo 一致的 NeMo/BioNeMo/Megatron 接口，而不是改成独立 PyTorch 或普通 PyTorch Lightning 训练流程。目标不是做轻微调参，而是在理解项目架构和数据约束的前提下，读取 `AI4AI/examples` 中所有已蒸馏出的算法原子设计作为背景知识，再组合成 `nemo_cellflow` 可训练、可推理、可评估的新算法。

## 0. 不可变约束

1. 数据集代码默认不变更。
   - 不修改 `dsets/` 下的数据集、datamodule、LMDB 读取逻辑。
   - 新算法必须复用已有 `dataset_name` 和 `dataset` 配置路径。
   - 如果某个 idea 需要新的样本字段、额外标注或新数据预处理，默认判定为不适合本轮算法实现；除非用户明确要求改数据管线。

2. 模型代码必须放在 `models/<MODEL_NAME>/`。
   - 训练入口通过 `model_name` 动态导入模型：
     `models.<MODEL_NAME>.nemo_model.<MODEL_NAME>_Nemo_Model`
   - 因此 YAML 中的 `model_name`、模型目录名、`nemo_model.py` 中的类名必须完全一致。
   - 新模型必须兼容默认 repo 的 NeMo/BioNeMo 接口：由 `MInterface` 实例化，交给 `nemo.collections.llm.train`，并使用项目已有 trainer、strategy、precision、optimizer、checkpoint 和 logger 流程。
   - 如果从 `models/JIT` 复制模板，必须清理所有旧 import，例如 `models.JIT...`，改成新模型自己的包路径，除非该 import 是明确复用共享模块并在设计文档中说明原因。

3. 新算法 idea 不能只停留在参数或网络宽度的小改动。
   - 不接受仅改变 `hidden_dims`、`dropout`、`num_heads`、`dit_depth`、学习率、batch size、epoch 等作为新方法。
   - 新方法至少要包含一个结构性机制变化，例如条件编码、latent dynamics、训练目标、gene/cell 表征、inference solver、regularization、prior/correction、multi-view alignment 等。
   - 推荐每个新方法由 2-4 个彼此独立的优秀算法原子设计组合而成，并且每个算法原子设计都能说明来源、动机、映射位置和风险。

4. 训练、推理、评估提交必须遵守项目 RJOB 规范。
   - 提交前阅读 `AI4AI/loop/2_Rjob_Submit.md`。
   - 提交前阅读 `RJOB_SUBMISSION_RULES.md`。
   - 仅提交任务时，不修改代码、配置或 wrapper。
   - 只有当用户明确要求提交 live RJOB 时，才执行 `bash submit_*_rjob.sh ...`。

## 1. 项目架构事实

训练入口是 `main_train.py`：

```text
OmegaConf.load(config) -> DInterface(conf) + MInterface(conf) -> llm.train(...)
```

这是 NeMo/BioNeMo 风格的训练入口：`DInterface` 提供 NeMo datamodule，`MInterface` 提供继承项目 `ModelInterfaceBase` 的 NeMo model，最终由 `nemo.collections.llm.train` 统一接管训练。新算法只替换或扩展模型实现，不替换项目训练入口和 NeMo/BioNeMo 生命周期。

数据加载由 `dataset_name` 决定：

```text
dataset_name: tahoe    -> dsets.tahoe.nemo_datamodule.tahoe_Nemo_Datamodule
dataset_name: pbmc     -> dsets.pbmc.nemo_datamodule.pbmc_Nemo_Datamodule
dataset_name: replogle -> dsets.replogle.nemo_datamodule.replogle_Nemo_Datamodule
```

模型加载由 `model_name` 决定：

```text
model_name: JIT      -> models.JIT.nemo_model.JIT_Nemo_Model
model_name: NEW_ALGO -> models.NEW_ALGO.nemo_model.NEW_ALGO_Nemo_Model
```

标准参考实现：

```text
模型目录: /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/models/JIT
参考配置: /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/config/jit_tahoe.yaml
```

JIT 的核心接口约定：

- `nemo_model.py` 继承 `ModelInterfaceBase`。
- 训练、验证、推理接口保持与默认 JIT repo 一致，能被 NeMo/BioNeMo 的 trainer、Megatron strategy、loss reduction、checkpoint save/load 调用。
- `data_step` 从 dataloader 取 batch，并移动到 GPU。
- `loss_fn(batch)` 返回 `{"loss": final_loss, "log_info": ...}`。
- `eval_fn(batch, test_mode=False)` 返回 validation 指标或预测。
- `predict_step` 在推理时返回 perturbation prediction。
- batch 中默认使用 `ctrl_cell_emb`、`pert_cell_emb`，条件字段来自 `layers_before_pool`，例如 `pert_onehot`、`batch_onehot`。
- `prepare_model` 是模型结构和训练机制的主要配置区。
- `Interpolant` 与 `flow/` 下的概率路径代码属于 JIT 的 latent flow 框架；新方法可以复用，也可以在新模型目录内替换或扩展。

## 2. 默认数据配置

新算法默认复制以下 YAML 作为数据与训练配置基座，只改 `model_name`、模型相关 `prepare_model` 字段、`experiment.group`、必要 checkpoint 字段和资源字段。不要改默认数据路径，除非用户明确要求。

### Tahoe

```text
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/config/jit_tahoe.yaml
```

默认标识：

```yaml
dataset_name: tahoe
model_name: JIT
```

### PBMC few-shot 5 splits

```text
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/config/jit_pbmc_split0.yaml
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/config/jit_pbmc_split1.yaml
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/config/jit_pbmc_split2.yaml
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/config/jit_pbmc_split3.yaml
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/config/jit_pbmc_split4.yaml
```

默认标识：

```yaml
dataset_name: pbmc
model_name: JIT
```

### Replogle 4 cell lines

```text
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/config/jit_replogle_hepg2.yaml
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/config/jit_replogle_jurkat.yaml
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/config/jit_replogle_k562.yaml
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/config/jit_replogle_rpe1.yaml
```

默认标识：

```yaml
dataset_name: replogle
model_name: JIT
```

## 3. 新算法设计流程

### Step 1: 明确任务卡

Codex 先把用户请求转成一个任务卡：

```text
算法目标:
主 benchmark: tahoe / pbmc / replogle / all
基准配置:
新模型名:
是否允许改数据代码: 默认 no
是否允许新增外部依赖: 默认 no
是否需要从 checkpoint 初始化:
计算预算:
预期输出: 设计文档 / 模型代码 / YAML / RJOB 提交 / 评估汇总
```

如果用户没有给新模型名，Codex 应提出一个全大写、可 import 的名字，例如 `JIT_STATEALIGN`、`FLOW_MATCH_PRIOR`、`PERT_TOKEN_FUSION`。目录名和类名前缀必须与该名字一致。

### Step 2: 读取 examples 中已蒸馏的算法原子设计作为背景知识

在动手写代码前，Codex 必须先读取以下目录中所有已蒸馏出的算法原子设计，把它们作为背景知识：

```text
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/AI4AI/examples/*_distill.md
```

这些文件由 `AI4AI/loop/0_Distall_Papers.md` 生成，是新算法设计的背景知识库。Codex 设计新方法时不应临时跳过 distill 文件直接从 repo 做印象式总结；如果 `AI4AI/examples` 中没有对应 repo 的 `*_distill.md`，应先按 `0_Distall_Papers.md` 对该 repo 完成蒸馏，再进入本 protocol。

每个候选算法原子设计必须记录为：

```text
来源 distill 文件:
来源 repo / 代码位置:
算法原子设计:
解决的问题:
为什么它优秀:
映射到本项目的位置:
需要的 batch 字段:
是否触碰数据代码:
主要风险:
最小 ablation:
```

合格算法原子设计的标准：

- 是算法机制设计，不是单纯超参或工程实现习惯。
- 能落到 `models/<MODEL_NAME>/` 内，不依赖数据代码变更。
- 与 perturbation prediction 的核心问题相关：control-to-perturbed mapping、条件扰动编码、cell state alignment、gene representation、latent flow、distribution matching、OOD generalization、uncertainty 或 inference calibration。
- 可以被单独 ablation，证明它不是无意义装饰。

建议在阅读 `AI4AI/examples/*_distill.md` 作为背景知识后，整理至少 3 个候选算法原子设计，最终选择 2-4 个组合成新算法。

### Step 3: 合成项目内算法

Codex 根据算法原子设计写出一个算法 spec：

```text
算法名:
一句话假设:
继承/参考的 baseline:
保留的数据接口:
新增模块:
修改的 forward/loss/inference:
新增 YAML 字段:
预期提升的数据集或场景:
失败风险:
ablation 计划:
```

合成时必须回答：

- 新算法相对 JIT 的结构性变化是什么？
- 它改变的是 condition encoder、cell map encoder/decoder、flow target、loss、prior、solver、regularization，还是推理校准？
- 它为什么可能提升 Tahoe、PBMC 或 Replogle？
- 它是否仍只需要 `ctrl_cell_emb`、`pert_cell_emb` 和已有 covariate keys？
- 它如何在不修改 `dsets/` 的情况下运行？

如果这些问题不能回答，不能进入实现阶段。

## 4. 实现规范

### 4.1 模型目录

新模型放在：

```text
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/models/<MODEL_NAME>/
```

最常用做法是复制 `models/JIT` 后改造：

```text
models/<MODEL_NAME>/nemo_model.py
models/<MODEL_NAME>/flow/
models/<MODEL_NAME>/networks/
```

必须满足：

```text
models/<MODEL_NAME>/nemo_model.py
class <MODEL_NAME>_Nemo_Model(...)
```

`model_interface.py` 会按如下方式导入：

```python
importlib.import_module(f"models.{model_name}.nemo_model")
getattr(module, f"{model_name}_Nemo_Model")
```

因此任何大小写、下划线不一致都会导致训练启动失败。

### 4.2 import 规范

新模型内部 import 优先使用自己的包路径：

```python
from models.<MODEL_NAME>.flow.interpolant import Interpolant
from models.<MODEL_NAME>.networks._nemo_vf import ConditionalVelocityField
import models.<MODEL_NAME>.networks._basic as nn_basic
```

禁止复制后遗留无意的旧路径：

```python
from models.JIT...
import models.JIT...
from models.PRIOR...
```

除非设计明确要共享旧模块，否则必须改成新模型路径。实现后用下面命令检查：

```bash
rg "models\\.(JIT|PRIOR|SCALE|STATE|ENDPOINT|NB)" models/<MODEL_NAME>
```

如果命中，逐条判断是否有设计理由。

### 4.3 配置规范

每个新算法至少需要一个 YAML，例如：

```text
config/<lower_model_name>_tahoe.yaml
config/<lower_model_name>_pbmc_split0.yaml
config/<lower_model_name>_replogle_rpe1.yaml
```

从对应 JIT YAML 复制后修改：

```yaml
dataset_name: <保持原值>
model_name: <MODEL_NAME>

prepare_model:
  # 保留必要 JIT 字段
  # 添加新算法字段

experiment:
  group: <model>-<dataset>-<split-or-cell>
```

原则：

- `dataset_name` 不因新算法而改变。
- `dataset` 下的 LMDB 路径不改。
- `layers_before_pool` 与 dataloader 输出字段保持一致。
- `prepare_model.megatron_ckpt` 只有在 checkpoint 结构兼容时才保留；结构改动后默认设为 `null`，避免 strict load 失败。
- `experiment.group` 必须唯一，方便 checkpoint、wandb offline 目录和评估结果追踪。

### 4.4 代码接口

新模型必须保留以下接口，并与默认 `models/JIT` 的 NeMo/BioNeMo 调用方式一致：

```python
def data_step(self, dataloader_iter): ...
def configure_model(self): ...
def loss_reduction(self, *args, **kwargs): ...
def forward_step(self, batch, mode="train"): ...
def training_step(self, batch, batch_idx=None): ...
def validation_step(self, batch, batch_idx=None): ...
def predict_step(self, batch, batch_idx=None): ...
def loss_fn(self, batch): ...
def eval_fn(self, batch, test_mode=False): ...
```

接口约束：

- `nemo_model.py` 中的模型类必须继承项目的 `ModelInterfaceBase`，不要改成裸 `torch.nn.Module` 或自定义训练脚本。
- `loss_reduction` 必须能被 NeMo/Megatron training loop 调用，并返回项目兼容的 loss reduction 对象。
- `training_step`、`validation_step`、`predict_step` 的返回结构要兼容当前训练、推理和评估 wrapper。
- checkpoint 加载和保存要遵守默认 repo 的 Megatron/NeMo state dict 约定；结构不兼容时不要强行复用旧 checkpoint。
- 不新增绕过 `main_train.py`、`MInterface`、`DInterface`、`llm.train` 的 parallel training path。

batch shape 要兼容 JIT 当前逻辑：

```text
ctrl_cell_emb: [B, S, D] 或 [B*S, D]
pert_cell_emb: [B, S, D] 或 [B*S, D]
condition keys: 来自 layers_before_pool，例如 pert_onehot、batch_onehot
```

不要假设实际 batch size 一定等于 YAML 中的 `dataset.batch_size`；最后一个 batch 可能更小。

### 4.5 新机制落点

推荐优先在这些位置做新算法，而不是改数据：

```text
models/<MODEL_NAME>/networks/_condition_encoders.py
models/<MODEL_NAME>/networks/_nemo_vf.py
models/<MODEL_NAME>/flow/interpolant.py
models/<MODEL_NAME>/nemo_model.py::loss_fn
models/<MODEL_NAME>/nemo_model.py::eval_fn
```

常见有效方向：

- perturbation/covariate 条件编码更强，例如 FiLM、cross-attention、token fusion、state-aware conditioning。
- control 与 perturbed cell map 的 alignment，例如 residual state alignment、contrastive alignment、distribution matching。
- latent flow 目标改造，例如预测 `x1`、预测 `v`、hybrid target、time-dependent consistency。
- decoder/gene representation 改造，例如 gene-wise adapter、low-rank decoder、gene graph prior，但不能要求新数据字段。
- inference correction，例如 train-only prior delta、endpoint calibration、solver consistency，但要避免使用 test target 信息。

## 5. 实验设计规范

新算法不是只跑一个成功训练。每轮需要有可解释实验梯度：

1. Smoke check
   - Python import 能通过。
   - YAML 能被 OmegaConf 解析。
   - 模型能实例化到 `MInterface(conf)`，如果依赖 GPU 或 NeMo 环境不足，要记录原因。

2. 单配置训练
   - 先选一个代表性配置，例如 PBMC split0、Replogle rpe1 或 Tahoe。
   - 与同数据配置的 JIT baseline 对齐比较。

3. Ablation
   - 每个算法原子设计至少一个关闭开关或替代路径。
   - ablation 写在 YAML 字段中，而不是手改代码产生不可追踪版本。

4. 多数据集验证
   - PBMC: split0-4。
   - Replogle: hepg2、jurkat、k562、rpe1。
   - Tahoe: `jit_tahoe.yaml` 对应数据。

5. 推理与评估
   - 训练完成后按 `RJOB_SUBMISSION_RULES.md` 选择推理 wrapper。
   - 每个评估 input_dir 只能放一对 h5ad 文件。
   - 不同 split/cell line 的评估结果放不同文件夹。

## 6. RJOB 使用约束

训练提交：

```bash
cd /mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow
bash submit_train_rjob.sh <job-name> <config-path>
```

推理提交：

```bash
bash submit_inference_pbmc_rjob.sh <job-name> <config-path>
bash submit_inference_replogle_rjob.sh <job-name> <config-path>
bash submit_inference_tahoe_rjob.sh <job-name> <config-path>
```

评估提交：

```bash
bash submit_evaluate_pbmc_rjob.sh <input-dir> <job-name>
bash submit_evaluate_replogle_rjob.sh <input-dir> <job-name>
bash submit_evaluate_tahoe_rjob.sh <input-dir> <job-name>
```

注意评估参数顺序与训练/推理不同：

```text
training/inference: job-name first, config-path second
evaluation: input-dir first, job-name second
```

当用户只要求“设计新方法”时，不提交 RJOB。当用户明确要求“提交训练/推理/评估”时，严格使用 `AI4AI/loop/2_Rjob_Submit.md` 中的 prompt 形式和 `RJOB_SUBMISSION_RULES.md` 中的 wrapper。

## 7. Codex 执行模板

当用户要求“基于项目自动设计新算法”时，Codex 按以下顺序执行：

```text
1. 读取本文件、AI4AI/loop/2_Rjob_Submit.md、RJOB_SUBMISSION_RULES.md。
2. 读取 main_train.py、models/model_interface.py、dsets/data_interface.py。
3. 读取 models/JIT 和目标 dataset 的 JIT YAML。
4. 明确任务卡，包括数据集、模型名、允许改动范围和输出物。
5. 读取 AI4AI/examples 中所有 `*_distill.md` 作为背景知识，并整理至少 3 个候选算法原子设计。
6. 选择 2-4 个算法原子设计，合成一个新算法 spec。
7. 检查该 spec 是否不改数据代码、不是轻微调参、能被 ablation。
8. 创建 models/<MODEL_NAME>/ 和 config/<...>.yaml。
9. 修正 import、类名、YAML model_name、experiment.group。
10. 做 smoke check 和静态检查。
11. 如果用户要求提交，再按 RJOB 规范提交训练/推理/评估。
12. 输出设计说明、改动文件、检查结果和下一步实验建议。
```

## 8. 交付物清单

一次合格的新算法设计至少交付：

```text
算法 spec:
  - 算法名
  - 来源算法原子设计
  - 项目内映射
  - 训练目标
  - 推理路径
  - ablation 计划

代码:
  - models/<MODEL_NAME>/...

配置:
  - config/<model>_<dataset>.yaml
  - 必要时覆盖 PBMC 5 splits、Replogle 4 cell lines、Tahoe

验证:
  - import/path 检查
  - YAML 解析检查
  - 不修改 dsets/ 的确认
  - stale import 检查
  - 如果运行过训练/推理/评估，记录 job-name 和 config/input_dir
```

## 9. 设计质量检查表

进入实现前：

- [ ] 是否明确使用哪些默认数据配置？
- [ ] 是否保持 `dataset_name` 不变？
- [ ] 是否不改 `dsets/`？
- [ ] 是否读取了 `AI4AI/examples` 中所有已蒸馏出的算法原子设计作为背景知识？
- [ ] 是否不是单纯调参？
- [ ] 是否写清每个算法原子设计的来源、作用和风险？
- [ ] 是否能映射到 `models/<MODEL_NAME>/` 内的具体文件？
- [ ] 是否能做 ablation？

实现后：

- [ ] `models/<MODEL_NAME>/nemo_model.py` 中存在 `<MODEL_NAME>_Nemo_Model`。
- [ ] YAML 顶层 `model_name` 与目录名、类名前缀一致。
- [ ] YAML 顶层 `dataset_name` 与所用数据集一致。
- [ ] `dataset` 下默认路径未被无意修改。
- [ ] `prepare_model.megatron_ckpt` 与新结构兼容；不兼容则为 `null`。
- [ ] 没有无意遗留旧模型 import。
- [ ] `loss_fn`、`eval_fn`、`predict_step` 能处理 JIT batch shape。
- [ ] 训练、推理、评估提交命令符合 `RJOB_SUBMISSION_RULES.md`。

## 10. 禁止事项

- 禁止把“新算法”定义成只改学习率、层数、宽度、dropout、batch size 或 epoch。
- 禁止为了适配 idea 默认修改数据集代码。
- 禁止在未说明来源的情况下编造外部 repo 设计。
- 禁止 YAML `model_name` 和模型类名不一致。
- 禁止复制 JIT 后留下无意的 `models.JIT...` import。
- 禁止在 submission-only 请求中改代码或配置。
- 禁止用户未明确要求时提交 live RJOB。
