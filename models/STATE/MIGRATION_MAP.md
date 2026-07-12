# State 模型迁移到 Nemo 框架的关系映射

本文档详细说明 `nemo_cellflow/models/STATE/` 中的代码与原始 `state-main` 仓库的对应关系。

## 📁 目录结构对比

### 原始 State 仓库 (`state-main/`)
```
state-main/
└── src/
    └── state/
        └── tx/
            └── models/
                ├── state_transition.py      ← 核心模型实现
                ├── base.py                  ← 基类定义
                ├── context_mean.py
                ├── decoder_only.py
                ├── embed_sum.py
                ├── perturb_mean.py
                ├── old_neural_ot.py
                ├── pseudobulk.py
                ├── decoders.py
                ├── decoders_nb.py
                └── utils.py
```

### 迁移后的 Nemo 版本 (`nemo_cellflow/models/STATE/`)
```
nemo_cellflow/models/STATE/
├── nemo_model.py              ← 🆕 Nemo 适配器层（新增）
└── networks/
    ├── state_transition.py    ← ✅ 从原始仓库移植（几乎不变）
    ├── base.py                ← ✅ 从原始仓库移植
    ├── context_mean.py        ← ✅ 从原始仓库移植
    ├── decoder_only.py        ← ✅ 从原始仓库移植
    ├── embed_sum.py           ← ✅ 从原始仓库移植
    ├── perturb_mean.py        ← ✅ 从原始仓库移植
    ├── old_neural_ot.py       ← ✅ 从原始仓库移植
    ├── pseudobulk.py          ← ✅ 从原始仓库移植
    ├── decoders.py            ← ✅ 从原始仓库移植
    ├── decoders_nb.py         ← ✅ 从原始仓库移植
    └── utils.py               ← ✅ 从原始仓库移植
```

## 🔄 核心迁移关系

### 1. 原始模型类 → 迁移后模型类

| 原始 State 仓库 | 迁移后 Nemo 版本 | 状态 |
|---------------|-----------------|------|
| `state-main/src/state/tx/models/state_transition.py`<br>`StateTransitionPerturbationModel` | `nemo_cellflow/models/STATE/networks/state_transition.py`<br>`StateTransitionPerturbationModel` | ✅ **几乎完全一致**<br>注释说明：`Ported from state-main/src/state/tx/models/state_transition.py` |

### 2. 关键代码对比

#### 原始 State 模型 (`state-main/src/state/tx/models/state_transition.py`)

```python
class StateTransitionPerturbationModel(PerturbationModel):
    """
    This model:
      1) Projects basal expression and perturbation encodings into a shared latent space.
      2) Uses an OT-based distributional loss (energy, sinkhorn, etc.) from geomloss.
      3) Enables cells to attend to one another, learning a set-to-set function rather than
      a sample-to-sample single-cell map.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, pert_dim, ...):
        # 原始实现
        ...
    
    def forward(self, batch: dict, padded=True):
        # 原始前向传播逻辑
        ...
```

#### 迁移后的 Nemo 版本 (`nemo_cellflow/models/STATE/networks/state_transition.py`)

```python
class StateTransitionPerturbationModel(PerturbationModel):
    """
    Ported "st" model (state transition) from `state-main/src/state/tx/models/state_transition.py`.

    NOTE: This is intentionally kept as close as possible to the original implementation.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, pert_dim, ...):
        # 与原始实现几乎完全一致
        ...
    
    def forward(self, batch: dict, padded=True):
        # 与原始实现几乎完全一致
        ...
```

**关键差异：**
- ✅ **模型核心逻辑完全保留**：`StateTransitionPerturbationModel` 的实现与原始版本几乎完全一致
- ✅ **注释明确标注来源**：代码注释明确说明来自 `state-main/src/state/tx/models/state_transition.py`
- ✅ **设计意图保持**：注释强调 "intentionally kept as close as possible to the original"

## 🆕 新增的适配层：`STATE_Nemo_Model`

### 设计目的

`STATE_Nemo_Model` 是一个**适配器（Adapter）**，将原始 State 模型包装为符合 Nemo 框架接口的 Lightning 模块。

### 架构关系图

```
┌─────────────────────────────────────────────────────────────┐
│  Nemo 框架 (Megatron + Lightning)                             │
│  - 分布式训练支持                                             │
│  - 数据并行/模型并行                                          │
│  - 配置系统 (YAML)                                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  STATE_Nemo_Model (适配器层)                                  │
│  📍 nemo_cellflow/models/STATE/nemo_model.py                │
│                                                              │
│  职责：                                                       │
│  1. 数据格式转换 (pert_onehot → pert_emb)                    │
│  2. 设备/并行管理                                             │
│  3. 配置参数映射                                              │
│  4. Lightning 接口实现                                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ 调用 self.module.forward()
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  StateTransitionPerturbationModel (原始模型)                 │
│  📍 nemo_cellflow/models/STATE/networks/state_transition.py  │
│                                                              │
│  ✅ 与 state-main/src/state/tx/models/state_transition.py   │
│     几乎完全一致                                              │
│                                                              │
│  核心组件：                                                   │
│  - pert_encoder: MLP 编码扰动                                 │
│  - basal_encoder: MLP 编码基础表达                           │
│  - transformer_backbone: GPT2 等 Transformer                │
│  - project_out: MLP 输出投影                                 │
│  - loss_fn: Energy/Sinkhorn 分布损失                         │
└─────────────────────────────────────────────────────────────┘
```

### 关键适配点

#### 1. 数据格式转换 (`data_step`)

**原始 State 训练流程：**
- 直接使用 `pert_onehot` [B, vocab] 作为输入
- 在模型内部通过 `pert_encoder` 转换为 `pert_emb` [B, S, pert_dim]

**Nemo 适配流程：**
```python
# nemo_model.py: data_step()
pert_onehot [B, vocab] 
    ↓ (可选编码)
pert_vec [B, pert_dim]
    ↓ (扩展到序列)
pert_emb [B, S, pert_dim]  ← 匹配 State 模型输入约定
    ↓
传递给 StateTransitionPerturbationModel.forward()
```

#### 2. 配置参数映射 (`configure_model`)

**原始 State 配置：**
- 使用 TOML 配置文件
- 参数直接传递给模型构造函数

**Nemo 配置：**
- 使用 YAML 配置文件
- `STATE_Nemo_Model` 负责将 YAML 配置转换为模型参数

```python
# nemo_model.py: configure_model()
kwargs = {
    "input_dim": self.cell_dim,      # 从 YAML 解析
    "hidden_dim": self.hidden_dim,   # 从 YAML 解析
    "pert_dim": self.pert_dim,       # 从 YAML 解析
    ...
}
self.module = StateTransitionPerturbationModel(**kwargs)
```

#### 3. 训练循环适配

**原始 State：**
- 使用 PyTorch Lightning 的 `training_step`/`validation_step`
- 直接在模型类中实现

**Nemo 适配：**
- `STATE_Nemo_Model` 实现 Lightning 接口
- 内部调用 `self.module.forward()` 和 `self.module.loss_fn()`

## 📊 文件对应关系表

| 功能模块 | 原始 State 仓库 | Nemo 迁移版本 | 变更说明 |
|---------|----------------|--------------|---------|
| **核心模型** | `state-main/src/state/tx/models/state_transition.py` | `nemo_cellflow/models/STATE/networks/state_transition.py` | ✅ 几乎不变 |
| **基类** | `state-main/src/state/tx/models/base.py` | `nemo_cellflow/models/STATE/networks/base.py` | ✅ 几乎不变 |
| **工具函数** | `state-main/src/state/tx/models/utils.py` | `nemo_cellflow/models/STATE/networks/utils.py` | ✅ 几乎不变 |
| **解码器** | `state-main/src/state/tx/models/decoders.py` | `nemo_cellflow/models/STATE/networks/decoders.py` | ✅ 几乎不变 |
| **Nemo 适配器** | ❌ 不存在 | `nemo_cellflow/models/STATE/nemo_model.py` | 🆕 **新增** |

## 🔍 代码验证

迁移后的代码在注释中明确标注了来源：

```python
# nemo_cellflow/models/STATE/networks/state_transition.py:73
"""
Ported "st" model (state transition) from `state-main/src/state/tx/models/state_transition.py`.

NOTE: This is intentionally kept as close as possible to the original implementation.
"""
```

## 📝 总结

1. **核心模型保持不变**：`StateTransitionPerturbationModel` 及其相关网络模块与原始 State 仓库几乎完全一致
2. **新增适配层**：`STATE_Nemo_Model` 是唯一新增的代码，负责将原始模型适配到 Nemo 框架
3. **向后兼容**：默认配置 (`use_raw_pert_onehot=True`) 确保与原始 State 训练流程完全一致
4. **可扩展性**：适配层支持多种 State 模型变体（通过 `MODEL_CLASSES` 字典）

## 🔗 相关文件路径

- **原始 State 仓库**：`/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/remote-chen/state-main/`
- **迁移后 Nemo 版本**：`/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/remote-chen/nemo_cellflow/models/STATE/`
- **对比测试脚本**：`/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/remote-chen/nemo_cellflow/compare_state_nemo_dataflow_smoke.py`
