# 数据处理优化

为减少adata大量细胞信息随机存取带来的IO开销，将细胞按照规则划分set后分块存储，训练时可连续读取。每个sample信息大致如下

```python
sample =
{
    "cartesian_key":  ('plate2', 'CVCL_1550', "[('ML264', 0.5, 'uM')]"),
    "cell_matrix": sp.csr_matrix(pert_sub) # shape为(1024, 2000)的HVG稀疏矩阵
}
```

control细胞信息作为字典独立存储

```python
control_dict = {
     ('plate2', 'CVCL_1550', "[('DMSO_TF', 0.0, 'uM')]"): sp.csr_matrix(control) # 保存所有该类control细胞HVG的稀疏矩阵
}
```
## Tahoe-100M

### 第一步：提取HVG

查看 tahoe/extract 文件夹下相关内容

### 第三步：提取关注条件的global_keys

运行 tahoe/group/read_global_keys.ipynb

### 第二步：按关注条件的笛卡尔积划分细胞

运行 tahoe/group/make_group.ipynb

### 第三步：划分训练/评估/测试数据集，存储为lmdb格式

使用toml文件进行划分，运行 group/train_val_test.ipynb