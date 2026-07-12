import pickle
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

plate = "plate9"
pkl_path = "/mnt/shared-storage-user/beam/yulang/remote-tang/tahoe-x1/my_tahoe/var_dims_tahoe.pkl"
raw_h5ad_path = f"/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/datasets/by_plate/h5ad/{plate}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
out_h5ad_path = f"/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/datasets/by_plate/hvg/{plate}_hvg.h5ad"


print(f"plate:{plate}")
print("1) 读取 HVG 列表...")
with open(pkl_path, "rb") as f:
    var_dims = pickle.load(f)

hvg_genes = list(var_dims["gene_names"][: var_dims["hvg_dim"]])

print("2) 读取原始 h5ad...")
adata = sc.read_h5ad(raw_h5ad_path)  # 如果数据超大可考虑 backed="r"，但写出前仍需加载子集

print("3) 计算 HVG 在 adata.var_names 中的位置（向量化）...")
var_index = pd.Index(adata.var_names)
idx = var_index.get_indexer(hvg_genes)  # 不存在 -> -1
mask = idx >= 0
valid_hvgs = [g for g, ok in zip(hvg_genes, mask) if ok]
valid_idx = idx[mask].astype(np.int64)

if len(valid_hvgs) != len(hvg_genes):
    print(f"⚠️ 警告: PKL 中 {len(hvg_genes)} 个 HVG 只有 {len(valid_hvgs)} 个存在于原始数据中！")
else:
    print("✅ 所有 HVG 均存在于原始数据中。")

print("4) 稀疏友好切片（只切 var 维度）...")
# 这里尽量直接切 X，避免带出不必要的 var/obs 冗余
X_hvg = adata.X[:, valid_idx]

# 确保是稀疏矩阵并转成 CSR（更适合按行访问/写入）
if not sp.issparse(X_hvg):
    X_hvg = sp.csr_matrix(X_hvg)
else:
    X_hvg = X_hvg.tocsr(copy=False)

print("5) 构建新的 AnnData，并把 X 和 obsm['X_hvg'] 都设为同一份稀疏矩阵...")
adata_hvg = ad.AnnData(
    X=X_hvg,
    obs=adata.obs.copy(),
    var=adata.var.iloc[valid_idx].copy(),
)

print("6) 使用scanpy进行处理")
sc.pp.normalize_total(adata_hvg, target_sum=None)
sc.pp.log1p(adata_hvg)


# 注意：这里不 copy，避免重复占用内存/磁盘
adata_hvg.obsm["X_hvg"] = adata_hvg.X

print("7) 写出（压缩）...")
adata_hvg.write_h5ad(out_h5ad_path, compression="gzip")
print(f"✅ 已保存到: {out_h5ad_path}")
print(f"   X type: {type(adata_hvg.X)}, nnz: {adata_hvg.X.nnz if sp.issparse(adata_hvg.X) else 'dense'}")
