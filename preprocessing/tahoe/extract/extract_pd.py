import scanpy as sc
import pandas as pd
from tqdm import tqdm
import os
pwd_folder = os.path.dirname(os.path.abspath(__file__))
print(os.path.exists(f"{pwd_folder}/obs_dataframe"))
plate_num = [2, 4, 5, 6, 7]
pkl_path = "/mnt/shared-storage-user/beam/yulang/remote-tang/tahoe-x1/my_tahoe/var_dims_tahoe.pkl"
for plate_id in tqdm(plate_num):
    raw_h5ad_path = f"/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/datasets/by_plate/hvg/plate{plate_id}_hvg.h5ad"
    adata = sc.read_h5ad(raw_h5ad_path) 
    
    # 可选：保存
    adata.obs.to_csv(f"{pwd_folder}/obs_dataframe/plate{plate_id}_obs.csv", index=False, sep="\t")
    
    del adata