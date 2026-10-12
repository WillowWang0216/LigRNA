
# dataset_adapter.py
"""
mlp_model 包的数据集适配器。

提供从项目 manifest 构造 RNALigandDataset 的辅助函数，用于训练/验证/测试。
该模块将数据集构造与训练逻辑分离，便于在不同脚本中复用相同的数据构造流程。
"""
import os
import pandas as pd
import sys
from dataset import RNALigandDataset
# --- 关键修复：路径注入 ---
# 获取当前文件 (dataset_adapter.py) 所在的目录 -> .../XRMLP/mlp_model
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 -> .../XRMLP
project_root = os.path.dirname(current_dir)

# 将根目录加入 sys.path 的第一位，确保能导入根目录下的 dataset.py
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------

def load_manifest(manifest_path: str):
    # 如果未提供 manifest_path，则使用默认的 external_data/manifest.csv
    if manifest_path is None:
        manifest_path = os.path.join('external_data', 'manifest.csv')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"找不到 manifest: {manifest_path}")
    return pd.read_csv(manifest_path)


def get_dataset(split: str = 'train', root_dir: str = '.', manifest_path: str = None):
    """
    构造并返回针对指定 split 的 `RNALigandDataset`。

    参数:
    - split: 'train' / 'val' / 'test'，如果数据集没有 split 列，则返回整个 manifest 对应的数据集。
    - root_dir: 数据集根目录（传给 RNALigandDataset 的 root_dir）。
    - manifest_path: 可选的 manifest 文件路径，未提供时使用默认位置。
    """
    manifest = load_manifest(manifest_path)
    # 如果 manifest 包含 'split' 列并且请求的 split 合法，则按 split 过滤；否则使用完整 manifest
    if 'split' in manifest.columns and split in ('train', 'val', 'test'):
        df = manifest[manifest['split'] == split].reset_index(drop=True)
    else:
        df = manifest
    return RNALigandDataset(root_dir=root_dir, manifest_df=df)


def get_all_datasets(root_dir: str = '.', manifest_path: str = None, pdb_dir: str = None, label_dist_threshold: float = 5.0):
    """
    同时构造 train / val / test 三个数据集并返回。

    如果 manifest 没有 'split' 列，则:
    - train 使用整个 manifest
    - val 返回一个空的 DataFrame（无验证样本）
    - test 使用整个 manifest
    这样可以保证返回的三个对象在 API 上可用。
    """
    manifest = load_manifest(manifest_path)
    train_df = manifest[manifest['split'] == 'train'] if 'split' in manifest.columns else manifest
    val_df = manifest[manifest['split'] == 'val'] if 'split' in manifest.columns else pd.DataFrame(columns=manifest.columns)
    test_df = manifest[manifest['split'] == 'test'] if 'split' in manifest.columns else manifest

    # [修改2] 实例化 Dataset 时传入 pdb_dir
    train_ds = RNALigandDataset(root_dir=root_dir, manifest_df=train_df, pdb_dir=pdb_dir,label_dist_threshold=label_dist_threshold)
    val_ds = RNALigandDataset(root_dir=root_dir, manifest_df=val_df, pdb_dir=pdb_dir,label_dist_threshold=label_dist_threshold)
    test_ds = RNALigandDataset(root_dir=root_dir, manifest_df=test_df, pdb_dir=pdb_dir,label_dist_threshold=label_dist_threshold)
    
    return train_ds, val_ds, test_ds
