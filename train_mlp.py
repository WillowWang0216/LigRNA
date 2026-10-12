#train_mlp.py

import argparse
import warnings

import torch
import importlib
import os
import sys
import random
import numpy as np
import logging
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from torch.utils.data import DataLoader # 需要显式导入 DataLoader

import dataset_adapter
from models import RNALigandGNNModel
from visualization import plot_all_evaluation_charts
from dataset import RNALigandDataset

# 导入 train 模块
#train_mod = importlib.import_module('mlp_model2_auto.train')
import train as train_mod
#过滤警告
warnings.filterwarnings("ignore", category=FutureWarning)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 强制使用确定性算法 (会牺牲少量性能，但保证完全复现)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_dir):
    """配置 Logger，同时输出到控制台和文件"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    logger = logging.getLogger("RLI-GNN")
    logger.setLevel(logging.INFO)

    # 清除之前的 handlers，防止重复打印
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    # 修改后：强制使用 utf-8 编码保存日志
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)

    # Stream Handler (Console)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging to {log_file}")
    return logger, timestamp

def pre_collate_hybrid(batch_list):
    """
    混合 Collate 函数：
    1. RNA 数据 (Seq): 使用 pad_sequence 进行填充。
    2. Ligand 数据 (Graph): 使用 PyG Batch.from_data_list 进行大图拼接。
    """
    # 过滤 None
    batch_list = [b for b in batch_list if b is not None]
    if not batch_list:
        return None

    rna_seqs = []
    ligand_graphs = []
    labels = []

    for b in batch_list:
        # --- 提取 RNA 特征 ---
        # 兼容多种 dataset 字段名
        if hasattr(b, 'rna_seq_feat'):
            rna = b.rna_seq_feat
        elif hasattr(b, 'rna_node_feat'):
            rna = b.rna_node_feat
        elif hasattr(b, 'rna_graph') and hasattr(b.rna_graph, 'x'):
            rna = b.rna_graph.x
        else:
            rna = torch.zeros((0, 1))

        rna_seqs.append(rna)

        # --- 提取 Ligand 图 ---
        # 确保是 PyG Data 对象
        if hasattr(b, 'ligand_graph'):
            ligand_graphs.append(b.ligand_graph)
        else:
            raise ValueError("Dataset must provide 'ligand_graph' (PyG Data object) for GNN model.")

        # --- 提取标签 ---
        if hasattr(b, 'rna_node_y'):
            lab = b.rna_node_y
        elif hasattr(b, 'node_y'):
            lab = b.node_y
        else:
            lab = torch.zeros(rna.shape[0])

        # 确保 label 是 1D
        if lab.dim() > 1: lab = lab.view(-1)
        labels.append(lab)

    # 1. Pad RNA Sequence & Labels
    # batch_first=True -> (B, L, D)
    rna_padded = pad_sequence(rna_seqs, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    # 生成 RNA Mask (根据实际长度)
    lengths = torch.tensor([len(x) for x in rna_seqs])
    B, max_L = rna_padded.shape[:2]
    mask_rna = torch.arange(max_L).expand(B, max_L) < lengths.unsqueeze(1)

    # 2. Batch Ligand Graphs
    # PyG 会自动处理 edge_index 的偏移
    ligand_batch = Batch.from_data_list(ligand_graphs)

    return {
        'rna_seq': rna_padded,
        'mask_rna': mask_rna,  # (B, L) bool
        'labels': labels_padded,  # (B, L) float
        'ligand_graph': ligand_batch  # PyG Batch Object
    }


def collate_hybrid(batch_list):
    """
    混合 Collate 函数：
    1. RNA 数据 (Seq): 使用 pad_sequence 进行填充。
    2. Ligand 数据 (Graph): 使用 PyG Batch.from_data_list 进行大图拼接。
    3. [新增] Graph Label: 提取 graph_y
    """
    # 过滤 None
    batch_list = [b for b in batch_list if b is not None]
    if not batch_list:
        return None

    rna_seqs = []
    ligand_graphs = []
    labels = []
    graph_labels = []  # [新增] 用于存储图级标签

    for b in batch_list:
        # --- 提取 RNA 特征 ---
        if hasattr(b, 'rna_seq_feat'):
            rna = b.rna_seq_feat
        elif hasattr(b, 'rna_node_feat'):
            rna = b.rna_node_feat
        elif hasattr(b, 'rna_graph') and hasattr(b.rna_graph, 'x'):
            rna = b.rna_graph.x
        else:
            rna = torch.zeros((0, 1))
        rna_seqs.append(rna)

        # --- 提取 Ligand 图 ---
        if hasattr(b, 'ligand_graph'):
            ligand_graphs.append(b.ligand_graph)
        else:
            raise ValueError("Dataset must provide 'ligand_graph'")

        # --- 提取位点标签 (Pixel Label) ---
        if hasattr(b, 'rna_node_y'):
            lab = b.rna_node_y
        elif hasattr(b, 'node_y'):
            lab = b.node_y
        else:
            lab = torch.zeros(rna.shape[0])
        if lab.dim() > 1: lab = lab.view(-1)
        labels.append(lab)

        # --- [新增] 提取图标签 (Graph Label) ---
        # 优先读取 graph_y，如果没有则默认为 1.0 (正样本)
        if hasattr(b, 'graph_y'):
            g_lab = b.graph_y.item() if isinstance(b.graph_y, torch.Tensor) else float(b.graph_y)
        else:
            g_lab = 1.0
        graph_labels.append(g_lab)

    # [新增] 收集 ID
    ids = [b.complex_id if hasattr(b, 'complex_id') else f"unknown_{i}" for i, b in enumerate(batch_list)]
    # [新增] 收集序列
    sequences = [b.sequence if hasattr(b, 'sequence') else "" for b in batch_list]
    # [新增] 收集 PDB 序号列表
    # 如果 data 里没有 pdb_resi (比如旧数据)，就给个空列表防止报错
    pdb_resis = [b.pdb_resi if hasattr(b, 'pdb_resi') else [] for b in batch_list]

    # 1. Pad RNA
    rna_padded = pad_sequence(rna_seqs, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    # Mask
    lengths = torch.tensor([len(x) for x in rna_seqs])
    B, max_L = rna_padded.shape[:2]
    mask_rna = torch.arange(max_L).expand(B, max_L) < lengths.unsqueeze(1)

    # 2. Batch Ligand
    ligand_batch = Batch.from_data_list(ligand_graphs)

    # 3. [新增] Tensorize Graph Labels
    graph_y_tensor = torch.tensor(graph_labels, dtype=torch.float)


    return {
        'rna_seq': rna_padded,
        'mask_rna': mask_rna,
        'labels': labels_padded,
        'ligand_graph': ligand_batch,
        'graph_y': graph_y_tensor,  # [新增] 返回图标签
        'ids':ids,#新增 返回ID列表
        # [新增] 返回序列列表
        'sequences': sequences,
        'pdb_resis': pdb_resis  # <--- [新增修改] 返回这个列表
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--manifest', type=str, default="../external_data/manifestBL_RB9.csv")
    parser.add_argument('--root_dir', type=str, default='.')
    # [新增] 添加 pdb_dir 参数
    parser.add_argument('--pdb_dir', type=str, default="../all_pdb_modifyCount", help='Custom path to PDB directory')
    parser.add_argument('--seed', type=int, default=1000, help='Random seed')
    parser.add_argument('--log_dir', type=str, default='log_RB9', help='Directory to save logs and plots')
    # 优化选项
    parser.add_argument('--loss_type', type=str, default='focal', choices=['bce', 'focal'])
    parser.add_argument('--focal_alpha', type=float, default=0.5, help='Alpha for Focal Loss (0-1)')  # <--- 新增
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma for Focal Loss')

    # BCE 选项 (仅当 loss_type=bce 时有效)
    parser.add_argument('--pos_weight', type=float, default=2.17, help='Manual pos_weight for BCE')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--threshold', type=float, default=0.5 , help='Manual classification threshold (0.0-1.0)')
    # 补齐模型深度参数
    parser.add_argument('--rna_depth', type=int, default=5, help='Depth of RNA ResNet blocks')
    parser.add_argument('--interact_depth', type=int, default=3, help='Depth of Cross-Attention layers')

    # [新增] 控制正负样本距离的参数
    parser.add_argument('--label_dist_threshold', type=float, default=5.0,
                        help='Distance in Angstroms to define positive sites')
    # 新增 Sample Level 参数
    #parser.add_argument('--eval_sample', action='store_true', help='Enable Sample-level evaluation (IoU, etc.)')
    parser.add_argument('--eval_sample', type=str, default='eval_sample', help='Enable Sample-level evaluation (IoU, etc.)')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='Threshold for IoU to consider a sample correctly predicted')
    #是否要输出根据一个正样本核苷酸就输出整个RNA为position(1)
    parser.add_argument('--eval_graph',type=bool, default = True, help='Enable Graph-level (Per-RNA binary classification) evaluation')


 # 🌟 新增：消融实验专用开关
    parser.add_argument('--fusion_type', type=str, default='attention', choices=['attention', 'concat'], help='RNA与配体特征的融合方式')
    parser.add_argument('--without_rna_resnet', action='store_true', help='移除 RNA 1D-ResNet 特征提取器Ablation Study: Disable RNA 1D-ResNet and use simple linear projection')
    # 🌟 在这里添加缺失的参数
    parser.add_argument('--without_chem_env', action='store_true', help='Ablation: Disable chemical environment features')
    parser.add_argument('--without_ensemble', action='store_true', help='Ablation: Skip ensemble stage')

    from datetime import datetime
    args = parser.parse_args()

    # [新增优化] 开启 TensorCore 加速 (针对 A100/V100/RTX30系列以上)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        
    # 按照优先级判断当前运行的是哪种消融
    if args.without_chem_env:
        ab_type = "chemEnv"
    elif args.without_rna_resnet:
        ab_type = "rnaResnet"
    elif args.fusion_type == 'concat':
        ab_type = "fusion"
    elif args.without_ensemble:
        ab_type = "ensemble"
    else:
        ab_type = "fullModel"

    # 临时写死，标注这是一个只用语言模型特征的实验
    #ab_type = "onlyLM" 
    # 将这个标签存入 args 对象的末尾，这样它会自动进入 config.json
    args.ab_type = ab_type

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = "RB9" if "RB9" in args.manifest else "TE18"
    # 动态构建存放路径
    exp_name = f"{timestamp}_{dataset_name}_{args.loss_type}_dist{args.label_dist_threshold}_{ab_type}"
    exp_dir = os.path.join("ab_experiments", f"{ab_type}_experiment", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 2. 覆盖 log_dir，让所有的日志、画出的图片都乖乖存进这个专属文件夹
    args.log_dir = exp_dir

    # 3. 随机种子与日志配置
    seed_everything(args.seed)

    # 因为我们上面已经自己生成了 timestamp，所以这里用 _ 忽略它返回的旧时间戳
    logger, _ = get_logger(args.log_dir)
    logger.info(f"Arguments: {args}")
    logger.info(f"\n\n{'='*50}\n🚀 新实验启动! 所有文件将保存在:\n📁 {exp_dir}\n{'='*50}\n\n")

    # 我们把 Final_Threshold 变成小写，直接从 args 拿，保持整洁
    final_threshold = args.threshold

    # 1. 获取数据集(传入距离参数)
    train_ds, val_ds, test_ds = dataset_adapter.get_all_datasets(
        root_dir=args.root_dir,
        manifest_path=args.manifest,
        pdb_dir=args.pdb_dir,
        label_dist_threshold=args.label_dist_threshold  # <--- [新增]
    )
    logger.info(f"Dataset Loaded: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")


    ##############
    # 3. 推断维度 (Robust Version)
    sample = next((s for s in train_ds if s is not None), None)
    if sample is None: raise RuntimeError("Empty dataset")

    # RNA 维度推断
    if hasattr(sample, 'rna_node_feat'):
        d_rna = sample.rna_node_feat.shape[1]
    else:
        d_rna = sample.rna_graph.x.shape[1]

    # Ligand 节点特征维度
    d_lig_node = sample.ligand_graph.x.shape[1]

    # Ligand 边特征维度 (关键修复)
    # 检查 edge_attr 是否存在且不为 None
    if hasattr(sample.ligand_graph, 'edge_attr') and sample.ligand_graph.edge_attr is not None:
        if sample.ligand_graph.edge_attr.numel() > 0:  # 检查是否为空 tensor
            d_lig_edge = sample.ligand_graph .edge_attr.shape[1]
        else:
            # 如果 tensor 是空的 (例如 tensor([], size=(0, 9)))，尝试读取第二维
            # 如果连 shape 都是 (0,)，那就设为 0
            if len(sample.ligand_graph.edge_attr.shape) > 1:
                d_lig_edge = sample.ligand_graph.edge_attr.shape[1]
            else:
                logger.warning("Sample has empty edge_attr. Setting d_lig_edge=0. This might cause issues with GATv2.")
                d_lig_edge = 0
    else:
        logger.warning("Sample has NO edge_attr. Setting d_lig_edge=0.")
        d_lig_edge = 0

    logger.info(f"Dims: RNA={d_rna}, Ligand Node={d_lig_node}, Ligand Edge={d_lig_edge}")
    logger.info(f"\n*****pos_weight{args.pos_weight}*******")
    # 3. 初始化模型
    # 深度配置: Ligand GNN=3, Cross=2, Predictor=2. Hidden=128.
    # 修改dropout
    # model = RNALigandGNNModel(d_rna, d_lig_node, d_lig_edge, hidden_dim=args.hidden, dropout=0.4,fusion_type=args.fusion_type,use_rna_resnet=not args.without_rna_resnet)

    model = RNALigandGNNModel(
        d_rna=d_rna, 
        d_lig_node=d_lig_node, 
        d_lig_edge=d_lig_edge,
        hidden_dim=args.hidden,
        dropout=0.4,         # (同上)
        rna_depth=args.rna_depth,     
        interact_depth=args.interact_depth, 
        fusion_type=args.fusion_type,
        use_rna_resnet=not args.without_rna_resnet,
        without_chem_env=args.without_chem_env
    )

    # 4. 开始训练
    print(f"开始训练with {args.loss_type} loss...")
    logger.info(">>> 开始训练...")

    trained_model = train_mod.train(
        model,
        train_ds,
        val_ds,  # 传入验证集用于早停
        args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        collate_fn=collate_hybrid,
        loss_type=args.loss_type,
        pos_weight=args.pos_weight,
        patience=args.patience,
        logger=logger,  # 传入 logger
        log_dir=args.log_dir,  # 传入路径用于绘图
        focal_alpha=args.focal_alpha,  # <--- 新增传参
        focal_gamma=args.focal_gamma,  # <--- 新增传参
    )

    # 7. 最终测试 (Test Phase) - 新增部分
    if len(test_ds) > 0:
        logger.info(">>> 开始最终的测试 (Hold-out Test Set)...")
        print(">>> 开始最终的测试 (Hold-out Test Set)")


        if args.threshold is not None:
            final_threshold = args.threshold
        else:
            # 如果保留自动搜索逻辑，放在这里
            # val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_hybrid)
            # final_threshold = train_mod.find_optimal_threshold(trained_model, val_loader, args.device)
            # logger.info(f"Using Optimal Threshold (Auto): {final_threshold:.2f}")
            # 不自动搜索，直接保持 0.5
            logger.info(f"Using Default Threshold: {final_threshold:.2f}")

        logger.info(f"Using Manual Threshold: {final_threshold:.2f}")
        print(f"Using Manual Threshold: {final_threshold:.2f}")
        # 构建测试集 Loader
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_hybrid)


        """ 绘图代码 """
        # [修改] 我们需要从 evaluate 获取原始数据来绘图
        # 为了不破坏原有 evaluate 函数的返回值结构，我们这里手动再跑一次推理收集数据
        # 或者：如果你不想改 evaluate，可以简单的把 evaluate 的逻辑复制出来一部分专门用于收集数据

        trained_model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                if batch is None: continue
                # 数据移动到设备
                rna_seq = batch['rna_seq'].to(args.device)
                mask_rna = batch['mask_rna'].to(args.device)
                ligand_graph = batch['ligand_graph'].to(args.device)
                labels = batch['labels'].to(args.device)

                logits = trained_model(rna_seq, ligand_graph, mask_rna)
                probs = torch.sigmoid(logits)

                # 只收集 mask=True 的部分
                mask_bool = mask_rna.bool()
                if mask_bool.any():
                    all_probs.append(probs[mask_bool].cpu().numpy())
                    all_labels.append(labels[mask_bool].cpu().numpy())

        # 拼接数据
        if all_probs:
            y_prob_flat = np.concatenate(all_probs)
            y_true_flat = np.concatenate(all_labels)

            # --- 调用可视化函数 ---
            logger.info("生成评估图表...")
            plot_all_evaluation_charts(
                y_true_flat,
                y_prob_flat,
                save_dir=args.log_dir,
                model_name=exp_name,  # 改用 exp_name
                threshold=final_threshold
            )
            # --------------------
        # 继续调用 evaluate 打印数值指标
        test_metrics, test_loss = train_mod.evaluate(
            trained_model,
            test_loader,
            args.device,
            loss_type=args.loss_type,
            pos_weight=args.pos_weight,
            threshold=final_threshold,
            sample_level=True,
            iou_threshold=args.iou_threshold,
            eval_graph=args.eval_graph,
            focal_alpha=args.focal_alpha,  # <--- 新增传参
            focal_gamma=args.focal_gamma
        )

        # ==========================================
        # 🎯 SOTA 自动对抗系统 (SOTA Radar)
        # ==========================================
        # 1. 自动识别数据集
        dataset_name = "Unknown"
        if "RB9" in args.manifest:
            dataset_name = "RB9"
        elif "TE18" in args.manifest:
            dataset_name = "TE18"

        # 2. 预设 SOTA 基线
        SOTA = {
            "RB9": {"prec": 0.77, "rec": 0.61, "mcc": 0.57, "f1": 0.68, "auc": 0.85},
            "TE18": {"prec": 0.68, "rec": 0.37, "mcc": 0.34, "f1": 0.49, "auc": 0.78}
        }

        my_f1 = test_metrics.get('f1', 0)
        my_mcc = test_metrics.get('mcc', 0)
        my_rec = test_metrics.get('rec', 0)
        my_prec = test_metrics.get('prec', 0)

        # 3. 计算与打印差距
        msg = f"\n{'=' * 45}\n🚀 FINAL TEST RESULTS on [{dataset_name}]\n{'=' * 45}\n"
        if dataset_name in SOTA:
            sota = SOTA[dataset_name]
            f1_diff = my_f1 - sota['f1']
            mcc_diff = my_mcc - sota['mcc']
            rec_diff = my_rec - sota['rec']
            prec_diff = my_prec - sota['prec']

            f1_flag = "🟢 WIN!" if f1_diff > 0 else "🔴 LOSE"
            mcc_flag = "🟢 WIN!" if mcc_diff > 0 else "🔴 LOSE"
            rec_flag = "🟢 WIN!" if rec_diff > 0 else "🔴 LOSE"
            prec_flag = "🟢 WIN!" if prec_diff > 0 else "🔴 LOSE"

            msg += (
                f"🎯 SOTA Target   (Prec): {sota['prec']:.4f} | (Recall): {sota['rec']:.4f} | (F1): {sota['f1']:.4f} | (MCC): {sota['mcc']:.4f}\n"
                f"🏆 Your Score    (Prec): {my_prec:.4f} ({prec_diff:+.4f}) {prec_flag}\n"
                f"🏆 Your Score  (Recall): {my_rec:.4f} ({rec_diff:+.4f}) {rec_flag}\n"
                f"🏆 Your Score      (F1): {my_f1:.4f} ({f1_diff:+.4f}) {f1_flag}\n"
                f"🏆 Your Score     (MCC): {my_mcc:.4f} ({mcc_diff:+.4f}) {mcc_flag}\n"
                f"{'-' * 50}\n")

        msg += (f"[GLOBAL PIXEL-LEVEL DETAILS]:\n"
                f"AUC:       {test_metrics.get('auc', 0):.4f}\n"
                f"AUPRC:     {test_metrics.get('auprc', 0):.4f}\n"
                f"Precision: {test_metrics.get('prec', 0):.4f}\n"
                f"Recall:    {test_metrics.get('rec', 0):.4f}\n"
                f"Mean IoU:  {test_metrics.get('sample_mean_iou', 0):.4f}\n")
        logger.info(msg)

        # ==========================================
        # 终极保存：模型 + JSON + 总表 CSV
        # ==========================================
        import pandas as pd
        import json

        logger.info("💾 正在自动归档实验结果...")

        save_path = os.path.join(exp_dir, "best_model.pth")
        torch.save(trained_model.state_dict(), save_path)

        # 在保存 config 之前，把动态推断出的特征维度塞进 args 里
        args.d_rna = d_rna
        args.d_lig_node = d_lig_node
        args.d_lig_edge = d_lig_edge

        config_path = os.path.join(exp_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=4)

        metrics_path = os.path.join(exp_dir, "metrics.json")
        clean_metrics = {k: float(v) for k, v in test_metrics.items()}
        with open(metrics_path, 'w') as f:
            json.dump(clean_metrics, f, indent=4)

        csv_summary_path = os.path.join("ab_experiments", f"results_summary_{dataset_name}_{args.loss_type}.csv")

        # [修改] 增加 Dataset 列和 SOTA_Diff 列
        f1_gap = (my_f1 - SOTA[dataset_name]['f1']) if dataset_name in SOTA else 0.0
        row_data = {
            "Dataset": dataset_name,  # <--- 知道这是哪个数据集跑出来的
            "F1_Gap": float(f1_gap),  # <--- 正数说明打败了SOTA，负数说明没打过
            "Experiment": exp_name,
            "F1": float(my_f1),
            "MCC": float(my_mcc),
            "Prec": float(test_metrics.get('prec', 0)),
            "Recall": float(test_metrics.get('rec', 0)),
            "AUC": float(test_metrics.get('auc', 0)),
            "Dist(A)": args.label_dist_threshold,
            "Loss": args.loss_type,
            "Pos_Wt": args.pos_weight,
            "Thresh": final_threshold,
            "LR": args.lr
        }

        df_new = pd.DataFrame([row_data])
        if not os.path.exists(csv_summary_path):
            df_new.to_csv(csv_summary_path, index=False, encoding='utf-8-sig')
        else:
            df_new.to_csv(csv_summary_path, mode='a', header=False, index=False, encoding='utf-8-sig')

        logger.info(f"✅ 结果已追加至: {csv_summary_path}")
        logger.info("=" * 45 + "\n")
    else:
        logger.warning("Test dataset is empty. Skipping testing phase.")


if __name__ == '__main__':
    main()