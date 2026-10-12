
"""
[修改] evaluate 函数集成了 Graph-Level 和 Sample-Level 的计算逻辑。
"""
import warnings

import torch
import numpy as np
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
import os
import logging
import random
#from mlp_model2_auto.utils import compute_sample_level_metrics  # 确保包名正确
from utils import masked_loss, masked_metrics, compute_sample_level_metrics, compute_graph_level_metrics

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

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.best_state_dict = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''验证集损失减少时保存模型'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_state_dict = copy.deepcopy(model.state_dict())
        # torch.save(model.state_dict(), self.path) # 可选：直接写入磁盘
        self.val_loss_min = val_loss


def plot_training_curves(history, save_dir):
    """绘制 Loss, AUC, AUPRC, F1 等曲线"""
    epochs = range(1, len(history['train_loss']) + 1)

    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # 2. Metrics Curve (AUC & AUPRC)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_auc'], label='Val AUC')
    plt.plot(epochs, history['val_auprc'], label='Val AUPRC')
    plt.title('Validation AUC & AUPRC')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'metrics_auc_auprc.png'))
    plt.close()

    # 3. F1 & MCC
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_f1'], label='Val F1')
    plt.plot(epochs, history['val_mcc'], label='Val MCC')
    plt.title('Validation F1 & MCC')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'metrics_f1_mcc.png'))
    plt.close()

def train(model, train_dataset, val_dataset, device, epochs=50, batch_size=8, lr=1e-4, 
          pos_weight=None, collate_fn=None, loss_type='focal', patience=10,
          logger=None, log_dir='log',focal_alpha=0.75, focal_gamma=2.0):

    # 使用 DataLoader
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, #shuffle=True, collate_fn=collate_fn , drop_last=True)# <--- 关键！丢弃最后一个凑不齐的 Batch

    # 使用 4 到 8 个 CPU 进程并行加载数据，并开启锁页内存加速 GPU 传输
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, collate_fn=collate_fn, 
                            drop_last=True, 
                            num_workers=8, # 增加并行度
                            pin_memory=True, 
                            persistent_workers=True,# [新增] 保持 worker 进程不销毁，加快 Epoch 切换
                            prefetch_factor=2)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True) if val_dataset else None
    # 验证集 loader (如果有的话)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if val_dataset else None

    #修改 加入权重衰减
    optimizer = torch.optim.Adam(model.parameters(), lr=lr , weight_decay=1e-4)
    # [新增] 初始化梯度缩放器，防止半精度下梯度消失
    scaler = torch.cuda.amp.GradScaler()
    
    # 初始化早停
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    model.to(device)

    # 记录历史数据
    history = {
        'train_loss': [], 'val_loss': [],
        'val_auc': [], 'val_auprc': [], 'val_f1': [], 'val_mcc': [],
        'val_acc': [], 'val_prec': [], 'val_rec': []
    }

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        total_loss = 0.0
        total_examples = 0
        
        for batch in train_loader:
            if batch is None: continue
        # 原来的# 获取数据
            # rna_seq = batch['rna_seq'].to(device)
            # mask_rna = batch['mask_rna'].to(device)
            # labels = batch['labels'].to(device)
            # # Ligand 是 PyG Batch 对象，自动包含 .batch 属性
            # ligand_graph = batch['ligand_graph'].to(device)
            # # 前向传播
            # logits = model(rna_seq, ligand_graph, mask_rna)
            
            # # 计算损失
            # loss = masked_loss(logits, labels, mask_rna, loss_type=loss_type, pos_weight=pos_weight,focal_alpha=focal_alpha, focal_gamma=focal_gamma)
            
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

        # [优化] 使用 non_blocking=True 加快数据拷贝
            rna_seq = batch['rna_seq'].to(device, non_blocking=True)
            mask_rna = batch['mask_rna'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            ligand_graph = batch['ligand_graph'].to(device, non_blocking=True)

            # [核心加速] 开启自动混合精度上下文
            optimizer.zero_grad()
            # 2. 混合精度前向传播 [核心加速]
            with torch.cuda.amp.autocast():
                logits = model(rna_seq, ligand_graph, mask_rna)
                loss = masked_loss(logits, labels, mask_rna, loss_type=loss_type, 
                                   pos_weight=pos_weight, focal_alpha=focal_alpha, 
                                   focal_gamma=focal_gamma)
            
            # 3. 混合精度反向传播 [核心加速]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            
            
            # 记录 (加权平均)
            num_valid = mask_rna.sum().item()
            total_loss += loss.item() * num_valid
            total_examples += num_valid
            
        avg_train_loss = total_loss / max(1, total_examples)
        
        # --- Validation ---
        val_metrics = {}
        avg_val_loss = avg_train_loss # 如果没有验证集，使用训练loss占位
        
        if val_loader:
            val_metrics, avg_val_loss = evaluate(model, val_loader, device, loss_type, pos_weight)
            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.5f} | Val Loss={avg_val_loss:.5f} | Val AUC={val_metrics.get('auc',0):.4f} MCC={val_metrics.get('mcc',0):.4f}")
        else:
            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.5f}")

        # 记录 Epoch 日志
        log_msg = (f"Epoch {epoch:03d}: Train Loss={avg_train_loss:.5f} | Val Loss={avg_val_loss:.5f} | "
                       f"AUC={val_metrics.get('auc', 0):.4f} | AUPRC={val_metrics.get('auprc', 0):.4f} | "
                       f"F1={val_metrics.get('f1', 0):.4f} | MCC={val_metrics.get('mcc', 0):.4f} | "
                       f"Acc={val_metrics.get('acc', 0):.4f} | Prec={val_metrics.get('prec', 0):.4f} | Rec={val_metrics.get('rec', 0):.4f}")

        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)

            # 更新历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_metrics.get('auc', 0))
        history['val_auprc'].append(val_metrics.get('auprc', 0))
        history['val_f1'].append(val_metrics.get('f1', 0))
        history['val_mcc'].append(val_metrics.get('mcc', 0))

        # --- Early Stopping Check ---
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
            
    # 恢复最佳模型
    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)
        print("Loaded best model from early stopping.")

    # 训练结束后绘图
    if log_dir:
        try:
            plot_training_curves(history, log_dir)
            if logger: logger.info(f"Training curves saved to {log_dir}")
        except Exception as e:
            if logger: logger.error(f"Failed to plot curves: {e}")

    return model


def evaluate(model, loader, device, loss_type='bce', pos_weight=None, threshold=0.5,
             sample_level=True, iou_threshold = 0.4, eval_graph=True,focal_alpha=0.75, focal_gamma=2.0):
    model.eval()
    total_loss = 0.0
    total_count = 0
    logits_list = []
    labels_list = []

    # 初始化全套累积器
    sample_agg = {
        'sum_iou': 0.0, 'sum_f1': 0.0, 'sum_mcc': 0.0,
        'sum_auc': 0.0, 'sum_auprc': 0.0,
        'sum_acc': 0.0, 'sum_prec': 0.0, 'sum_rec': 0.0,
        'passed': 0, 'total': 0
    }
    #修改：Graph Level 累积器
    graph_preds_all , graph_trues_all  = [], []

    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            rna_seq = batch['rna_seq'].to(device)
            mask_rna = batch['mask_rna'].to(device)
            labels = batch['labels'].to(device)
            ligand_graph = batch['ligand_graph'].to(device)

            logits = model(rna_seq, ligand_graph, mask_rna)

            loss = masked_loss(logits, labels, mask_rna, loss_type=loss_type, pos_weight=pos_weight,focal_alpha=focal_alpha, focal_gamma=focal_gamma)

            num = mask_rna.sum().item()
            total_loss += loss.item() * num
            total_count += num

            mask_bool = mask_rna.bool()
            if mask_bool.any():
                logits_list.append(logits[mask_bool].cpu())
                labels_list.append(labels[mask_bool].cpu())

            # --- 样本级计算 ---
            if sample_level:
                res = compute_sample_level_metrics(logits, labels, mask_rna, threshold, iou_threshold)

                # 累加所有指标
                for key in ['sum_iou', 'sum_f1', 'sum_mcc', 'sum_auc', 'sum_auprc', 'sum_acc', 'sum_prec', 'sum_rec']:
                    sample_agg[key] += res[key]

                sample_agg['passed'] += res['passed_count']
                sample_agg['total'] += res['total_count']

            #修改
            if eval_graph:
                # 获取 Graph Label (注意: 这里的 batch 是字典，不是对象，如果你用了 collate_hybrid)
                # 在 train.py 的 evaluate 函数里，batch 通常是一个 dict
                if 'graph_y' in batch:
                    graph_y = batch['graph_y'].to(device)
                else:
                    # 默认全为 1
                    graph_y = torch.ones(rna_seq.shape[0], device=device)
                preds, trues = compute_graph_level_metrics(logits, labels, mask_rna, graph_y, threshold)
                graph_preds_all.extend(preds)
                graph_trues_all.extend(trues)

    avg_loss = total_loss / max(1, total_count)

    # Global Metrics
    if logits_list:
        logits_flat = torch.cat(logits_list)
        labels_flat = torch.cat(labels_list)
        metrics = masked_metrics(logits_flat, labels_flat, mask=None, threshold=threshold)
    else:
        metrics = {'auc': 0.5, 'auprc': 0.0, 'prec': 0.0, 'rec': 0.0, 'acc': 0.0, 'f1': 0.0, 'mcc': 0.0}

    # Sample Level Metrics (计算平均值)
    # Sample Level Metrics (计算平均值)
    if sample_level and sample_agg['total'] > 0:
        N = sample_agg['total']
        metrics['sample_mean_iou'] = sample_agg['sum_iou'] / N
        metrics['sample_mean_f1'] = sample_agg['sum_f1'] / N
        metrics['sample_mean_mcc'] = sample_agg['sum_mcc'] / N
        metrics['sample_mean_auc'] = sample_agg['sum_auc'] / N
        metrics['sample_mean_auprc'] = sample_agg['sum_auprc'] / N
        metrics['sample_mean_acc'] = sample_agg['sum_acc'] / N
        metrics['sample_mean_prec'] = sample_agg['sum_prec'] / N
        metrics['sample_mean_rec'] = sample_agg['sum_rec'] / N

        metrics['sample_pass_rate'] = sample_agg['passed'] / N

    #return metrics, avg_loss

        # 计算 Graph Level Metrics
        if eval_graph and len(graph_trues_all) > 0:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

            g_true = np.array(graph_trues_all)
            g_pred = np.array(graph_preds_all)

            metrics['graph_acc'] = accuracy_score(g_true, g_pred)
            # zero_division=0 防止分母为0报错
            metrics['graph_prec'] = precision_score(g_true, g_pred, zero_division=0)
            metrics['graph_rec'] = recall_score(g_true, g_pred, zero_division=0)
            metrics['graph_f1'] = f1_score(g_true, g_pred, zero_division=0)

            # [新增] 计算 Graph MCC
            # 如果只有一种类别（比如全是正样本），MCC 也是无法定义的，通常返回 0.0
            if len(np.unique(g_true)) > 1:
                metrics['graph_mcc'] = matthews_corrcoef(g_true, g_pred)
            else:
                metrics['graph_mcc'] = 0.0

            metrics['graph_pos_pred'] = np.sum(g_pred)

            # 如果全是正样本，Specificity 没法算，这里简单处理
            tn, fp, fn, tp = confusion_matrix(g_true, g_pred, labels=[0, 1]).ravel() if len(
                np.unique(g_true)) > 1 else (0, 0, 0, len(g_true))
            metrics['graph_tp'] = int(tp)
            metrics['graph_fn'] = int(fn)

    return metrics, avg_loss