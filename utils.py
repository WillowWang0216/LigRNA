"""
[修改] 增加了 compute_graph_level_metrics 和
compute_sample_level_metrics。

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, precision_score, recall_score, accuracy_score, \
    average_precision_score

Alpha = 0.75
# --- 新增：样本级指标计算逻辑 ---
def compute_sample_level_metrics(logits, labels, mask, threshold=0.5, iou_threshold=0.5):
    batch_size = logits.shape[0]

    # 初始化存储列表
    metrics_lists = {
        'iou': [], 'f1': [], 'mcc': [],
        'auc': [], 'auprc': [],
        'acc': [], 'prec': [], 'rec': []
    }

    passed_samples = 0

    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labs = labels.detach().cpu().numpy()
    masks = mask.detach().cpu().numpy().astype(bool)

    for i in range(batch_size):
        valid_mask = masks[i]
        if not np.any(valid_mask): continue

        # 获取单个 RNA 的有效数据
        p = probs[i][valid_mask]
        l = labs[i][valid_mask]
        pred = (p > threshold).astype(int)

        # --- 1. 基础指标 (IoU) ---
        intersection = np.logical_and(pred == 1, l == 1).sum()
        union = np.logical_or(pred == 1, l == 1).sum()
        iou = 1.0 if (union == 0 and intersection == 0) else (intersection / union if union > 0 else 0.0)

        # --- 2. 分类指标 ---
        # 注意处理只有单一类别的情况 (例如该 RNA 没有结合位点，或全是结合位点)
        if len(np.unique(l)) > 1:
            auc = roc_auc_score(l, p)
            auprc = average_precision_score(l, p)
            mcc = matthews_corrcoef(l, pred)
        else:
            # 无法计算 AUC/AUPRC 时赋默认值
            auc = 0.5
            auprc = 0.0
            mcc = 0.0

        f1 = f1_score(l, pred, zero_division=0)
        acc = accuracy_score(l, pred)
        prec = precision_score(l, pred, zero_division=0)
        rec = recall_score(l, pred, zero_division=0)

        # 存入列表
        metrics_lists['iou'].append(iou)
        metrics_lists['f1'].append(f1)
        metrics_lists['mcc'].append(mcc)
        metrics_lists['auc'].append(auc)
        metrics_lists['auprc'].append(auprc)
        metrics_lists['acc'].append(acc)
        metrics_lists['prec'].append(prec)
        metrics_lists['rec'].append(rec)

        if iou > iou_threshold:
            passed_samples += 1

    # 返回总和，方便在 train.py 中累加
    return {
        'sum_iou': np.sum(metrics_lists['iou']),
        'sum_f1': np.sum(metrics_lists['f1']),
        'sum_mcc': np.sum(metrics_lists['mcc']),
        'sum_auc': np.sum(metrics_lists['auc']),
        'sum_auprc': np.sum(metrics_lists['auprc']),
        'sum_acc': np.sum(metrics_lists['acc']),
        'sum_prec': np.sum(metrics_lists['prec']),
        'sum_rec': np.sum(metrics_lists['rec']),
        'passed_count': passed_samples,
        'total_count': len(metrics_lists['iou'])
    }


# class FocalLoss(nn.Module):
#     """
#     Focal Loss 用于解决类别不平衡和难易样本挖掘。
#     Formula: Loss = -alpha * (1-pt)^gamma * log(pt)
#     """
#     #把FocalLoss 的 alpha 改为 0.75 或 0.8。如果不改这个，深层 CNN 仍然会倾向于预测全 0 来“偷懒”降低 Loss。
#     def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, inputs, targets, mask=None):
#         # inputs: logits (B, L)
#         # targets: labels (B, L)
#
#         # BCE With Logits 计算 log(pt)
#         bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#
#         # pt 是预测正确的概率
#         pt = torch.exp(-bce_loss)
#
#         # Focal term
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
#
#         if mask is not None:
#             focal_loss = focal_loss * mask
#             if self.reduction == 'mean':
#                 return focal_loss.sum() / (mask.sum() + 1e-8)
#             elif self.reduction == 'sum':
#                 return focal_loss.sum()
#         else:
#             if self.reduction == 'mean':
#                 return focal_loss.mean()
#             elif self.reduction == 'sum':
#                 return focal_loss.sum()
#
#         return focal_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        # inputs: logits (B, L)
        # targets: labels (B, L)

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)

        # 🎯 [核心修复]：动态分配正负样本的权重
        # 如果 target=1 (正样本)，权重为 alpha
        # 如果 target=0 (负样本)，权重为 1 - alpha
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Focal term
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if mask is not None:
            focal_loss = focal_loss * mask
            if self.reduction == 'mean':
                return focal_loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return focal_loss.sum()
        else:
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()

        return focal_loss


def masked_loss(logits, labels, mask, loss_type='bce', pos_weight=None,focal_alpha=0.75, focal_gamma=2.0):
    """
    统一的 Loss 接口，支持 BCE 和 Focal Loss
    """
    # 确保 mask 存在
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)

    labels = labels.float()

    # 空数据检查
    logits_flat = logits[mask]
    if logits_flat.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    if loss_type == 'focal':
        # 推荐 alpha=0.5 (平衡) ||alpha=0.68(与实际数据比例相符), gamma=2.0
        loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
        return loss_fn(logits, labels, mask)

    else:  # 'bce'
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], device=logits.device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')
        else:
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        loss = loss_fn(logits, labels)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss


# 修改：[新增] 图级别预测逻辑
def compute_graph_level_metrics(logits, labels, mask, graph_labels, threshold=0.5):
    """
    计算基于 Max Pooling 的图级别（Per-RNA）分类指标。

    参数:
        logits: (B, L)
        graph_labels: (B,) 每个样本的真实图标签 (0或1)
    """
    probs = torch.sigmoid(logits)  # (B, L)
    batch_size = probs.shape[0]

    pred_graph_labels = []
    true_graph_labels = []

    # 将 Tensor 转为 numpy 方便处理
    probs_np = probs.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy().astype(bool)
    graph_y_np = graph_labels.detach().cpu().numpy()

    for i in range(batch_size):
        # 获取该样本有效区域的概率
        valid_probs = probs_np[i][mask_np[i]]
        if len(valid_probs) == 0: continue

        # Max Pooling 策略: 只要有一个点 > threshold，就认为结合
        is_binding = np.any(valid_probs > threshold)
        pred_label = 1 if is_binding else 0

        pred_graph_labels.append(pred_label)
        true_graph_labels.append(int(graph_y_np[i]))

    return pred_graph_labels, true_graph_labels

def masked_metrics(logits, labels, mask, threshold=0.5):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labs = labels.detach().cpu().numpy()
    if mask is None:
        m = np.ones_like(labs, dtype=bool)
    else:
        m = mask.detach().cpu().numpy().astype(bool)

    probs = probs[m]
    labs = labs[m]

    if len(labs) == 0:
        return {'auc': 0.5, 'prec': 0.0, 'rec': 0.0, 'acc': 0.0, 'mcc': 0.0}

    # 处理只有一个类别导致 AUC 报错的情况
    if len(np.unique(labs)) > 1:
        auc = roc_auc_score(labs, probs)
        auprc = average_precision_score(labs, probs)
        mcc = matthews_corrcoef(labs, (probs > threshold).astype(int))
    else:
        auc = 0.5
        mcc = 0.0

    preds = (probs > threshold).astype(int)
    return {
        'auc': float(auc),
        'auprc': float(auprc),  # Area Under Precision-Recall Curve
        'prec': float(precision_score(labs, preds, zero_division=0)),
        'rec': float(recall_score(labs, preds, zero_division=0)),
        'acc': float(accuracy_score(labs, preds)),
        'mcc': float(mcc),
        'f1': float(f1_score(labs, preds, zero_division=0)),  # F1 Score
    }