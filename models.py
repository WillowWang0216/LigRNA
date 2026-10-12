"""
1. LigandGNN: 支持无边特征输入，add_self_loops=True 支持离子。
2. 架构: 3层 GATv2 + 2层 Cross-Attention。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv,global_mean_pool
from torch_geometric.utils import to_dense_batch


class LigandGNN(nn.Module):
    """
    使用 GATv2 提取小分子图特征，显式利用 Edge Attributes (键类型、共轭等)。
    深度: 3层
    """

    def __init__(self, in_dim, edge_dim, hidden_dim, num_layers=3, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 修改容错：无边特征时 edge_dim 为 None
        if edge_dim == 0: edge_dim = None
        # 强制 add_self_loops=True 以支持单节点图
        self.convs.append(GATv2Conv(in_dim, hidden_dim, edge_dim=edge_dim, heads=heads, concat=False, add_self_loops=True))
        self.bns.append(nn.LayerNorm(hidden_dim))

        """
        # 第一层
        self.convs.append(GATv2Conv(in_dim, hidden_dim, edge_dim=edge_dim, heads=heads, concat=False))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        """

        # 后续层
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, heads=heads, concat=False))
            #修改 BatchNorm1d LayerNorm
            self.bns.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        # x: (Total_Atoms, in_dim)
        # edge_attr: (Total_Edges, edge_dim)

        # 残差连接需要维度匹配，第一层如果维度变了就不加残差，或者加投影。
        # 这里为了简化，直接在层间传递
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_in = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.elu(x)
            # 如果不是第一层（维度已对齐），添加残差连接
            if i > 0:
                x = x + x_in
        return x


# 2. 定义 CrossAttentionBlock (保持在同一个文件中)
class CrossAttentionBlock(nn.Module):
    """
    Transformer Decoder 风格的 Cross Attention。
    Query: RNA
    Key/Value: Ligand
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        attn_out, _ = self.mha(query, key, value, key_padding_mask=key_padding_mask)
        query = self.norm1(query + self.dropout(attn_out))

        ffn_out = self.ffn(query)
        query = self.norm2(query + self.dropout(ffn_out))
        return query


# 3. 新增: 1D ResNet Block (修正了这里的拼写错误)
class ResBlock1D(nn.Module):
    """
    1D Residual Block: 用于加深 RNA 序列特征提取
    结构: Conv1d -> BN -> ReLU -> Conv1d -> BN -> Add(Residual) -> ReLU
    """

    #kernel_size 对应感受野
    def __init__(self, channels, kernel_size=5, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #out += residual
        # --- 修复开始 ---
        # 使用 out = out + residual 代替 out += residual
        # 这会创建一个新的张量，避免覆盖 Hook 追踪的内存
        out = out + residual
        # --- 修复结束 ---
        return F.relu(out)


class RNALigandGNNModel(nn.Module):
    def __init__(self, d_rna, d_lig_node, d_lig_edge,
                 hidden_dim=128, dropout=0.2,
                 rna_depth=5, interact_depth=3,
                 fusion_type='attention',
                 use_rna_resnet=True,
                 without_chem_env=False): 
        # CNN 深度 (rna_depth): 建议设置为 3 到 5
        # 交互层深度 (interact_depth): 建议增加到 3 或 4（原代码是 2）。这能让 RNA 和配体进行更充分的信息交换。
        # Kernel Size: 在 ResBlock1D 中，我建议使用 kernel_size=5 或 7。结合位点通常涉及 3-5 个核苷酸，较大的卷积核能一次性看到整个结合口袋的局部序列。
        super().__init__()
        # [新增优化] 预注册化学属性索引，让它待在 GPU 上
        chem_idx = torch.tensor([3, 7, 9, 10, 11, 12, 13], dtype=torch.long)
        self.register_buffer('chem_indices', chem_idx)
        edge_abl_idx = torch.tensor([0, 1, 2], dtype=torch.long)
        self.register_buffer('edge_abl_indices', edge_abl_idx)

        self.fusion_type = fusion_type
        self.use_rna_resnet = use_rna_resnet
        self.without_chem_env = without_chem_env
        
        # --- 修改 1: RNA Encoder 从 MLP 变为 CNN ---
        # 1.1 维度投影 (先把输入维度 d_rna 变成 hidden_dim)
        # 🌟 注意：无论开不开 ResNet，这个投影层都必须保留，用来对齐维度！
        self.rna_input_proj = nn.Conv1d(d_rna, hidden_dim, kernel_size=1)

        # 1.2 堆叠 ResBlock 加深网络 (rna_depth 层)
        # 🌟 2. 只有当开关打开时，才初始化深层 ResNet
        if self.use_rna_resnet:
            self.rna_encoder = nn.Sequential(*[
                ResBlock1D(hidden_dim, kernel_size=5, dropout=dropout)  # kernel=5 能看更宽的视野
                for _ in range(rna_depth)
            ])

        # --- 修改 2: Ligand GNN (保持原样或微调) ---
        self.lig_gnn = LigandGNN(d_lig_node, d_lig_edge, hidden_dim, num_layers=3)

        # --- 3 & 4. 交互层与预测头 (根据 fusion_type 动态选择) ---
        if self.fusion_type == 'attention':
            self.cross_layers = nn.ModuleList([
                CrossAttentionBlock(hidden_dim, num_heads=4, dropout=dropout)
                for _ in range(interact_depth)
            ])
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        elif self.fusion_type == 'concat':
            # 如果是 Concat，RNA(H) 和 Ligand(H) 拼在一起变成 2H
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        


    def forward(self, rna_seq, batch_ligand_graph, mask_rna=None):
        # --- A. RNA Process (CNN) ---
        h_rna = rna_seq.permute(0, 2, 1)  # (B, d_rna, L)
        h_rna = self.rna_input_proj(h_rna)  
        
        # 🌟 RNA ResNet 消融逻辑
        if getattr(self, 'use_rna_resnet', True):
            h_rna = self.rna_encoder(h_rna)

        h_rna = h_rna.permute(0, 2, 1)  # 转回 (B, L, H)

        # --- B. Ligand Process (GNN) ---
        x, edge_index, edge_attr, batch = batch_ligand_graph.x.float(), batch_ligand_graph.edge_index, batch_ligand_graph.edge_attr, batch_ligand_graph.batch
        
        # 确保 edge_attr 是 float 类型（如果存在的话）
        if edge_attr is not None:
            edge_attr = edge_attr.float()

        # # 🌟 化学理化属性消融逻辑 (精准打击)
        # if getattr(self, 'without_chem_env', False):
        #     # 1. 剥离 Atom Features
        #     # 3(电荷), 7(芳香), 9(氢供), 10(氢受), 11(碱), 12(酸), 13(卤素)
        #     chem_indices = [3, 7, 9, 10, 11, 12, 13] 
        #     x_ablated = x.clone()
        #     x_ablated[:, chem_indices] = 0.0
            
        #     # 2. 剥离 Edge Features (加入鲁棒性判断，防止单原子离子报错)
        #     if edge_attr is not None and edge_attr.numel() > 0:
        #         edge_attr_ablated = edge_attr.clone()
        #         # 0:bond_type, 1:bond_stereo, 2:is_conjugated 置0
        #         edge_attr_ablated[:, [0, 1, 2]] = 0.0 
        #         # 3:atom_pair 设为未知(14)
        #         edge_attr_ablated[:, 3] = 14.0 
        # 🌟 化学理化属性消融逻辑 (精准打击)
        if getattr(self, 'without_chem_env', False):
            # 1. 剥离 Atom Features
            x_ablated = x.clone()
            # 修改：使用 self.chem_indices 代替临时列表
            x_ablated[:, self.chem_indices] = 0.0
            
            # 2. 剥离 Edge Features
            if edge_attr is not None and edge_attr.numel() > 0:
                edge_attr_ablated = edge_attr.clone()
                # 修改：使用 self.edge_abl_indices
                edge_attr_ablated[:, self.edge_abl_indices] = 0.0 
                edge_attr_ablated[:, 3] = 14.0
            else:
                edge_attr_ablated = edge_attr # 离子没有键，直接跳过
                
            h_lig_flat = self.lig_gnn(x_ablated, edge_index, edge_attr_ablated)
            
        else:
            # 正常训练路线
            h_lig_flat = self.lig_gnn(x, edge_index, edge_attr)

        # --- C. Interaction & D. Predict ---
        if self.fusion_type == 'attention':
            h_lig_dense, mask_lig = to_dense_batch(h_lig_flat, batch)
            key_padding_mask = ~mask_lig
            
            for layer in self.cross_layers:
                h_rna = layer(query=h_rna, key=h_lig_dense, value=h_lig_dense, key_padding_mask=key_padding_mask)
            logits = self.predictor(h_rna).squeeze(-1)
            
        elif self.fusion_type == 'concat':
            h_lig_global = global_mean_pool(h_lig_flat, batch)
            L = h_rna.shape[1]
            h_lig_broadcast = h_lig_global.unsqueeze(1).expand(-1, L, -1)
            h_fused = torch.cat([h_rna, h_lig_broadcast], dim=-1)
            logits = self.predictor(h_fused).squeeze(-1)

        return logits
