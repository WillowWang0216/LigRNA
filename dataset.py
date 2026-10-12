# 文件: dataset.py 

import os
import torch
import pymol2
from torch_geometric.data import Dataset, Data
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB import DSSP
        
# Toggle for using PyMOL exact SASA calculation when available
INCLUDE_SASA = True

# Nucleotide name set for PyMOL residue filtering
NUCLEOTIDES = set(['A', 'G', 'C', 'U', 'ADE', 'GUA', 'CYT', 'URA', 'PSU', 'I'])

Rna_features_dir = '../RNAfeatures_icon'
Ligand_graphs_dir = '../ligand_graph_ions'
#Pdb_dir = '../all_pdb_icon'  #已经从命令行传入了


def calculate_all_atom_sasa_with_pymol(pdb_file_path: str) -> dict:
    """Use PyMOL (via pymol2) to compute per-atom SASA and return mapping keyed by (chain, resi, atom_name).

    Returns: dict[(chain, resi, atom_name)] = sasa_value
    """
    atom_sasa_map = {}
    if pymol2 is None:
        return {}
    try:
        with pymol2.PyMOL() as pymol:
            cmd = pymol.cmd
            pdb_name = "s"
            cmd.load(pdb_file_path, pdb_name)
            # ensure element names normalized
            cmd.alter(f'({pdb_name})', "elem=name[0:1].upper()")
            cmd.set('dot_solvent', 1)
            cmd.set('dot_density', 3)
            # compute area into b-factor
            cmd.get_area(f'({pdb_name})', load_b=1)
            atom_data = []
            cmd.iterate(f'({pdb_name})', 'atom_data.append((chain, resi, resn, name, b))', space={'atom_data': atom_data})
        for chain, resi, resn, name, sasa in atom_data:
            try:
                if isinstance(resn, bytes):
                    resn = resn.decode()
                if isinstance(resi, str) and resi.isdigit():
                    resi_int = int(resi)
                else:
                    try:
                        resi_int = int(float(resi))
                    except Exception:
                        resi_int = None
                if resn.strip() in NUCLEOTIDES and resi_int is not None:
                    key = (str(chain).strip(), int(resi_int), str(name).strip())
                    atom_sasa_map[key] = float(sasa)
            except Exception:
                continue
        return atom_sasa_map
    except Exception as e:
        print(f" 警告: PyMOL 计算失败: {e}")
        return {}


def aggregate_sasa_parts(all_atom_sasa: dict) -> dict:
    """Aggregate per-atom SASA into per-residue parts: base, sugar, phosphate.

    Returns: dict[(chain, res_seq)] = {'sasa_base': val, 'sasa_sugar': val, 'sasa_phosphate': val}
    """
    base_atoms = {"N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9", "O2", "N4", "O4", "N2", "O6"}
    sugar_atoms = {"C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'", "O4'", "O5'"}
    phosphate_atoms = {"P", "OP1", "OP2"}
    residue_sasa_parts = {}
    for (chain_id, res_seq, atom_name), sasa_val in all_atom_sasa.items():
        res_key = (chain_id, res_seq)
        if res_key not in residue_sasa_parts:
            residue_sasa_parts[res_key] = {'sasa_base': 0.0, 'sasa_sugar': 0.0, 'sasa_phosphate': 0.0}
        if atom_name in base_atoms:
            residue_sasa_parts[res_key]['sasa_base'] += float(sasa_val)
        elif atom_name in sugar_atoms:
            residue_sasa_parts[res_key]['sasa_sugar'] += float(sasa_val)
        elif atom_name in phosphate_atoms:
            residue_sasa_parts[res_key]['sasa_phosphate'] += float(sasa_val)
    return residue_sasa_parts

# Secondary structure mapping (from manifest or external file)
SS_MAP = {
    'helix': [1, 0, 0, 0, 0, 0, 0, 0],
    'hairpin_loop': [0, 1, 0, 0, 0, 0, 0, 0],
    'internal_loop': [0, 0, 1, 0, 0, 0, 0, 0],
    'bulge': [0, 0, 0, 1, 0, 0, 0, 0],
    'knotted': [0, 0, 0, 0, 1, 0, 0, 0],
    'other': [0, 0, 0, 0, 0, 1, 0, 0], 
    'junction': [0, 0, 0, 0, 0, 0, 1, 0],
    'g_quadruplex': [0, 0, 0, 0, 0, 0, 0, 1]
}


def get_binding_site_labels(pdb_path, ligand_name, threshold=5.0):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", pdb_path)
    except FileNotFoundError:
        print(f"错误: PDB文件未找到于 {pdb_path}")
        return None

    """过滤全部小分子结合的样本
    ligand_atoms = [atom for atom in structure.get_atoms() if atom.get_parent().get_resname().strip() == ligand_name]
    if not ligand_atoms:
        print(f"警告: 在PDB文件 {pdb_path} 中未找到配体 '{ligand_name}'。")
        return None
    """

    """# --- 修改后的代码 (只统计第一个出现的离子/配体) ---
    # 1. 先按残基(Residue)级别收集，而不是直接收集原子
    ligand_residues = [res for res in structure.get_residues() if res.get_resname().strip() == ligand_name.upper()]

    if not ligand_residues:
        print(f"警告: 在PDB文件 {pdb_path} 中未找到配体 '{ligand_name}'。")
        return None

    # 2. 强制只取第一个残基 (The First Instance)
    # PDB解析通常按 Chain ID 和 Residue Number 排序
    target_residue = ligand_residues[0]

    # 3. 获取该特定残基下的原子
    ligand_atoms = [atom for atom in target_residue.get_atoms()]
    # (调试打印，可选)
    # print(f"仅使用第一个配体生成标签: {target_residue.get_full_id()}")
    """

    """过滤所有离子，只取第一个结合位点"""
    # 常见的金属离子列表 (大写)
    IONS = {'MG', 'ZN', 'NA', 'K', 'CA', 'MN', 'FE', 'CO', 'NI', 'CU', 'CD', 'HG', 'CL'}
    clean_name = ligand_name.strip().upper()
    # 1. 找到所有同名配体残基
    ligand_residues = [res for res in structure.get_residues() if res.get_resname().strip().upper() == clean_name]

    if not ligand_residues:
        print(f"警告: 在PDB文件 {pdb_path} 中未找到配体 '{ligand_name}'。")
        return None

    # 2. 分类处理
    target_atoms = []
    #全部取
    #for res in ligand_residues:
    #    target_atoms.extend([atom for atom in res.get_atoms()])


    if clean_name in IONS:
        # --- 情况 A: 是离子 ---
        # 只取第一个残基 (Index 0) 取两个# 列表切片 [:2]：如果有1个就取1个，有10个就取前2个
        # 1. 取前 8 个残基 (如果不足8个，切片会自动取所有，返回的是一个列表)，适用于取第一个
        target_residues_list = ligand_residues[:4]

        target_atoms = []
        # 2. 必须遍历这个列表，从每一个残基中提取原子
        for res in target_residues_list:
            target_atoms.extend([atom for atom in res.get_atoms()])

        # (可选) 打印调试信息看看到底取了几个
        # print(f"离子 {clean_name}: 取了 {len(target_residues_list)} 个实例")
    else:
        # --- 情况 B: 是有机小分子 ---
        # 取所有残基的所有原子 (保持原样)
        for res in ligand_residues:
            target_atoms.extend([atom for atom in res.get_atoms()])

    if not target_atoms:
        return None

    ligand_coords = np.array([atom.get_coord() for atom in target_atoms])
    """over"""

    #原来的ligand_coords = np.array([atom.get_coord() for atom in ligand_atoms])

    node_labels = []
    for residue in structure.get_residues():
        if residue.get_id()[0] == ' ':
            rna_atom_coords = np.array([atom.get_coord() for atom in residue])
            if rna_atom_coords.shape[0] == 0: continue

            diffs = np.expand_dims(rna_atom_coords, axis=1) - np.expand_dims(ligand_coords, axis=0)
            min_dist = np.min(np.sqrt(np.sum(diffs ** 2, axis=-1)))
            node_labels.append(1 if min_dist < threshold else 0)

    if not node_labels:
        print(f"警告: 在PDB文件 {pdb_path} 中未能提取任何RNA核苷酸。")
        return None

    return torch.tensor(node_labels, dtype=torch.float)


class RNALigandDataset(Dataset):
    def __init__(self, root_dir, manifest_df, transform=None, pre_transform=None, pdb_dir=None,
                 label_dist_threshold=5.0):
        self.root_dir = root_dir
        self.manifest = manifest_df.reset_index(drop=True)
        self.label_dist_threshold = label_dist_threshold
        '''文件路径配置'''
        self.rna_features_dir = Rna_features_dir
        self.ligand_graphs_dir = Ligand_graphs_dir
        self.pdb_dir = pdb_dir
        super(RNALigandDataset, self).__init__(root_dir, transform, pre_transform)

    def len(self):
        return len(self.manifest)

    def _load_torch_file(self, path):
        # --- *** 这是对警告的修复 *** ---
        # 这个辅助函数会优先尝试新的、安全的方法。
        # 如果失败，它会回退到旧的方法。
        try:
            return torch.load(path, map_location=torch.device('cpu'), weights_only=True)
        except:
            # 为可能不兼容的旧.pt文件提供回退方案
            return torch.load(path, map_location=torch.device('cpu'))
        # --- *** 修复结束 *** ---

    def get(self, idx):
        info = self.manifest.iloc[idx]
        complex_id = info['complex_id']
        ligand_name = info['ligand_name']

        pdb_path = os.path.join(self.pdb_dir, f"{complex_id}.pdb")
        rna_node_labels = get_binding_site_labels(pdb_path, ligand_name, threshold=self.label_dist_threshold)
        if rna_node_labels is None:
            print(f"跳过样本 {complex_id}，因为无法生成标签。")
            return None

        # 使用新的辅助函数来加载文件
        rna_feature_path = os.path.join(self.rna_features_dir, f"{complex_id}.pt")
        rna_data_dict = self._load_torch_file(rna_feature_path)

        rna_node_feat = rna_data_dict['seq_feat']

        # --- 额外特征: SASA (base/sugar/phosphate) as local features;
        # 全局特征包括 secondary-structure one-hot (8 dims), gc_content (mean) 和 chain_length
        # 默认值：如果无法计算，将补零以保持与现有特征对齐
        extra_feats = None
        gc_mean = 0.0
        chain_length = 0
        ss_global = [0.0] * 8
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('complex', pdb_path)
            # 收集 RNA residues (same logic as labels)
            rna_residues = [res for res in structure.get_residues() if res.get_id()[0] == ' ']

            # Attempt DSSP for ASA and secondary structure when available
            dssp_map = None
            if DSSP is not None:
                try:
                    model = structure[0]
                    dssp = DSSP(model, pdb_path)
                    # dssp key is (chain_id, (' ', resseq, icode))
                    dssp_map = {k: v for k, v in dssp.property_dict.items()} if hasattr(dssp, 'property_dict') else None
                except Exception:
                    dssp_map = None

            # Build base sequence from residue names with simple mapping
            three_to_one = {
                'A': 'A', 'G': 'G', 'C': 'C', 'U': 'U',
                'ADE': 'A', 'GUA': 'G', 'CYT': 'C', 'URA': 'U',
                'A  ': 'A', 'G  ': 'G', 'C  ': 'C', 'U  ': 'U',
                'PSU': 'U', 'I': 'G'
            }
            seq = []
            pdb_resi_list = []  # <--- [新增 1] 用于存储 PDB 序号
            for res in rna_residues:
                name = res.get_resname().strip()
                base = three_to_one.get(name, None)
                if base is None:
                    # try first character
                    if len(name) >= 1 and name[0] in 'AGCU':
                        base = name[0]
                    else:
                        base = 'N'
                seq.append(base)

                # <--- [新增 2] 提取残基编号 (Residue Sequence Number)
                # res.get_id()[1] 是整数编号
                # res.get_id()[2] 是插入码(Insertion Code)，通常为空，但也可能有 "A"
                # 这里我们组合一下，变成字符串，比如 "104" 或 "104A"
                resi_num = str(res.get_id()[1]) + res.get_id()[2].strip()
                pdb_resi_list.append(resi_num)


            # GC content per-residue using sliding window -> compute global mean and chain length
            win = 11
            half = win // 2
            seq_upper = [s if s in 'GC' else 'N' for s in seq]
            gc_list = []
            L = len(seq_upper)
            for i in range(L):
                left = max(0, i - half)
                right = min(L, i + half + 1)
                window = seq_upper[left:right]
                gc = 0.0
                for c in window:
                    if c == 'G' or c == 'C':
                        gc += 1
                gc_list.append(gc / (right - left) if (right - left) > 0 else 0.0)
            gc_mean = float(np.mean(gc_list)) if L > 0 else 0.0
            chain_length = L

            # Secondary structure default per-residue placeholder (not used as local feature)
            ss_onehot = []
            # SASA placeholders per residue: we'll compute sugar/base/phosphate and store in that order
            sasa_sb_p = []

            # helper to classify atom into sugar/base/phosphate by atom name
            def atom_group(atom_name):
                an = atom_name.upper()
                if an in ('P', 'OP1', 'OP2', "OP3", 'O5\'') or an.startswith('OP'):
                    return 'P'
                # sugar atoms often have prime notation or O4', C1', etc
                if '\'' in an or any(x in an for x in ("C1", "C2", "C3", "C4", "C5", "O4", 'O2')):
                    return 'S'
                # base atoms: N, C that are not sugar or phosphate
                return 'B'

            # Prepare data for ASA calculation
            # Try PyMOL-based exact per-atom SASA if requested and available; otherwise prepare atom coords for density proxy
            residue_sasa_lookup = None
            if INCLUDE_SASA and pymol2 is not None:
                try:
                    all_atom_sasa = calculate_all_atom_sasa_with_pymol(pdb_path)
                    if all_atom_sasa:
                        residue_sasa_lookup = aggregate_sasa_parts(all_atom_sasa)
                except Exception:
                    residue_sasa_lookup = None

            try:
                # Build KD-tree of all atom coords for density fallback
                all_atom_coords = np.array([atom.get_coord() for atom in structure.get_atoms()])
            except Exception:
                all_atom_coords = None

            for i, res in enumerate(rna_residues):
                # secondary structure
                ss_code = 'C'  # coil default
                try:
                    # Try to find dssp entry for this residue
                    if DSSP is not None:
                        # Biopython DSSP mapping varies; attempt a safe lookup
                        for key in list(DSSP.__dict__.keys()):
                            pass
                except Exception:
                    pass

                # crude one-hot mapping: H (helix), E (strand), else C (kept for backward compatibility)
                if ss_code == 'H':
                    ss_onehot.append([1.0, 0.0, 0.0])
                elif ss_code == 'E':
                    ss_onehot.append([0.0, 1.0, 0.0])
                else:
                    ss_onehot.append([0.0, 0.0, 1.0])

                # SASA proxy: distribute residue-level exposure to S/B/P by atom counts
                atoms = [a for a in res.get_atoms()]
                count_S = 0
                count_B = 0
                count_P = 0
                for a in atoms:
                    grp = atom_group(a.get_name())
                    if grp == 'S':
                        count_S += 1
                    elif grp == 'P':
                        count_P += 1
                    else:
                        count_B += 1

                total_atoms = max(1, count_S + count_B + count_P)

                # Try to get per-residue ASA from PyMOL aggregation if available
                total_asa = None
                b_asa = 0.0
                s_asa = 0.0
                p_asa = 0.0
                if residue_sasa_lookup is not None:
                    try:
                        chain_id = res.get_parent().get_id()
                        resseq = res.get_id()[1]
                        key = (chain_id, resseq)
                        parts = residue_sasa_lookup.get(key)
                        if parts is not None:
                            b_asa = parts.get('sasa_base', 0.0)
                            s_asa = parts.get('sasa_sugar', 0.0)
                            p_asa = parts.get('sasa_phosphate', 0.0)
                            total_asa = b_asa + s_asa + p_asa
                    except Exception:
                        total_asa = None

                if total_asa is None:
                    # proxy: use inverse local atom density as exposure
                    if all_atom_coords is not None and len(atoms) > 0:
                        coord = atoms[0].get_coord()
                        dists = np.sqrt(np.sum((all_atom_coords - coord) ** 2, axis=1))
                        # count neighbors within 6A
                        neigh = (dists < 6.0).sum()
                        # higher neighbors -> more buried -> lower exposure
                        exposure = max(0.0, 1.0 - (neigh / 200.0))
                        total_asa = exposure * 100.0
                    else:
                        total_asa = 0.0

                    # distribute using atom counts when no PyMOL exact data
                    s_asa = total_asa * (count_S / total_atoms)
                    b_asa = total_asa * (count_B / total_atoms)
                    p_asa = total_asa * (count_P / total_atoms)

                sasa_sb_p.append([b_asa, s_asa, p_asa])

                # 1. SASA (Log normalized)
                sasa_tensor = torch.tensor(sasa_sb_p, dtype=torch.float).view(-1, 3)
                sasa_log = torch.log1p(sasa_tensor)

                # 2. Local Secondary Structure
                ss_tensor = torch.tensor(ss_onehot, dtype=torch.float).view(-1, 3)
                # [修复] 提前初始化 ss_label，防止 UnboundLocalError
                ss_label = None
                if 'secondary_structure' in self.manifest.columns:
                    try:
                        ss_label = info.get('secondary_structure')
                    except Exception:
                        ss_label = None
                key = ss_label.lower() if isinstance(ss_label, str) else ''
                ss_global = SS_MAP.get(key, [0.0] * 8)

                global_vec = list(ss_global) + [float(gc_mean), float(chain_length)]
                global_tensor = torch.tensor(global_vec, dtype=torch.float)
                global_expanded = global_tensor.unsqueeze(0).expand(sasa_tensor.shape[0], -1)
                
                # 4. 移除 ss_tensor，直接拼接 SASA (3维) + Global (10维) = 13 dims
                # [修复] 确保使用 sasa_log
                extra_feats = torch.cat([sasa_log, global_expanded], dim=1)

        except Exception as e:
            print(f"Warning: 计算额外特征时出错: {e}")
            # [修改] 维度从 16 改为 13
            extra_feats = torch.zeros((rna_node_feat.shape[0], 13), dtype=torch.float)

        #正确代码！！！！
        if extra_feats is not None:
            # [核心修复] 强制对齐基础语言模型特征的数据类型和维度
            rna_node_feat = rna_node_feat.float()
            if rna_node_feat.dim() == 3 and rna_node_feat.shape[0] == 1:
                rna_node_feat = rna_node_feat.squeeze(0)
            
            # 确保 extra_feats 也在同一个设备并且是 float32
            extra_feats = extra_feats.to(rna_node_feat.device).float()

            try:
                if extra_feats.shape[0] == rna_node_feat.shape[0]:
                    rna_node_feat = torch.cat([rna_node_feat, extra_feats], dim=1)
                else:
                    # 维度长度不匹配时的容错补零 (13维)
                    extra_feats = torch.zeros((rna_node_feat.shape[0], 13), dtype=torch.float, device=rna_node_feat.device)
                    rna_node_feat = torch.cat([rna_node_feat, extra_feats], dim=1)
            except Exception as e:
                # 暴露真实的拼接错误，而不是用 pass 掩盖
                print(f"致命错误：特征拼接失败，原因: {e}")

        # # ==========================================
        # # 🧪 消融实验模式：只保留语言模型特征 (768维)
        # # ==========================================
        # # 1. 依然要强制对齐数据类型为 float32，否则后续网络会报错
        # rna_node_feat = rna_node_feat.float()
        # # 2. 依然要去除可能多余的 batch 维度 (1, L, 768) -> (L, 768)
        # if rna_node_feat.dim() == 3 and rna_node_feat.shape[0] == 1:
        #     rna_node_feat = rna_node_feat.squeeze(0)
        # # 3. 彻底抛弃 extra_feats，不进行任何 torch.cat 拼接操作！
        # # rna_node_feat 完美保持在原始的 768 维

        # Append global features (secondary structure one-hot, gc_mean, chain_length) to rna_cls_feat later

        if rna_node_feat.shape[0] != len(rna_node_labels):
            print(
                f"跳过样本 {complex_id}: 特征节点数({rna_node_feat.shape[0]})与PDB核苷酸数({len(rna_node_labels)})不匹配。")
            return None

        rna_adj_mat = rna_data_dict['mat_feat']
        rna_edge_index = (rna_adj_mat > 0).nonzero(as_tuple=False).t().contiguous()
        rna_graph = Data(x=rna_node_feat.float(), edge_index=rna_edge_index.long())

        # --- 修改开始：加载 Ligand Graph 并增强鲁棒性 ---
        ligand_graph_path = os.path.join(self.ligand_graphs_dir, f"{ligand_name}.pt")
        ligand_data_dict = self._load_torch_file(ligand_graph_path)

        ligand_node_feat = ligand_data_dict['atom_features']
        ligand_edge_index = ligand_data_dict['edge_index']

        # [新增] 鲁棒性转换：如果是 numpy 数组，转为 Tensor
        if isinstance(ligand_node_feat, np.ndarray):
            ligand_node_feat = torch.from_numpy(ligand_node_feat)

        if isinstance(ligand_edge_index, np.ndarray):
            ligand_edge_index = torch.from_numpy(ligand_edge_index)

        # 获取边特征
        ligand_edge_attr = None
        if 'edge_attr' in ligand_data_dict:
            ligand_edge_attr = ligand_data_dict['edge_attr']
        elif 'edge_features' in ligand_data_dict:
            ligand_edge_attr = ligand_data_dict['edge_features']
        elif 'bond_features' in ligand_data_dict:
            ligand_edge_attr = ligand_data_dict['bond_features']

        # [新增] 边特征鲁棒性转换
        if ligand_edge_attr is not None:
            if isinstance(ligand_edge_attr, np.ndarray):
                ligand_edge_attr = torch.from_numpy(ligand_edge_attr)

        # 构造 PyG Data 对象
        if ligand_edge_attr is not None:
            ligand_graph = Data(
                x=ligand_node_feat.float(),
                edge_index=ligand_edge_index.long(),  # 现在它是 Tensor 了，可以使用 .long()
                edge_attr=ligand_edge_attr.float()
            )
        else:
            ligand_graph = Data(
                x=ligand_node_feat.float(),
                edge_index=ligand_edge_index.long()
            )
        # --- 修改结束 ---


        """"# 使用新的辅助函数来加载文件
        # --- START OF FIX: 适配 edge_features 键名 ---
        # 使用新的辅助函数来加载文件
        ligand_graph_path = os.path.join(self.ligand_graphs_dir, f"{ligand_name}.pt")
        ligand_data_dict = self._load_torch_file(ligand_graph_path)

        # 1. 获取节点特征
        ligand_node_feat = ligand_data_dict['atom_features']

        # 2. 获取边索引
        ligand_edge_index = ligand_data_dict['edge_index']

        # 3. 获取边特征 (兼容多种命名)
        ligand_edge_attr = None
        if 'edge_features' in ligand_data_dict:
            ligand_edge_attr = ligand_data_dict['edge_features']
        elif 'edge_attr' in ligand_data_dict:
            ligand_edge_attr = ligand_data_dict['edge_attr']
        elif 'bond_features' in ligand_data_dict:
            ligand_edge_attr = ligand_data_dict['bond_features']
        #修改：
        else:
            ligand_edge_attr = None  # 对单原子离子可能为 None

        # 4. 构造 PyG Data 对象
        # 注意：PyG 模型默认寻找 .edge_attr 属性，所以我们必须把读取到的特征传给 edge_attr 参数
        if ligand_edge_attr is not None:
            ligand_graph = Data(
                x=ligand_node_feat.float(),
                edge_index=ligand_edge_index.long(),
                edge_attr=ligand_edge_attr.float()  # 映射成功！
            )
        else:
            # 如果真的没有边特征，edge_attr 为 None
            ligand_graph = Data(
                x=ligand_node_feat.float(),
                edge_index=ligand_edge_index.long()
            )

"""

        """
        ligand_graph_path = os.path.join(self.ligand_graphs_dir, f"{ligand_name}.pt")
        ligand_data_dict = self._load_torch_file(ligand_graph_path)

        ligand_node_feat = ligand_data_dict['atom_features']
        ligand_edge_index = ligand_data_dict['edge_index']
        ligand_graph = Data(x=ligand_node_feat.float(), edge_index=ligand_edge_index.long())
        """

        rna_cls_feat = rna_data_dict['cls_feat']
        if rna_cls_feat.dim() > 1 and rna_cls_feat.shape[0] > 1:
            rna_cls_feat = rna_cls_feat.mean(dim=0)

        # Ensure rna_cls_feat is 1D tensor
        try:
            rna_cls_feat = rna_cls_feat.view(-1)
        except Exception:
            rna_cls_feat = rna_cls_feat.flatten()

        # Build and append global features: ss_global (8 dims), gc_mean (1), chain_length (1)
        try:
            global_feats = torch.tensor(list(ss_global) + [float(gc_mean), float(chain_length)], dtype=torch.float)
            rna_cls_feat = torch.cat([rna_cls_feat, global_feats], dim=0)
        except Exception:
            # fallback: no global features appended
            pass

        # [修改] 确保把 seq (列表) 转成字符串
        sequence_str = "".join(seq)

        main_data = Data(
            rna_graph=rna_graph,
            ligand_graph=ligand_graph,
            rna_cls_feat=rna_cls_feat.float().unsqueeze(0),
            rna_node_y=rna_node_labels.unsqueeze(1),
            graph_y=torch.tensor([info['label']], dtype=torch.float),
            complex_id = complex_id,
            sequence=sequence_str, #原始序列字符串
            pdb_resi = pdb_resi_list  # <--- [新增 3] 存入 Data
        )
        return main_data