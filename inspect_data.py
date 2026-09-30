# inspect_data.py
# -*- coding: utf-8 -*-
"""
数据流检测与 Lynch 式个体化领地 (E_sk) 假说验证脚本。

检测内容：
    1. 节点与标签空间映射 (Group Prior G vs Personalized P_s)
    2. 匈牙利匹配 (Hungarian Matching) 标签重对齐矩阵
    3. Conserved (C_sk), Expansion (E_sk), Contraction (L_sk) 节点提取
    4. Lynch 功能一致性验证：E_sk 的 BOLD FC / Embedding 距离是更偏向目标网络 G_k 还是宿主网络 G_host
    5. Scan1 与 Scan2 (Split-half) 跨时间段重叠率 (Test-Retest Stability)

使用说明：
    1. 先运行本地聚类脚本生成结果目录，目录结构应如下：
           results/
           └── sub-<id>/
               ├── scan1/
               │   └── parcellation_result.npz
               └── scan2/
                   └── parcellation_result.npz
    2. 再执行本脚本进行后处理与假说验证：
           python inspect_data.py --checkpoint checkpoints/brain_branch.pt --out-root results --device cuda

    3. 若使用 CPU 进行推理，可替换为：
           python inspect_data.py --checkpoint checkpoints/brain_branch.pt --out-root results --device cpu

参数说明：
    --checkpoint   模型检查点路径（例如 checkpoints/brain_branch.pt）
    --out-root     本地聚类结果根目录，默认值为 results
    --device       推理设备，默认使用 cuda（可选: cpu, cuda）

输出说明：
    - 会读取每个 sub-* 目录中的 scan1/scan2 parcellation_result.npz
    - 计算匈牙利重对齐后的领地区域统计与 Lynch 功能转型指标
    - 打印 Conserved / Expansion / Contraction 占比和跨时间段可重复性 Dice
"""

import argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
import torch

from common import (
    load_model_from_checkpoint,
    extract_node_rep,
    mean_pairwise_fc,
    within_z,
    between_z,
    to_scipy_csr,
)
from clustering import run_mixed_spectral_clustering
from data import get_dataset


def align_parcels_hungarian(pred_labels: np.ndarray, prior_labels: np.ndarray):
    """
    基于节点组重叠交集计算匈牙利匹配，将 pred_labels 映射到 prior_labels 的 ID 空间。
    
    返回:
        aligned_pred : 重映射后的个体化标签
        matching_map : {pred_cluster_id: prior_parcel_id}
        overlap_matrix : 重叠节点数混淆矩阵
    """
    valid_mask = (pred_labels >= 0) & (prior_labels >= 0)
    p_valid = pred_labels[valid_mask]
    g_valid = prior_labels[valid_mask]

    p_ids = np.unique(p_valid)
    g_ids = np.unique(g_valid)

    # 构建重叠矩阵 O_ij = |P_i \cap G_j|
    overlap_matrix = np.zeros((len(p_ids), len(g_ids)), dtype=np.int64)
    p_id_to_idx = {pid: i for i, pid in enumerate(p_ids)}
    g_id_to_idx = {gid: j for j, gid in enumerate(g_ids)}

    for p_val, g_val in zip(p_valid, g_valid):
        overlap_matrix[p_id_to_idx[p_val], g_id_to_idx[g_val]] += 1

    # 匈牙利算法最大化重叠 (等价于最小化负重叠)
    row_ind, col_ind = linear_sum_assignment(-overlap_matrix)

    matching_map = {}
    for r, c in zip(row_ind, col_ind):
        matching_map[p_ids[r]] = g_ids[c]

    aligned_pred = np.full_like(pred_labels, -1)
    for p_id, mapped_g_id in matching_map.items():
        aligned_pred[pred_labels == p_id] = mapped_g_id

    return aligned_pred, matching_map, overlap_matrix


def analyze_territory_decomposition(aligned_pred: np.ndarray, prior_labels: np.ndarray):
    """
    分解三大领地区域：
        C_sk = P_sk \cap G_sk  (Conserved)
        E_sk = P_sk \ G_sk      (Expansion / Encroachment)
        L_sk = G_sk \ P_sk      (Contraction / Lost)
    """
    unique_parcels = np.unique(prior_labels[prior_labels >= 0])
    territories = {}

    for k in unique_parcels:
        mask_P = aligned_pred == k
        mask_G = prior_labels == k

        C_mask = mask_P & mask_G
        E_mask = mask_P & (~mask_G) & (prior_labels >= 0)
        L_mask = mask_G & (~mask_P)

        territories[k] = {
            "C_nodes": np.where(C_mask)[0],
            "E_nodes": np.where(E_mask)[0],
            "L_nodes": np.where(L_mask)[0],
            "P_nodes": np.where(mask_P)[0],
            "G_nodes": np.where(mask_G)[0],
        }
    return territories


def evaluate_lynch_encroachment_identity(
    record: dict,
    territories: dict,
    prior_labels: np.ndarray
):
    """
    核心假说检验 (Lynch et al. 2024 方法论)：
    针对扩张区域 E_{s,k}，比较其与：
        1. 目标网络 G_k 的功能连接 FC(E_{s,k}, G_k)
        2. 被侵占的宿主网络 G_host 的功能连接 FC(E_{s,k}, G_host)
    如果 FC(E_{s,k}, G_k) > FC(E_{s,k}, G_host)，说明 E_{s,k} 确实发生了功能特性的真实转移！
    """
    bold = record["raw_bold"]
    z_emb = record["node_rep"]

    results = []

    for k, t in territories.items():
        E_nodes = t["E_nodes"]
        G_k_nodes = t["G_nodes"]

        if len(E_nodes) < 5:  # 忽略节点过小的侵占区
            continue

        # 找出 E_nodes 原本在 Group Prior 中属于哪些宿主网络 (G_host)
        host_labels = prior_labels[E_nodes]
        host_counts = np.bincount(host_labels[host_labels >= 0])
        if len(host_counts) == 0:
            continue
        primary_host_k = int(np.argmax(host_counts))

        if primary_host_k == k:
            continue

        G_host_nodes = np.where(prior_labels == primary_host_k)[0]

        # 1. 功能连接 (BOLD FC)
        fc_to_target = mean_pairwise_fc(bold, E_nodes, G_k_nodes)
        fc_to_host = mean_pairwise_fc(bold, E_nodes, G_host_nodes)

        # 2. 特征表达空间距离 (Z Embedding Distance)
        z_dist_to_target = between_z(z_emb, E_nodes, G_k_nodes)
        z_dist_to_host = between_z(z_emb, E_nodes, G_host_nodes)

        results.append({
            "parcel_k": k,
            "primary_host_k": primary_host_k,
            "E_size": len(E_nodes),
            "FC_target": fc_to_target,
            "FC_host": fc_to_host,
            "FC_diff": fc_to_target - fc_to_host,  # > 0 表示成功转型
            "Z_dist_target": z_dist_to_target,
            "Z_dist_host": z_dist_to_host,
            "Z_dist_diff": z_dist_to_host - z_dist_to_target,  # > 0 表示距离目标更近
        })

    return results


def run_inspection(checkpoint_path: str, out_root: str, device_str: str = "cuda"):
    device = torch.device(device_str)
    model, cfg = load_model_from_checkpoint(checkpoint_path, device)
    brain_dataset = get_dataset("brain", pe_dim=cfg["pe_dim"])

    print("\n" + "=" * 80)
    print("本地聚类结果加载与 Lynch 假说验证")
    print("=" * 80)

    # 遍历 results/ 目录下的所有被试
    out_dir = Path(out_root)
    if not out_dir.exists():
        raise FileNotFoundError(f"找不到指定的输出结果根目录: {out_root}")

    subject_dirs = [d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    if not subject_dirs:
        print(f"[提示] 在 {out_root} 下未找到 sub-* 目录，请检查路径。")
        return

    # 为确保能够计算 BOLD FC 和 Embedding 距离，我们需要重新提取一次 node_rep 和 raw_bold（这个过程非常快，不需要重新聚类）
    for sub_path in subject_dirs:
        subject_id = sub_path.name
        print(f"\n--------------------------------------------------")
        print(f"正在分析被试: {subject_id}")

        scan1_npz = sub_path / "scan1" / "parcellation_result.npz"
        scan2_npz = sub_path / "scan2" / "parcellation_result.npz"

        if not scan1_npz.exists() or not scan2_npz.exists():
            print(f"  [跳过] 缺少 scan1 或 scan2 的 parcellation_result.npz")
            continue

        # 加载本地保存的 npz 结果
        data_scan1 = np.load(scan1_npz, allow_pickle=True)
        data_scan2 = np.load(scan2_npz, allow_pickle=True)

        pred_raw_1 = data_scan1["cluster_pred_raw"]
        pred_raw_2 = data_scan2["cluster_pred_raw"]
        prior_labels = data_scan1["prior_labels"]

        # 获取该被试在 dataset 中的索引并提取特征（用于 Lynch FC/Embedding 评估）
        ds_idx = -1
        for i, sid in enumerate(brain_dataset.subject_ids):
            if sid == subject_id:
                ds_idx = i
                break
        if ds_idx == -1:
            print(f"  [跳过] 在 dataset 中未找到被试 {subject_id}")
            continue

        sample = brain_dataset[ds_idx]
        rec_scan1 = extract_node_rep(model, sample, cfg, device, which_scan="scan1")
        rec_scan2 = extract_node_rep(model, sample, cfg, device, which_scan="scan2")

        # 匈牙利匹配
        aligned_P1, _, _ = align_parcels_hungarian(pred_raw_1, prior_labels)
        aligned_P2, _, _ = align_parcels_hungarian(pred_raw_2, prior_labels)

        # 领地分解
        terr1 = analyze_territory_decomposition(aligned_P1, prior_labels)
        terr2 = analyze_territory_decomposition(aligned_P2, prior_labels)

        total_nodes = len(prior_labels)
        c1_total = sum(len(t["C_nodes"]) for t in terr1.values())
        e1_total = sum(len(t["E_nodes"]) for t in terr1.values())
        l1_total = sum(len(t["L_nodes"]) for t in terr1.values())

        print(f"  [领地划分占比 (Scan 1)]:")
        print(f"    - Conserved (C):   {c1_total:6d} 节点 ({c1_total / total_nodes * 100:.2f}%)")
        print(f"    - Expansion (E):   {e1_total:6d} 节点 ({e1_total / total_nodes * 100:.2f}%)")
        print(f"    - Contraction (L): {l1_total:6d} 节点 ({l1_total / total_nodes * 100:.2f}%)")

        # Lynch 评估
        lynch_evals = evaluate_lynch_encroachment_identity(rec_scan1, terr1, prior_labels)
        if lynch_evals:
            print(f"  [Lynch 侵占领地功能转型评估 (抽样前 3 个 Parcel)]:")
            print(f"    {'Parcel K':<10}{'Host K':<10}{'E Size':<10}{'FC->Target':<14}{'FC->Host':<14}{'FC Shift':<12}")
            print("    " + "-" * 70)
            for res in lynch_evals[:3]:
                print(
                    f"    {res['parcel_k']:<10}"
                    f"{res['primary_host_k']:<10}"
                    f"{res['E_size']:<10}"
                    f"{res['FC_target']:<14.4f}"
                    f"{res['FC_host']:<14.4f}"
                    f"{res['FC_diff']:<12.4f}"
                )

        # 可重复性 (Test-Retest Dice)
        e_dices = []
        for k in terr1:
            e1_set = set(terr1[k]["E_nodes"])
            e2_set = set(terr2[k]["E_nodes"])
            if len(e1_set) + len(e2_set) > 0:
                dice = 2.0 * len(e1_set & e2_set) / (len(e1_set) + len(e2_set))
                e_dices.append(dice)
        mean_e_dice = np.mean(e_dices) if e_dices else 0.0
        print(f"  [跨时间段可重复性 (Scan1 vs Scan2 Test-Retest)]:")
        print(f"    - Expansion Territory (E_sk) Mean Dice: {mean_e_dice:.4f}")
        if mean_e_dice > 0.4:
            print("    -> 结论: 发现稳定的个体特异性 Cortical Territory！")
        else:
            print("    -> 结论: 更倾向于图谱边界的随机噪声重划分。")
        # break  # 默认展示第一个被试详情，如需全量可去掉 break
        # print("  -> 结论倾向: 更接近图谱边界的随机噪声重划分。")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="checkpoint 路径")
    parser.add_argument("--out-root", default="results", help="本地聚类结果根目录（默认 results）")
    parser.add_argument("--device", default="cuda", help="推理设备（默认: cuda，可选: cpu, cuda）")
    args = parser.parse_args()

    run_inspection(args.checkpoint, args.out_root, args.device)