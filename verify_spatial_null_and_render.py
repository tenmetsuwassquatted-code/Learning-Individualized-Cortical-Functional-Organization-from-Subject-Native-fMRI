# verify_spatial_null_and_render.py
# -*- coding: utf-8 -*-
"""
基于 Spin Test (Spatial Null) 的个体化领地置换检验与 Lynch 风格边界轮廓渲染。

逻辑链条：
    Observed Dice → Spatial Null Distribution (Spin Test) → Permutation p-value & Effect Size
    → 筛选显著扩张 Parcel → Lynch 风格黑线轮廓 + 彩色 Residual 叠加渲染

使用说明：
    1. 先准备模型 checkpoint，并确保结果目录中存在 scan1/scan2 的 parcellation_result.npz。
    2. 运行脚本前，确认 --out-root 指向保存聚类结果的根目录，脚本会从该目录下读取被试结果。
    3. 脚本会对每个 Parcel 执行 Spin Test，并渲染前 3 个最显著效果的 Lynch 风格视图。

示例命令：
    python verify_spatial_null_and_render.py \
        --checkpoint checkpoints/brain_branch.pt \
        --out-root results \
        --subject-dir-root dataset \
        --device cuda

参数说明：
    --checkpoint  : 训练好的模型 checkpoint 路径
    --out-root    : 结果根目录，包含 sub-*/scan1 和 scan2 等结果
    --n-perm      : Spin Test 的随机置换次数，推荐正式实验使用 1000
    --device      : 运行设备，如 cpu、cuda 或 cuda:0

输出：
    生成每个样本的 lynch_territory_viz/ 中的 PNG 渲染图，以及终端中的统计检验结果。
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import torch

from common import (
    load_model_from_checkpoint,
    extract_node_rep,
    load_surface_gii,
    detect_sphere_reg_file,
    load_nodes_mapping,
    render_preview,
)
from data import get_dataset


# ======================================================================
# 1. 匈牙利对齐与领地分解
# ======================================================================

def align_parcels_hungarian(pred_labels: np.ndarray, prior_labels: np.ndarray):
    """基于交集节点数混淆矩阵的匈牙利标签对齐。"""
    valid_mask = (pred_labels >= 0) & (prior_labels >= 0)
    p_valid, g_valid = pred_labels[valid_mask], prior_labels[valid_mask]

    p_ids, g_ids = np.unique(p_valid), np.unique(g_valid)
    overlap_matrix = np.zeros((len(p_ids), len(g_ids)), dtype=np.int64)
    p_id_to_idx = {pid: i for i, pid in enumerate(p_ids)}
    g_id_to_idx = {gid: j for j, gid in enumerate(g_ids)}

    for p_val, g_val in zip(p_valid, g_valid):
        overlap_matrix[p_id_to_idx[p_val], g_id_to_idx[g_val]] += 1

    row_ind, col_ind = linear_sum_assignment(-overlap_matrix)

    matching_map = {p_ids[r]: g_ids[c] for r, c in zip(row_ind, col_ind)}
    aligned_pred = np.full_like(pred_labels, -1)
    for p_id, mapped_g_id in matching_map.items():
        aligned_pred[pred_labels == p_id] = mapped_g_id

    return aligned_pred


def extract_territory_masks(aligned_pred: np.ndarray, prior_labels: np.ndarray, k: int):
    """提取 Parcel k 的领地区域掩码 (C_sk, E_sk, L_sk)。"""
    mask_P = aligned_pred == k
    mask_G = prior_labels == k

    E_mask = mask_P & (~mask_G) & (prior_labels >= 0)  # Expansion / Encroachment
    C_mask = mask_P & mask_G                           # Conserved
    L_mask = mask_G & (~mask_P)                           # Contraction

    return C_mask, E_mask, L_mask, mask_G, mask_P


def calc_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个布尔掩码的 Dice 相似度系数。"""
    s1, s2 = mask1.sum(), mask2.sum()
    if s1 + s2 == 0:
        return 0.0
    return float(2.0 * (mask1 & mask2).sum() / (s1 + s2))


# ======================================================================
# 2. 空间空模型 (Spatial Null Model / Spin Test)
# ======================================================================

def generate_spherical_spin_permutation(
    coords_sphere: np.ndarray,
    seed: int = None
) -> np.ndarray:
    """
    生成三维随机旋转矩阵，作用于球面坐标，返回旋转后的节点最近邻重映射索引。
    保持原始皮层数据极强的空间自相关性 (Spatial Autocorrelation)。
    """
    if seed is not None:
        np.random.seed(seed)

    # 随机 3D 旋转
    rot = R.random().as_matrix()
    rotated_coords = coords_sphere @ rot.T

    # 建立 KD-Tree 寻找到旋转后最近的原节点位置
    tree = cKDTree(coords_sphere)
    _, perm_indices = tree.query(rotated_coords, k=1)
    return perm_indices


def compute_spatial_null_dice_distribution(
    E_scan1: np.ndarray,
    E_scan2: np.ndarray,
    coords_sphere: np.ndarray,
    n_perm: int = 1000
) -> np.ndarray:
    """对 Expansion 领地 E_sk 进行 Spin Test，构建 Spatial Null Dice 分布。"""
    null_dices = np.zeros(n_perm, dtype=np.float64)

    for i in range(n_perm):
        perm_idx = generate_spherical_spin_permutation(coords_sphere, seed=i)
        # 将 scan2 的 E 区域进行空间随机旋转，打乱具体位点但保持皮层拓扑斑块连续性
        E_scan2_spun = E_scan2[perm_idx]
        null_dices[i] = calc_dice(E_scan1, E_scan2_spun)

    return null_dices


# ======================================================================
# 3. Lynch 风格黑线轮廓与领地叠加渲染
# ======================================================================

def render_lynch_style_parcel(
    subject_dir: Path,
    subject: str,
    hemi: str,
    orig_v: np.ndarray,
    hemi_mask: np.ndarray,
    mask_G_k: np.ndarray,
    mask_E_k: np.ndarray,
    out_png: Path,
    parcel_k: int
):
    from common import load_native_vertex_count

    # 1. 确保正确的 subject 路径 (dataset/sub-009)
    sub_anat_dir = subject_dir / subject / "anat"
    
    # 2. 读取原生顶点数 (传递 dataset/sub-009 给 load_native_vertex_count 或直接在此读取)
    # common.load_native_vertex_count 期望传入包含 anat 的被试目录
    n_v = load_native_vertex_count(subject_dir / subject, subject, hemi)
    
    roi_map = np.zeros(n_v, dtype=np.float32)

    v_ids = orig_v[hemi_mask]
    g_nodes_hemi = mask_G_k[hemi_mask]
    e_nodes_hemi = mask_E_k[hemi_mask]

    # 给 Group Parcel 赋基底值 (1.0)，Expansion 区域赋高亮值 (2.0)
    roi_map[v_ids[g_nodes_hemi]] = 1.0
    roi_map[v_ids[e_nodes_hemi]] = 2.0  

    # 3. 关联解剖表面文件 (优先使用 inflated，若无则退回 pial)
    inflated = sub_anat_dir / f"{subject}_hemi-{hemi}_inflated.surf.gii"
    pial = sub_anat_dir / f"{subject}_hemi-{hemi}_pial.surf.gii"
    sulc = sub_anat_dir / f"{subject}_hemi-{hemi}_sulc.shape.gii"

    surf_mesh = inflated if inflated.exists() else pial
    bg_map = sulc if sulc.exists() else None

    if not surf_mesh.exists():
        print(f"  [跳过渲染] 找不到表面文件: {surf_mesh}")
        return

    hemi_label = "left" if hemi == "L" else "right"

    render_preview(
        surf_mesh=surf_mesh,
        roi_vec=roi_map,
        out_png=out_png,
        hemi=hemi_label,
        bg_map=bg_map,
        title=f"{subject} Parcel-{parcel_k} Lynch-style Territory (Base=Group, High=Expansion)"
    )


# ======================================================================
# 4. 主检验与诊断入口
# ======================================================================

def run_spatial_null_validation(
    checkpoint_path: str,
    out_root: str,
    n_perm: int = 200,
    device_str: str = "cpu"
):
    device = torch.device(device_str)
    model, cfg = load_model_from_checkpoint(checkpoint_path, device)
    brain_dataset = get_dataset("brain", pe_dim=cfg["pe_dim"])

    out_dir = Path(out_root)
    subject_dirs = [d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")]

    if not subject_dirs:
        print(f"[错误] 找不到结果目录: {out_root}")
        return

    for sub_path in subject_dirs:
        subject_id = sub_path.name
        print("\n" + "=" * 80)
        print(f"严密统计检验分析: 被试 {subject_id}")
        print("=" * 80)

        scan1_npz = sub_path / "scan1" / "parcellation_result.npz"
        scan2_npz = sub_path / "scan2" / "parcellation_result.npz"

        if not scan1_npz.exists() or not scan2_npz.exists():
            continue

        data_s1 = np.load(scan1_npz, allow_pickle=True)
        data_s2 = np.load(scan2_npz, allow_pickle=True)

        pred_raw_1, pred_raw_2 = data_s1["cluster_pred_raw"], data_s2["cluster_pred_raw"]
        prior_labels = data_s1["prior_labels"]

        # 匈牙利匹配
        aligned_P1 = align_parcels_hungarian(pred_raw_1, prior_labels)
        aligned_P2 = align_parcels_hungarian(pred_raw_2, prior_labels)

        # 读取节点的球面坐标（用于 Spin Test）
        nodes_npz_path = Path(data_s1["path"].item()).with_suffix(".nodes.npz")
        mapping = load_nodes_mapping(nodes_npz_path)

        # 提取左半球球面坐标构建 3D 旋转模型
        sphere_path_L = detect_sphere_reg_file(
            subject_dir=args.subject_dir_root, 
            subject=subject_id, 
            hemi="L", 
            space="fsaverage"
        )
        coords_sphere_L, _ = load_surface_gii(sphere_path_L)

        mask_L = np.array(["L" in str(h).upper() for h in mapping["hemisphere"]])
        node_orig_v_L = mapping["original_surface_vertex"][mask_L]
        nodes_sphere_coords = coords_sphere_L[node_orig_v_L]  # 左半球图节点的球面坐标

        parcels = np.unique(prior_labels[prior_labels >= 0])

        results = []

        print(f"\n[执行 Spatial Null Spin Test (n_perm={n_perm})...]")
        print(f"{'Parcel K':<10}{'Observed Dice':<16}{'Null Mean±SD':<20}{'Permutation p':<16}{'Effect Size (z)':<15}")
        print("-" * 77)

        for k in parcels:
            C1, E1, _, G1, _ = extract_territory_masks(aligned_P1, prior_labels, k)
            C2, E2, _, G2, _ = extract_territory_masks(aligned_P2, prior_labels, k)

            # 左半球节点掩码
            E1_L = E1[mask_L]
            E2_L = E2[mask_L]

            if E1_L.sum() < 5 or E2_L.sum() < 5:
                continue

            # 1. 计算真实观察到的 Observed Dice
            obs_dice = calc_dice(E1_L, E2_L)

            # 2. 空间 Null 分布 (Spin Test)
            null_dices = compute_spatial_null_dice_distribution(E1_L, E2_L, nodes_sphere_coords, n_perm=n_perm)

            # 3. 置换 p 值与 Z-score 效应量 (Effect Size)
            p_val = (np.sum(null_dices >= obs_dice) + 1.0) / (n_perm + 1.0)
            # 计算标准差与 Z-score 修正
            null_mean = np.mean(null_dices)
            null_std = np.std(null_dices)

            if null_std < 1e-5:
                z_score = 0.0  # 避免除以极小值导致 Z-score 暴涨
            else:
                z_score = (obs_dice - null_mean) / null_std

            results.append({
                "k": k,
                "obs_dice": obs_dice,
                "null_mean": null_mean,
                "null_std": null_std,
                "p_val": p_val,
                "z_score": z_score,
                "mask_G": G1,
                "mask_E": E1,
            })

            print(f"{k:<10}{obs_dice:<16.4f}{f'{null_mean:.3f}±{null_std:.3f}':<20}{p_val:<16.4f}{z_score:<15.2f}")

        # 4. 按 Effect Size 效应量（Z-score）降序排列，筛选出最显著突出的 Territory Residual Parcel
        results.sort(key=lambda x: x["z_score"], reverse=True)

        print("\n" + "=" * 80)
        print("Lynch 风格 Territory Residual 图形渲染")
        print("=" * 80)

        viz_dir = sub_path / "lynch_territory_viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        top_k_parcels = results[:3]  # 抽样前 3 个最显著效应量的 Parcel
        for res in top_k_parcels:
            k = res["k"]
            print(f"  -> 正在渲染效应量最大的 Parcel {k} (Z-score={res['z_score']:.2f}, p={res['p_val']:.4f})...")

            out_png = viz_dir / f"Parcel_{k}_territory_residual.png"
            render_lynch_style_parcel(
                subject_dir=Path(args.subject_dir_root),
                subject=subject_id,
                hemi="L",
                orig_v=mapping["original_surface_vertex"],
                hemi_mask=mask_L,
                mask_G_k=res["mask_G"],
                mask_E_k=res["mask_E"],
                out_png=out_png,
                parcel_k=k
            )

        print(f"\n[完成] 渲染图已存入: {viz_dir}")
        break  # 默认跑完第一个被试样例


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spin Test 与 Lynch 风格领地渲染")
    parser.add_argument("--checkpoint", required=True, help="checkpoint 路径")
    parser.add_argument("--out-root", default="results", help="本地聚类结果目录")
    parser.add_argument("--n-perm", type=int, default=200, help="Spin test 置换次数 (推荐正式实验设为 1000)")
    parser.add_argument("--device", default="cpu")
    # 新增 fMRIPrep anat 目录参数
    parser.add_argument("--subject-dir-root", required=True, help="fMRIPrep 被试根目录（含 sub-XXX/anat/）")
    args = parser.parse_args()

    run_spatial_null_validation(args.checkpoint, args.out_root, args.n_perm, args.device)