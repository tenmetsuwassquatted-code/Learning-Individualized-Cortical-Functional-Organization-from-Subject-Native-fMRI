# -*- coding: utf-8 -*-
"""
eval_scan_rescan.py
===================
脑图谱重测可靠性评估脚本 (Scan-Rescan Reliability Assessment)

指标：
    1. Global ARI: adjusted_rand_score(pred_scan1, pred_scan2)
    2. Parcel-level Dice:
       - 通过匈牙利算法 (Linear Sum Assignment) 求解最优簇对应关系
       - 计算匹配簇对的 Dice 相似度: Dice_k = 2 * |A_k ∩ B_k'| / (|A_k| + |B_k'|)
       - 汇总 Mean / Median / Std / Quantiles，导出各簇 Dice 分布

用法：
    python eval_scan_rescan.py \
        --result-root results \
        --out-dir results_diagnostics/scan_rescan \
        --atlas-key cluster_pred_raw
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from data import get_dataset  # 引入 Dataset 模块


# ======================================================================
# 1. 核心重测指标计算
# ======================================================================

def compute_parcel_dice_hungarian(pred_a: np.ndarray, pred_b: np.ndarray):
    """
    基于匈牙利算法对齐簇标签，并计算各簇的 Parcel-level Dice。
    
    返回:
        dice_scores: 每个匹配簇对的 Dice 数组 [K]
        matching_map: Scan A 簇标签 -> Scan B 簇标签 的映射字典
    """
    # 仅评估有效簇 (label >= 0)
    valid_mask = (pred_a >= 0) & (pred_b >= 0)
    pa = pred_a[valid_mask]
    pb = pred_b[valid_mask]

    labels_a = np.unique(pa)
    labels_b = np.unique(pb)
    
    if len(labels_a) == 0 or len(labels_b) == 0:
        return np.array([]), {}

    # 构建重叠混淆矩阵
    # cm[i, j] 表示 pred_a == labels_a[i] 且 pred_b == labels_b[j] 的节点数
    cm = confusion_matrix(pa, pb, labels=labels_a)
    
    sizes_a = np.sum(cm, axis=1, keepdims=True)  # [K_a, 1]
    sizes_b = np.sum(cm, axis=0, keepdims=True)  # [1, K_b]
    
    # 计算 Dice 矩阵: 2 * |A_i ∩ B_j| / (|A_i| + |B_j|)
    denom = sizes_a + sizes_b
    dice_matrix = np.divide(2.0 * cm, denom, out=np.zeros_like(cm, dtype=float), where=denom > 0)

    # 转化为最小化 Cost 矩阵求解最大匹配
    cost_matrix = 1.0 - dice_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_dice = dice_matrix[row_ind, col_ind]
    matching_map = {int(labels_a[r]): int(labels_b[c]) for r, c in zip(row_ind, col_ind)}

    return matched_dice, matching_map


def evaluate_subject_reliability(pred_scan1: np.ndarray, pred_scan2: np.ndarray) -> dict:
    """计算单被试的 Global ARI 与 Parcel-level Dice"""
    # 忽略背景/未标记节点 (label < 0)
    mask = (pred_scan1 >= 0) & (pred_scan2 >= 0)
    p1 = pred_scan1[mask]
    p2 = pred_scan2[mask]

    # 1. Global ARI
    ari = adjusted_rand_score(p1, p2)

    # 2. Parcel-level Dice (Hungarian Optimal Match)
    dice_arr, _ = compute_parcel_dice_hungarian(pred_scan1, pred_scan2)

    return {
        "global_ari": float(ari),
        "mean_dice": float(np.mean(dice_arr)) if len(dice_arr) > 0 else 0.0,
        "median_dice": float(np.median(dice_arr)) if len(dice_arr) > 0 else 0.0,
        "std_dice": float(np.std(dice_arr)) if len(dice_arr) > 0 else 0.0,
        "min_dice": float(np.min(dice_arr)) if len(dice_arr) > 0 else 0.0,
        "max_dice": float(np.max(dice_arr)) if len(dice_arr) > 0 else 0.0,
        "all_parcel_dice": dice_arr.tolist(),
    }


# ======================================================================
# 2. 主流程与可视化
# ======================================================================

def run_scan_rescan_evaluation(result_root: str,
                               atlas_key: str = "cluster_pred_raw") -> tuple:
    result_path = Path(result_root)
    
    # 【对齐关键】使用 BrainSubjectDataset 提取官方有效被试列表，而不是直接 iterdir()
    brain_dataset = get_dataset("brain")
    valid_subjects = [brain_dataset[i]["subject"] for i in range(len(brain_dataset))]
    
    summary_rows = []
    all_parcels_dice = []

    print("=" * 80)
    print(f"Scan-Rescan Reliability Assessment | atlas_key={atlas_key}")
    print(f"Valid Dataset Subjects: {len(valid_subjects)} ({valid_subjects[0]} -> {valid_subjects[-1]})")
    print("=" * 80)

    for sub_id in sorted(valid_subjects):
        p1_path = result_path / sub_id / "scan1" / "parcellation_result.npz"
        p2_path = result_path / sub_id / "scan2" / "parcellation_result.npz"

        if not p1_path.exists() or not p2_path.exists():
            print(f"  [SKIP] {sub_id}: 缺少 scan1 或 scan2 的 npz 文件")
            continue

        npz1 = np.load(p1_path, allow_pickle=True)
        npz2 = np.load(p2_path, allow_pickle=True)

        pred1 = np.asarray(npz1[atlas_key]) if atlas_key in npz1 else np.asarray(npz1["cluster_pred_raw"])
        pred2 = np.asarray(npz2[atlas_key]) if atlas_key in npz2 else np.asarray(npz2["cluster_pred_raw"])

        res = evaluate_subject_reliability(pred1, pred2)
        
        summary_rows.append({
            "subject": sub_id,
            "global_ari": res["global_ari"],
            "mean_dice": res["mean_dice"],
            "median_dice": res["median_dice"],
            "std_dice": res["std_dice"],
            "min_dice": res["min_dice"],
            "max_dice": res["max_dice"],
        })
        all_parcels_dice.extend(res["all_parcel_dice"])

        print(f"  - [{sub_id}] Global ARI: {res['global_ari']:.4f} | "
              f"Parcel Dice (Mean/Med): {res['mean_dice']:.4f} / {res['median_dice']:.4f}")

    return pd.DataFrame(summary_rows), all_parcels_dice


def plot_reliability_report(df_sub: pd.DataFrame, all_dice: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Subject-level Global ARI Distribution
    sns.boxplot(y=df_sub["global_ari"], ax=axes[0], color="#3498db", width=0.3)
    sns.stripplot(y=df_sub["global_ari"], ax=axes[0], color="#2980b9", size=7, jitter=0.15)
    mean_ari = df_sub["global_ari"].mean()
    med_ari = df_sub["global_ari"].median()
    axes[0].set_title(f"Global ARI across Subjects\nMean: {mean_ari:.4f} | Median: {med_ari:.4f}", fontweight="bold")
    axes[0].set_ylabel("Adjusted Rand Index (ARI)")
    axes[0].set_ylim(-0.05, 1.05)

    # Panel 2: All Parcels Dice Distribution
    sns.histplot(all_dice, ax=axes[1], bins=30, kde=True, color="#2ecc71", stat="density")
    mean_d = np.mean(all_dice)
    med_d = np.median(all_dice)
    axes[1].axvline(mean_d, color="red", linestyle="--", label=f"Mean: {mean_d:.4f}")
    axes[1].axvline(med_d, color="darkgreen", linestyle="-.", label=f"Median: {med_d:.4f}")
    axes[1].set_title(f"Parcel-level Dice Distribution (All Clusters)\nTotal Parcels: {len(all_dice)}", fontweight="bold")
    axes[1].set_xlabel(r"$Dice_k(A_s^A, A_s^B)$")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].set_xlim(-0.05, 1.05)

    fig.tight_layout()
    fig_path = out_dir / "scan_rescan_reliability_report.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"\n[Plot] 重测可靠性评估图表已保存至: {fig_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Scan-Rescan Reliability Assessment")
    p.add_argument("--result-root", default="results", help="results 目录路径")
    p.add_argument("--out-dir", default="results_diagnostics/scan_rescan")
    p.add_argument("--atlas-key", choices=["cluster_pred_raw", "cluster_pred_post", "cluster_pred"],
                   default="cluster_pred_raw", help="采用哪个级别的分割结果计算重测率")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    
    df_sub, all_dice = run_scan_rescan_evaluation(args.result_root, atlas_key=args.atlas_key)
    
    if df_sub.empty:
        print("[ERROR] 未查找到可评估的被试 Scan1/Scan2 数据，程序退出。")
        return

    # 导出 CSV / JSON 报告
    df_sub.to_csv(out_dir / "scan_rescan_subject_metrics.csv", index=False)
    
    report = {
        "n_subjects": int(len(df_sub)),
        "global_ari_mean": float(df_sub["global_ari"].mean()),
        "global_ari_median": float(df_sub["global_ari"].median()),
        "global_ari_std": float(df_sub["global_ari"].std()),
        "parcel_dice_mean": float(np.mean(all_dice)),
        "parcel_dice_median": float(np.median(all_dice)),
        "parcel_dice_std": float(np.std(all_dice)),
    }
    
    with open(out_dir / "scan_rescan_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    plot_reliability_report(df_sub, all_dice, out_dir)

    print("\n" + "=" * 80)
    print("             SCAN-RESCAN RELIABILITY SUMMARY REPORT")
    print("=" * 80)
    print(f"  • Evaluated Subjects    : {report['n_subjects']}")
    print(f"  • Global ARI (Mean±Std) : {report['global_ari_mean']:.4f} ± {report['global_ari_std']:.4f}")
    print(f"  • Global ARI (Median)   : {report['global_ari_median']:.4f}")
    print(f"  • Parcel Dice (Mean±Std): {report['parcel_dice_mean']:.4f} ± {report['parcel_dice_std']:.4f}")
    print(f"  • Parcel Dice (Median)  : {report['parcel_dice_median']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()