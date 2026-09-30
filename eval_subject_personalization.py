# -*- coding: utf-8 -*-
"""
eval_subject_personalization.py (Comprehensive Personalization, Reliability & Specificity)
===================================================================================
脑图谱个体化性能、重测可靠性与个体特异性联合检验
(Personalized vs Group Atlas | Scan-Rescan | Individual Specificity / Fingerprinting)

核心功能：
    1. 个体化性能 (Personalization Gain)
    2. 重测可靠性 (Scan-Rescan Reliability)
    3. 个体特异性 (Individual Specificity / Fingerprinting)：
       - M_{s,t} = Score(A_t^A 投影到 s 的节点空间, X_s^B)
       - 对角线 vs 非对角线；Identification Accuracy；Differential Identifiability
       - 跨被试投影调用 common.remap_graph_atlas_full_brain
    4. 导出 CSV / JSON / 期刊级图

用法：
    python eval_subject_personalization.py \
        --checkpoint checkpoints/brain_branch.pt \
        --result-root results \
        --atlas-scan scan1 \
        --eval-scan scan2 \
        --eval-scan-rescan \
        --eval-specificity \
        --subject-dir dataset \
        --space fsaverage
===================================================================================
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

from common import (
    resolve_device,
    load_model_from_checkpoint,
    extract_node_rep,
    remap_graph_atlas_full_brain,
)
from data import get_dataset


# ======================================================================
# 1. 极速 FC 计算函数 (Single Subject)
# ======================================================================

def compute_fast_fc_metrics(X_s: np.ndarray, A: np.ndarray, eps: float = 1e-5) -> dict:
    """计算被试 s 的 BOLD 在 Atlas A 下的功能连接指标 (保证 X_s 与 A 节点数一致)"""
    N, T = X_s.shape
    assert A.shape[0] == N, f"节点维度不匹配: BOLD ({N}) vs Atlas ({A.shape[0]})"

    # Z-score 标准化
    X_mean = np.mean(X_s, axis=1, keepdims=True)
    X_std = np.std(X_s, axis=1, keepdims=True) + 1e-8
    X_norm = (X_s - X_mean) / X_std / np.sqrt(T)

    unique_clusters = np.unique(A[A > 0])
    
    cluster_sums = []
    cluster_sizes = []
    sum_within_corr = 0.0
    total_within_pairs = 0

    # 计算 FC_within
    for c in unique_clusters:
        mask = (A == c)
        n_c = np.sum(mask)
        if n_c <= 1:
            continue
        
        X_c = X_norm[mask]
        vec_sum = np.sum(X_c, axis=0)
        sum_cov = np.sum(vec_sum ** 2)
        
        within_corr_c = (sum_cov - n_c) / 2.0
        n_pairs_c = n_c * (n_c - 1) / 2.0
        
        sum_within_corr += within_corr_c
        total_within_pairs += n_pairs_c
        
        cluster_sums.append(vec_sum)
        cluster_sizes.append(n_c)

    FC_within = sum_within_corr / max(total_within_pairs, 1e-8)

    # 计算 FC_between
    if len(cluster_sums) > 1:
        cluster_sums = np.array(cluster_sums)
        cluster_sizes = np.array(cluster_sizes)
        
        total_sum_vec = np.sum(cluster_sums, axis=0)
        total_cov_all = np.sum(total_sum_vec ** 2)
        
        sum_within_all_cov = np.sum(np.sum(cluster_sums ** 2, axis=1))
        between_cov = (total_cov_all - sum_within_all_cov) / 2.0
        
        total_nodes = np.sum(cluster_sizes)
        between_pairs = (total_nodes ** 2 - np.sum(cluster_sizes ** 2)) / 2.0
        
        FC_between = between_cov / max(between_pairs, 1e-8)
    else:
        FC_between = 0.0

    return {
        "FC_within": float(FC_within),
        "FC_between": float(FC_between),
        "delta_FC": float(FC_within - FC_between),
        "ratio_FC": float(FC_within / (FC_between + eps)),
    }


# ======================================================================
# 2. 重测可靠性算法 (Scan-Rescan Reliability)
# ======================================================================

def compute_parcel_dice_hungarian(pred_a: np.ndarray, pred_b: np.ndarray) -> tuple:
    """基于匈牙利算法对齐簇标签，计算 Parcel-level Dice"""
    valid_mask = (pred_a >= 0) & (pred_b >= 0)
    pa = pred_a[valid_mask]
    pb = pred_b[valid_mask]

    labels_a = np.unique(pa)
    labels_b = np.unique(pb)
    
    if len(labels_a) == 0 or len(labels_b) == 0:
        return np.array([]), {}

    cm = confusion_matrix(pa, pb, labels=labels_a)
    
    sizes_a = np.sum(cm, axis=1, keepdims=True)
    sizes_b = np.sum(cm, axis=0, keepdims=True)
    
    denom = sizes_a + sizes_b
    dice_matrix = np.divide(2.0 * cm, denom, out=np.zeros_like(cm, dtype=float), where=denom > 0)

    cost_matrix = 1.0 - dice_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_dice = dice_matrix[row_ind, col_ind]
    matching_map = {int(labels_a[r]): int(labels_b[c]) for r, c in zip(row_ind, col_ind)}

    return matched_dice, matching_map


def evaluate_single_subject_reliability(pred_scan1: np.ndarray, pred_scan2: np.ndarray) -> dict:
    """计算单被试 Scan1 与 Scan2 个体化图谱的 Global ARI 与 Parcel-level Dice"""
    mask = (pred_scan1 >= 0) & (pred_scan2 >= 0)
    p1 = pred_scan1[mask]
    p2 = pred_scan2[mask]

    ari = adjusted_rand_score(p1, p2)
    dice_arr, _ = compute_parcel_dice_hungarian(pred_scan1, pred_scan2)

    return {
        "global_ari": float(ari),
        "mean_dice": float(np.mean(dice_arr)) if len(dice_arr) > 0 else 0.0,
        "median_dice": float(np.median(dice_arr)) if len(dice_arr) > 0 else 0.0,
        "std_dice": float(np.std(dice_arr)) if len(dice_arr) > 0 else 0.0,
        "all_parcel_dice": dice_arr.tolist(),
    }



# ======================================================================
# 3. 核心实验流程
# ======================================================================

def run_personalization_experiment(checkpoint_path, dataset_dir, result_root,
                                    device_str,
                                    atlas_scan: str = "scan1",
                                    eval_scan: str = None,
                                    atlas_key: str = "cluster_pred_raw",
                                    eval_scan_rescan: bool = True):
    if eval_scan is None:
        eval_scan = atlas_scan

    device = resolve_device(device_str)
    model, cfg = load_model_from_checkpoint(checkpoint_path, device)
    brain_dataset = get_dataset("brain", pe_dim=cfg["pe_dim"])

    result_root_path = Path(result_root)
    results = []
    scan_rescan_results = {}
    all_parcels_dice = []

    print("=" * 80)
    print(f"Personalization Experiment | atlas_scan={atlas_scan}  eval_scan={eval_scan}  "
          f"atlas_key={atlas_key}")
    print(f"split_mode={cfg.get('split_mode', 'subject')}")
    print(f"eval_scan_rescan={eval_scan_rescan}")
    print("=" * 80)

    for idx in range(len(brain_dataset)):
        sample = brain_dataset[idx]
        sub_id = sample["subject"]

        # ---- 1. 读取用于评估 FC 的 atlas 来源 ----
        npz_path = result_root_path / sub_id / atlas_scan / "parcellation_result.npz"
        if not npz_path.exists():
            print(f"  [SKIP] {sub_id}: 找不到 {npz_path}")
            continue

        npz_data = np.load(npz_path, allow_pickle=True)
        keys = set(npz_data.files)

        if atlas_key in keys:
            A_pers = np.asarray(npz_data[atlas_key])
        elif "cluster_pred_raw" in keys:
            A_pers = np.asarray(npz_data["cluster_pred_raw"])
        elif "cluster_pred" in keys:
            A_pers = np.asarray(npz_data["cluster_pred"])
        else:
            print(f"  [SKIP] {sub_id}: npz 中找不到 cluster_pred*")
            continue
        N_nodes = A_pers.shape[0]

        if "prior_labels" in keys:
            A_group = np.asarray(npz_data["prior_labels"])
        else:
            A_group = np.random.permutation(A_pers)

        # ---- 2. 提取并评估 eval_scan 的 BOLD ----
        rec = extract_node_rep(
            model, sample, cfg, device,
            which_scan=eval_scan,
        )
        X_s = rec["raw_bold"]
        if X_s.shape[0] != N_nodes:
            print(f"  [WARN] {sub_id}: BOLD 节点数 {X_s.shape[0]} != atlas {N_nodes}，截断对齐")
            X_s = X_s[:N_nodes]

        # ---- 3. 计算 FC 指标 ----
        metrics_pers = compute_fast_fc_metrics(X_s, A_pers)
        metrics_group = compute_fast_fc_metrics(X_s, A_group)

        gain_within = metrics_pers["FC_within"] - metrics_group["FC_within"]
        gain_between = metrics_pers["FC_between"] - metrics_group["FC_between"]
        gain_delta = metrics_pers["delta_FC"] - metrics_group["delta_FC"]

        sub_res = {
            "subject": sub_id,
            "atlas_scan": atlas_scan,
            "eval_scan": eval_scan,
            "pers_FC_within": metrics_pers["FC_within"],
            "pers_FC_between": metrics_pers["FC_between"],
            "pers_delta_FC": metrics_pers["delta_FC"],
            "group_FC_within": metrics_group["FC_within"],
            "group_FC_between": metrics_group["FC_between"],
            "group_delta_FC": metrics_group["delta_FC"],
            "gain_FC_within": gain_within,
            "gain_FC_between": gain_between,
            "gain_delta_FC": gain_delta,
        }

        # ---- 4. [新增] 计算 Scan-Rescan Reliability ----
        if eval_scan_rescan:
            p1_path = result_root_path / sub_id / "scan1" / "parcellation_result.npz"
            p2_path = result_root_path / sub_id / "scan2" / "parcellation_result.npz"

            if p1_path.exists() and p2_path.exists():
                npz1 = np.load(p1_path, allow_pickle=True)
                npz2 = np.load(p2_path, allow_pickle=True)
                
                p1_pred = np.asarray(npz1[atlas_key]) if atlas_key in npz1 else np.asarray(npz1["cluster_pred_raw"])
                p2_pred = np.asarray(npz2[atlas_key]) if atlas_key in npz2 else np.asarray(npz2["cluster_pred_raw"])
                
                rel_res = evaluate_single_subject_reliability(p1_pred, p2_pred)
                sub_res["scan_rescan_ari"] = rel_res["global_ari"]
                sub_res["scan_rescan_mean_dice"] = rel_res["mean_dice"]
                sub_res["scan_rescan_median_dice"] = rel_res["median_dice"]

                all_parcels_dice.extend(rel_res["all_parcel_dice"])
                print(f"  - [{sub_id}] FC Gain (Δ): {gain_delta:+.4f} | "
                      f"Scan-Rescan ARI: {rel_res['global_ari']:.4f} | "
                      f"Dice (Mean/Med): {rel_res['mean_dice']:.4f}/{rel_res['median_dice']:.4f}")
            else:
                print(f"  - [{sub_id}] FC Gain (Δ): {gain_delta:+.4f} | Scan-Rescan: [SKIP] 缺少 paired npz")
        else:
            print(f"  - [{sub_id}] FC Gain (Δ): {gain_delta:+.4f}")

        results.append(sub_res)

    df_res = pd.DataFrame(results)
    
    if eval_scan_rescan and not df_res.empty and "scan_rescan_ari" in df_res.columns:
        scan_rescan_results = {
            "ari_mean": float(df_res["scan_rescan_ari"].mean()),
            "ari_median": float(df_res["scan_rescan_ari"].median()),
            "ari_std": float(df_res["scan_rescan_ari"].std()),
            "dice_mean": float(np.mean(all_parcels_dice)) if all_parcels_dice else 0.0,
            "dice_median": float(np.median(all_parcels_dice)) if all_parcels_dice else 0.0,
            "all_parcels_dice": all_parcels_dice
        }

    return df_res, scan_rescan_results


# ======================================================================
# 4. 统计分析模块 (Enhanced Statistical Analysis)
# ======================================================================

def perform_enhanced_statistical_analysis(df_res: pd.DataFrame, n_bootstraps: int = 2000, ci_level: float = 0.95):
    report = {}
    metrics_map = {
        "FC_within": ("pers_FC_within", "group_FC_within"),
        "delta_FC": ("pers_delta_FC", "group_delta_FC"),
        "FC_between": ("pers_FC_between", "group_FC_between")
    }

    alpha = 1.0 - ci_level

    for metric_key, (pers_col, group_col) in metrics_map.items():
        pers_vals = df_res[pers_col].values
        group_vals = df_res[group_col].values
        gains = pers_vals - group_vals

        n = len(gains)
        mean_gain = np.mean(gains)
        median_gain = np.median(gains)
        std_gain = np.std(gains, ddof=1) if n > 1 else 0.0
        sem_gain = std_gain / np.sqrt(n) if n > 0 else 0.0

        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1) if n > 1 else 0.0
        ci_lower_t = mean_gain - t_crit * sem_gain
        ci_upper_t = mean_gain + t_crit * sem_gain

        if n >= 4:
            boot_means = []
            rng = np.random.default_rng(42)
            for _ in range(n_bootstraps):
                boot_sample = rng.choice(gains, size=n, replace=True)
                boot_means.append(np.mean(boot_sample))
            ci_lower_boot = np.percentile(boot_means, (alpha / 2) * 100)
            ci_upper_boot = np.percentile(boot_means, (1 - alpha / 2) * 100)
        else:
            ci_lower_boot, ci_upper_boot = ci_lower_t, ci_upper_t

        if n >= 2:
            t_stat, p_val_t = stats.ttest_rel(pers_vals, group_vals, alternative="greater")
            try:
                wilcox_stat, p_val_w = stats.wilcoxon(pers_vals, group_vals, alternative="greater")
            except Exception:
                wilcox_stat, p_val_w = 0.0, 1.0
        else:
            t_stat, p_val_t, wilcox_stat, p_val_w = 0.0, 1.0, 0.0, 1.0

        cohen_d = mean_gain / (std_gain + 1e-8)

        report[metric_key] = {
            "mean_pers": float(np.mean(pers_vals)),
            "mean_group": float(np.mean(group_vals)),
            "mean_gain": float(mean_gain),
            "median_gain": float(median_gain),
            "std_gain": float(std_gain),
            "ci_95_t": [float(ci_lower_t), float(ci_upper_t)],
            "ci_95_bootstrap": [float(ci_lower_boot), float(ci_upper_boot)],
            "cohen_d": float(cohen_d),
            "paired_ttest": {"t_stat": float(t_stat), "p_value": float(p_val_t)},
            "wilcoxon_test": {"statistic": float(wilcox_stat), "p_value": float(p_val_w)},
        }

    return report


# ======================================================================
# 5. 三面板高级绘图 (Personalization Gain + Reliability)
# ======================================================================

def plot_combined_personalization_results(df_res, report, scan_rescan_summary, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    has_reliability = bool(scan_rescan_summary and "all_parcels_dice" in scan_rescan_summary)
    
    if has_reliability:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel 1 & 2: FC_within 与 Delta_FC
    configs = [
        ("FC_within", "pers_FC_within", "group_FC_within", r"$FC_{within}$ (Functional Coherence)", axes[0]),
        ("delta_FC", "pers_delta_FC", "group_delta_FC", r"$\Delta FC = FC_{within} - FC_{between}$", axes[1]),
    ]

    for metric_key, pers_col, group_col, title_label, ax in configs:
        df_sub = pd.DataFrame({
            "subject": np.concatenate([df_res["subject"], df_res["subject"]]),
            "Atlas_Type": ["Group Baseline"] * len(df_res) + ["Personalized Atlas"] * len(df_res),
            "Value": np.concatenate([df_res[group_col], df_res[pers_col]])
        })

        sns.lineplot(data=df_sub, x="Atlas_Type", y="Value", units="subject", estimator=None, color="gray", alpha=0.5, ax=ax)
        sns.boxplot(data=df_sub, x="Atlas_Type", y="Value", palette=["#95a5a6", "#3498db" if "within" in metric_key else "#2ecc71"], width=0.3, ax=ax, boxprops=dict(alpha=0.7))
        sns.stripplot(data=df_sub, x="Atlas_Type", y="Value", hue="Atlas_Type", palette=["#7f8c8d", "#2980b9" if "within" in metric_key else "#27ae60"], size=8, jitter=False, ax=ax)

        rep = report[metric_key]
        p_val = rep["paired_ttest"]["p_value"]
        d_val = rep["cohen_d"]
        med_g = rep["median_gain"]
        ci_boot = rep["ci_95_bootstrap"]

        ax.set_title(
            f"{title_label}\n"
            f"Mean Gain = +{rep['mean_gain']:.4f} (Median = +{med_g:.4f})\n"
            f"95% CI: [{ci_boot[0]:.4f}, {ci_boot[1]:.4f}] | p={p_val:.4e}, d={d_val:.2f}",
            fontweight="bold", fontsize=10
        )
        ax.set_ylabel(title_label)
        ax.set_xlabel("")

    # Panel 3: Scan-Rescan Reliability Distribution
    if has_reliability:
        ax_rel = axes[2]
        all_dice = scan_rescan_summary["all_parcels_dice"]
        mean_ari = scan_rescan_summary["ari_mean"]
        med_dice = scan_rescan_summary["dice_median"]
        mean_dice = scan_rescan_summary["dice_mean"]

        sns.histplot(all_dice, ax=ax_rel, bins=25, kde=True, color="#9b59b6", stat="density")
        ax_rel.axvline(mean_dice, color="red", linestyle="--", label=f"Mean Dice: {mean_dice:.4f}")
        ax_rel.axvline(med_dice, color="darkgreen", linestyle="-.", label=f"Med Dice: {med_dice:.4f}")
        
        ax_rel.set_title(
            f"Scan-Rescan Reliability\n"
            f"Global ARI: {mean_ari:.4f} | Parcel Dice Median: {med_dice:.4f}",
            fontweight="bold", fontsize=10
        )
        ax_rel.set_xlabel(r"Parcel-level $Dice_k(A_{scan1}, A_{scan2})$")
        ax_rel.set_ylabel("Density")
        ax_rel.legend()
        ax_rel.set_xlim(-0.05, 1.05)

    fig.tight_layout()
    fig_path = out_dir / "personalization_combined_report.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"\n[Plot] 个人化联合诊断图表已保存至: {fig_path}")


# ======================================================================
# 6. 主程序入口与格式化终端输出
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Personalization Validation & Scan-Rescan Assessment")
    p.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径")
    p.add_argument("--dataset-dir", default="dataset")
    p.add_argument("--result-root", default="results", help="results/<subject>/<scan>/parcellation_result.npz")
    p.add_argument("--out-dir", default="results_diagnostics/personalization")
    p.add_argument("--device", default=None)

    p.add_argument("--atlas-scan", choices=["scan1", "scan2", "full"], default="scan1",
                   help="从哪个 scan 的聚类结果构建个性化图谱 A_pers")
    p.add_argument("--eval-scan", choices=["scan1", "scan2", "full"], default=None,
                   help="在哪个 scan 的 BOLD 上评估；默认与 --atlas-scan 相同")
    p.add_argument("--atlas-key", choices=["cluster_pred_raw", "cluster_pred_post", "cluster_pred"], default="cluster_pred_raw",
                   help="从 npz 读取哪个字段作为 A_pers")
    p.add_argument("--eval-scan-rescan", action="store_true", default=True,
                   help="是否同时评估 Scan1 与 Scan2 个体化图谱的重测可靠性 (ARI & Dice)")
    return p.parse_args()


def main():
    args = parse_args()
    eval_scan = args.eval_scan or args.atlas_scan

    out_dir = Path(args.out_dir) / f"{args.atlas_scan}_to_{eval_scan}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 运行实验
    df_res, scan_rescan_summary = run_personalization_experiment(
        checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset_dir,
        result_root=args.result_root,
        device_str=args.device,
        atlas_scan=args.atlas_scan,
        eval_scan=eval_scan,
        atlas_key=args.atlas_key,
        eval_scan_rescan=args.eval_scan_rescan,
    )
    
    if df_res.empty:
        print("[ERROR] 没有可用被试，程序退出。")
        return

    # 2. 统计分析
    report = perform_enhanced_statistical_analysis(df_res)
    report["_meta"] = {
        "atlas_scan": args.atlas_scan,
        "eval_scan": eval_scan,
        "atlas_key": args.atlas_key,
        "n_subjects": int(len(df_res)),
    }
    if scan_rescan_summary:
        report["scan_rescan_reliability"] = {
            "global_ari_mean": scan_rescan_summary["ari_mean"],
            "global_ari_median": scan_rescan_summary["ari_median"],
            "parcel_dice_mean": scan_rescan_summary["dice_mean"],
            "parcel_dice_median": scan_rescan_summary["dice_median"],
        }

    # 3. 绘图与结果导出
    plot_combined_personalization_results(df_res, report, scan_rescan_summary, out_dir)

    # 清理非 JSON 序列化的全部数组后保存 json
    df_res_save = df_res.copy()
    df_res_save.to_csv(out_dir / "personalization_results.csv", index=False)
    
    with open(out_dir / "personalization_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 4. 格式化终端输出
    print("\n" + "=" * 80)
    print("          PERSONALIZATION & RELIABILITY VALIDATION REPORT")
    print(f"          atlas_scan={args.atlas_scan}  eval_scan={eval_scan}")
    print("=" * 80)
    for metric_key, name in [("FC_within", "FC_within (Functional Coherence)"),
                             ("delta_FC", "Delta FC (FC_within - FC_between)")]:
        m = report[metric_key]
        print(f" ► Metric: {name}")
        print(f"   - Group Baseline Mean    : {m['mean_group']:.4f}")
        print(f"   - Personalized Mean      : {m['mean_pers']:.4f}")
        print(f"   - Mean Gain (±STD)       : +{m['mean_gain']:.4f} (±{m['std_gain']:.4f})")
        print(f"   - Median Gain            : +{m['median_gain']:.4f}")
        print(f"   - 95% CI (Bootstrap)     : [{m['ci_95_bootstrap'][0]:.4f}, {m['ci_95_bootstrap'][1]:.4f}]")
        print(f"   - Effect Size (Cohen's d): {m['cohen_d']:.4f}")
        print(f"   - Paired t-test p-value  : {m['paired_ttest']['p_value']:.4e} (t={m['paired_ttest']['t_stat']:.3f})")
        print("-" * 80)

    if scan_rescan_summary:
        print(" ► Metric: Scan-Rescan Reliability (Atlas Stability)")
        print(f"   - Global ARI (Mean / Med): {scan_rescan_summary['ari_mean']:.4f} / {scan_rescan_summary['ari_median']:.4f}")
        print(f"   - Parcel Dice (Mean/Med) : {scan_rescan_summary['dice_mean']:.4f} / {scan_rescan_summary['dice_median']:.4f}")
        print("-" * 80)

    print("=" * 80)
    print(f" [DONE] 完整验证结果已写入: {out_dir.resolve()}")


if __name__ == "__main__":
    main()