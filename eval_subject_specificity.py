# -*- coding: utf-8 -*-
"""
eval_subject_specificity.py
===================================================================================
个体化图谱的个体特异性 / 指纹识别检验 (Subject-Specificity / Fingerprinting)
===================================================================================

两个并列实验（arm-1 完整矩阵 + arm-2 单行探针）：

  ▶ ARM-1  Atlas-Transfer (完整 S×S 矩阵, 原实验)
      M[s, t] = Score( remap(A_t^A → s), X_s^B )
      问题：别人的图谱 A_t 是否能适配我自己的 scan 数据 X_s？
      期望：对角线 M[s, s] 显著高于非对角线。

  ▶ ARM-2  Repr-Swap single-row (probe 单行, 廉价版 shuffle null)
      仅对探针被试 s*，计算
        M_shuf[s*, t] = Score( Cluster( G_{s*}, project(Z_t^A → s*) ), X_{s*}^B )
      问题：如果只保留 t 的“表示”，在 s* 的解剖图上重新聚类，
            是否还能重建出一个适配 s* 自己的图谱？
      期望：对角线 ≈ 非对角线（null 假设下）。

  ▶ ARM1 − ARM2 (diag contrast)
      同一探针 s*，比较两条路径下非对角线项的差异：
      “图谱几何”比“仅表示”多贡献了多少身份信号？

统计模块：
    - 完整矩阵：diag vs off-mean / off-max paired t、permutation、bootstrap CI
    - 单行：rank-based 精确置换 p (= rank / S)、符号检验
    - 跨臂：one-sample t on (M_atlas − M_shuf) 非对角线项

投影质量 (ARI round-trip)：s →(sphere→s)→ ŝ 的 ARI / exact-match / Dice
    用于量化球面 NN 投影引入的几何噪声上界。

用法：
    # 只跑 arm-1（向后兼容）
    python eval_subject_specificity.py \
        --checkpoint checkpoints/brain_branch.pt \
        --dataset-dir dataset --subject-dir-root /path/to/fmriprep \
        --result-root results --atlas-scan scan1 --eval-scan scan2

    # 两个 arm 一起跑
    python eval_subject_specificity.py ... --run-repr-swap \
        --shuffle-mode knn_gaussian --shuffle-k 4
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
    project_node_features_between_subjects,
)
from data import get_dataset


# ======================================================================
# 1. FC 工具（与 eval_subject_personalization 数学等价，但接收预归一化 BOLD）
# ======================================================================

def normalize_bold(X_s: np.ndarray) -> np.ndarray:
    """与 eval_subject_personalization.compute_fast_fc_metrics 的 z-score 一致。"""
    X_mean = np.mean(X_s, axis=1, keepdims=True)
    X_std = np.std(X_s, axis=1, keepdims=True) + 1e-8
    return (X_s - X_mean) / X_std / np.sqrt(X_s.shape[1])


def compute_fc_from_norm(X_norm: np.ndarray, A: np.ndarray, eps: float = 1e-5) -> dict:
    """
    基于已归一化 BOLD 计算 FC_within / FC_between / delta_FC。

    约定：A 中标签 >= 0 视为有效簇（0-based 或 1-based 均可），
          未映射节点用 -1 表示（与 remap_graph_atlas_full_brain(fill_value=-1) 一致）。
    """
    N = X_norm.shape[0]
    assert A.shape[0] == N, f"节点维度不匹配: BOLD ({N}) vs Atlas ({A.shape[0]})"

    valid_A = A[A >= 0]
    if valid_A.size == 0:
        return {"FC_within": 0.0, "FC_between": 0.0, "delta_FC": 0.0, "ratio_FC": 0.0}

    unique_clusters = np.unique(valid_A)

    cluster_sums = []
    cluster_sizes = []
    sum_within_corr = 0.0
    total_within_pairs = 0

    for c in unique_clusters:
        mask = (A == c)
        n_c = int(np.sum(mask))
        if n_c <= 1:
            continue
        X_c = X_norm[mask]
        vec_sum = np.sum(X_c, axis=0)
        sum_cov = float(np.sum(vec_sum ** 2))
        within_corr_c = (sum_cov - n_c) / 2.0
        n_pairs_c = n_c * (n_c - 1) / 2.0

        sum_within_corr += within_corr_c
        total_within_pairs += n_pairs_c
        cluster_sums.append(vec_sum)
        cluster_sizes.append(n_c)

    FC_within = sum_within_corr / max(total_within_pairs, 1e-8)

    if len(cluster_sums) > 1:
        cluster_sums = np.array(cluster_sums)
        cluster_sizes = np.array(cluster_sizes)
        total_sum_vec = np.sum(cluster_sums, axis=0)
        total_cov_all = float(np.sum(total_sum_vec ** 2))
        sum_within_all_cov = float(np.sum(np.sum(cluster_sums ** 2, axis=1)))
        between_cov = (total_cov_all - sum_within_all_cov) / 2.0
        total_nodes = int(np.sum(cluster_sizes))
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
# 2. 数据加载辅助
# ======================================================================

_FALLBACK_KEYS = ("cluster_pred_raw", "cluster_pred", "cluster_pred_post")


def load_atlas_npz(path: Path, atlas_key: str) -> np.ndarray:
    npz = np.load(path, allow_pickle=True)
    keys = set(npz.files)
    if atlas_key in keys:
        return np.asarray(npz[atlas_key])
    for k in _FALLBACK_KEYS:
        if k in keys:
            return np.asarray(npz[k])
    raise KeyError(f"{path} 中找不到 atlas 键 {atlas_key} (可用: {npz.files})")


def precompute_bold_normalized(model, dataset, cfg, device, which_scan: str) -> dict:
    """对全部 subject 预计算 scan 的归一化 BOLD。"""
    print(f"[Precompute] 提取 {which_scan} 的归一化 BOLD ...")
    X_dict = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        rec = extract_node_rep(model, sample, cfg, device, which_scan=which_scan)
        X_dict[sample["subject"]] = normalize_bold(rec["raw_bold"])
    print(f"[Precompute] 完成，共 {len(X_dict)} 个 subject。")
    return X_dict


# ======================================================================
# 3. ARM-1：Atlas-Transfer 完整矩阵
# ======================================================================

def build_atlas_transfer_matrix(subjects, X_norm_dict, atlas_dict, nodes_paths,
                                subject_dir_root: Path, sphere_space: str,
                                verbose: bool = True):
    """
    M[s,t] = Score( remap(A_t^A → s space), X_s^B )。

    对 t == s 直接复用原生 A_s，不经过球面重映射（避免几何误差污染对角线）。
    """
    S = len(subjects)
    M_delta = np.zeros((S, S), dtype=float)
    M_within = np.zeros((S, S), dtype=float)
    M_between = np.zeros((S, S), dtype=float)

    remap_cache = {}
    print(f"[Arm-1|Atlas-Transfer] {S}x{S} 打分（含 {S*(S-1)} 次重映射）...")

    for i, s in enumerate(subjects):
        X_s = X_norm_dict[s]
        for j, t in enumerate(subjects):
            key = (t, s)
            if key not in remap_cache:
                if t == s:
                    remap_cache[key] = atlas_dict[t]
                else:
                    remap_cache[key] = remap_graph_atlas_full_brain(
                        labels_nodes_src   = atlas_dict[t],
                        src_nodes_npz_path = nodes_paths[t],
                        src_subject_dir    = subject_dir_root, src_subject = t,
                        tgt_nodes_npz_path = nodes_paths[s],
                        tgt_subject_dir    = subject_dir_root, tgt_subject = s,
                        space              = sphere_space,
                        offset_src=0, offset_tgt=0,
                        mode="nearest", fill_value=-1,
                    )
            A_t_in_s = remap_cache[key]
            n = min(X_s.shape[0], A_t_in_s.shape[0])
            m = compute_fc_from_norm(X_s[:n], A_t_in_s[:n])
            M_delta[i, j]   = m["delta_FC"]
            M_within[i, j]  = m["FC_within"]
            M_between[i, j] = m["FC_between"]
        if verbose and ((i + 1) % max(1, S // 10) == 0 or (i + 1) == S):
            print(f"  row {i+1}/{S} 完成")

    return M_delta, M_within, M_between


# ======================================================================
# 3b. ARM-2：Repr-Swap 单探针行
# ======================================================================

def build_repr_swap_row(probe_s, subjects, X_norm_dict, Z_A_dict,
                        adj_dict, prior_dict, nodes_paths, atlas_dict,
                        subject_dir_root: Path, sphere_space: str,
                        shuffle_mode: str = "knn_gaussian",
                        shuffle_k: int = 4,
                        shuffle_sigma=None,
                        verbose: bool = True):
    """
    计算 M_shuf[probe_s, :] 一行（长度 S），按 subjects 顺序返回。

    条目定义（j 为 donor t 在 subjects 中的索引）：
        j == s_idx : A_shuf = atlas_dict[probe_s]                            （复用，代价 0）
        j != s_idx : A_shuf = Cluster( G_{s*}, project(Z_t^A → s*) )         （1 次聚类）

    返回 (M_delta_row, M_within_row, M_between_row)。
    """
    from clustering import run_mixed_spectral_clustering  # 延迟 import，避免循环依赖

    assert probe_s in subjects, f"{probe_s} 不在 subjects 中"
    s_idx = subjects.index(probe_s)

    X_s = X_norm_dict[probe_s]
    n_s = X_s.shape[0]
    S = len(subjects)

    M_delta  = np.zeros(S, dtype=float)
    M_within = np.zeros(S, dtype=float)
    M_between = np.zeros(S, dtype=float)

    print(f"[Arm-2|Repr-Swap row] probe={probe_s}  将执行 {S-1} 次聚类 "
          f"(mode={shuffle_mode}, k={shuffle_k})")

    for j, t in enumerate(subjects):
        if t == probe_s:
            A_shuf = np.asarray(atlas_dict[probe_s])
            tag = "(diag, reuse)"
        else:
            Z_t_to_s = project_node_features_between_subjects(
                features_src        = Z_A_dict[t],
                src_nodes_npz_path  = nodes_paths[t],
                src_subject_dir     = subject_dir_root, src_subject = t,
                tgt_nodes_npz_path  = nodes_paths[probe_s],
                tgt_subject_dir     = subject_dir_root, tgt_subject = probe_s,
                space               = sphere_space,
                mode                = shuffle_mode,
                k                   = shuffle_k,
                sigma               = shuffle_sigma,
            )

            # ★ debug 检查放在 else 分支内（Z_t_to_s 在这里才有定义）
            if not np.isfinite(Z_t_to_s).all():
                n_bad = int((~np.isfinite(Z_t_to_s)).sum())
                print(f"    [WARN] t={t}: Z_t_to_s 含 {n_bad} 个 NaN/Inf，跳过该 donor")
                M_delta[j]   = np.nan
                M_within[j]  = np.nan
                M_between[j] = np.nan
                continue

            res = run_mixed_spectral_clustering(
                Z_node       = Z_t_to_s,
                adj_tensor   = adj_dict[probe_s],
                prior_labels = prior_dict[probe_s],
                post_merge   = False,
            )
            A_shuf = res.pred_raw
            tag = "(cluster)"

        n = min(n_s, A_shuf.shape[0])
        m = compute_fc_from_norm(X_s[:n], A_shuf[:n])
        M_delta[j]   = m["delta_FC"]
        M_within[j]  = m["FC_within"]
        M_between[j] = m["FC_between"]

        if verbose:
            print(f"    t={t:>12s}  ΔFC={M_delta[j]:+.4f}  {tag}")
            
    return M_delta, M_within, M_between


# ======================================================================
# 4. 完整矩阵统计分析（含 bootstrap CI + permutation）
# ======================================================================

def analyze_specificity_matrix(M: np.ndarray, subjects, metric_name: str,
                               n_perm: int = 5000) -> dict:
    S = M.shape[0]
    if S < 2:
        return {"metric": metric_name, "error": "S < 2"}

    diag = np.diag(M)
    off_mask = ~np.eye(S, dtype=bool)
    off_diag = M[off_mask]

    row_off_mean = np.array([M[i, off_mask[i]].mean() for i in range(S)])
    row_off_max  = np.array([M[i, off_mask[i]].max()  for i in range(S)])

    # ---- 指纹识别 ----
    rank_desc = (-M).argsort(axis=1).argsort(axis=1) + 1   # 1 = best
    diag_ranks = rank_desc[np.arange(S), np.arange(S)]

    topk = {}
    for k in (1, 3, 5, 10):
        if k <= S:
            topk[f"top{k}_acc"] = float((diag_ranks <= k).mean())

    # ---- paired t ----
    def paired(a, b):
        if S < 2:
            return {"t_stat": 0.0, "p_value": 1.0, "cohen_d": 0.0}
        t_stat, p_val = stats.ttest_rel(a, b, alternative="greater")
        diff = a - b
        d = float(diff.mean() / (diff.std(ddof=1) + 1e-8))
        return {"t_stat": float(t_stat), "p_value": float(p_val), "cohen_d": d}

    test_vs_off_mean = paired(diag, row_off_mean)
    test_vs_off_max  = paired(diag, row_off_max)

    # ---- bootstrap CI (subject-level resampling) ----
    rng_b = np.random.default_rng(43)
    n_boot = max(2000, n_perm)
    boot_top1 = np.empty(n_boot)
    boot_rank = np.empty(n_boot)
    boot_diag = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng_b.integers(0, S, size=S)
        boot_top1[b] = float((diag_ranks[idx] == 1).mean())
        boot_rank[b] = float(diag_ranks[idx].mean())
        boot_diag[b] = float(diag[idx].mean())

    def _ci(arr, lo=2.5, hi=97.5):
        return [float(np.percentile(arr, lo)), float(np.percentile(arr, hi))]

    bootstrap_ci = {
        "top1_acc":   _ci(boot_top1),
        "mean_rank":  _ci(boot_rank),
        "diag_mean":  _ci(boot_diag),
        "n_bootstrap": int(n_boot),
    }

    # ---- permutation test ----
    rng = np.random.default_rng(42)
    null_diag_mean = np.zeros(n_perm)
    for k in range(n_perm):
        idx = rng.integers(0, S, size=S)
        null_diag_mean[k] = M[np.arange(S), idx].mean()

    obs_diag_mean = float(diag.mean())
    p_perm = float((null_diag_mean >= obs_diag_mean).mean())

    return {
        "metric": metric_name,
        "n_subjects": int(S),
        "diag": {
            "mean":   float(diag.mean()),
            "median": float(np.median(diag)),
            "std":    float(diag.std(ddof=1)) if S > 1 else 0.0,
            "min":    float(diag.min()),
            "max":    float(diag.max()),
        },
        "off_diag": {
            "mean":   float(off_diag.mean()),
            "median": float(np.median(off_diag)),
            "std":    float(off_diag.std(ddof=1)) if off_diag.size > 1 else 0.0,
        },
        "diag_minus_off_mean": float(diag.mean() - off_diag.mean()),
        "row_off_mean_mean":   float(row_off_mean.mean()),
        "row_off_max_mean":    float(row_off_max.mean()),
        "diag_vs_row_off_mean": test_vs_off_mean,
        "diag_vs_row_off_max":  test_vs_off_max,
        "bootstrap_ci": bootstrap_ci,
        "permutation_test": {
            "obs_diag_mean":        obs_diag_mean,
            "null_mean":            float(null_diag_mean.mean()),
            "null_std":             float(null_diag_mean.std()),
            "p_value_one_sided":    p_perm,
            "n_permutations":       int(n_perm),
        },
        "identification": {
            "top1_acc":        topk.get("top1_acc", 0.0),
            "chance_top1":     1.0 / S,
            "topk_acc":        topk,
            "mean_rank":       float(diag_ranks.mean()),
            "median_rank":     float(np.median(diag_ranks)),
            "chance_mean_rank":(S + 1) / 2.0,
        },
    }


# ======================================================================
# 4b. 投影质量（ARI round-trip）
# ======================================================================

def _parcel_dice_hungarian(pred_a: np.ndarray, pred_b: np.ndarray) -> np.ndarray:
    """匈牙利对齐后的 parcel Dice 数组。"""
    valid = (pred_a >= 0) & (pred_b >= 0)
    pa, pb = pred_a[valid], pred_b[valid]
    la, lb = np.unique(pa), np.unique(pb)
    if len(la) == 0 or len(lb) == 0:
        return np.array([])
    cm = confusion_matrix(pa, pb, labels=la)
    sizes_a = cm.sum(axis=1, keepdims=True)
    sizes_b = cm.sum(axis=0, keepdims=True)
    denom = sizes_a + sizes_b
    dice_mat = np.divide(2.0 * cm, denom,
                         out=np.zeros_like(cm, dtype=float), where=denom > 0)
    r, c = linear_sum_assignment(1.0 - dice_mat)
    return dice_mat[r, c]


def compute_projection_quality(subjects, atlas_dict, nodes_paths,
                               subject_dir_root: Path, sphere_space: str) -> pd.DataFrame:
    """
    对每个 s：把 A_s 通过 (s native → sphere → s native) 往返一次，
    与原生 A_s 比较 ARI / 一致率 / Dice。

    理想 ARI = 1.0；实际值反映球面 NN 投影引入的噪声上界。
    """
    rows = []
    print("[Projection quality] s→s round-trip ...")
    for s in subjects:
        A_s = atlas_dict[s]
        A_s_rt = remap_graph_atlas_full_brain(
            labels_nodes_src   = A_s,
            src_nodes_npz_path = nodes_paths[s],
            src_subject_dir    = subject_dir_root, src_subject = s,
            tgt_nodes_npz_path = nodes_paths[s],
            tgt_subject_dir    = subject_dir_root, tgt_subject = s,
            space              = sphere_space,
            offset_src=0, offset_tgt=0,
            mode="nearest", fill_value=-1,
        )
        mask = (A_s >= 0) & (A_s_rt >= 0)
        if mask.sum() < 2:
            print(f"  [WARN] {s}: 有效节点 < 2，跳过")
            continue

        ari = float(adjusted_rand_score(A_s[mask], A_s_rt[mask]))
        exact = float((A_s[mask] == A_s_rt[mask]).mean())
        dice_arr = _parcel_dice_hungarian(A_s, A_s_rt)

        rows.append({
            "subject": s,
            "n_nodes": int(mask.sum()),
            "ari": ari,
            "exact_match_rate": exact,
            "mean_dice": float(dice_arr.mean()) if len(dice_arr) else 0.0,
            "median_dice": float(np.median(dice_arr)) if len(dice_arr) else 0.0,
        })
        print(f"  [{s}] ARI={ari:.4f}  exact={exact:.4f}  "
              f"Dice(mean/med)={rows[-1]['mean_dice']:.4f}/{rows[-1]['median_dice']:.4f}")

    return pd.DataFrame(rows)


# ======================================================================
# 4c. 单行统计（ARM-2 与跨臂对比）
# ======================================================================

def analyze_specificity_row(M_row: np.ndarray, subjects, probe_s: str,
                            metric_name: str) -> dict:
    """对 M[s*, :] 做排名 + 精确置换 p（row-internal coordinates）。"""
    S = len(subjects)
    s_idx = subjects.index(probe_s)
    diag_val = float(M_row[s_idx])
    off_vals = np.delete(M_row, s_idx)

    rank_desc = int((M_row > diag_val).sum() + 1)   # 1 = best
    perm_p = rank_desc / S

    return {
        "metric": metric_name,
        "probe_subject": probe_s,
        "n_subjects": int(S),
        "diag_value": diag_val,
        "off_mean":   float(off_vals.mean()),
        "off_median": float(np.median(off_vals)),
        "off_max":    float(off_vals.max()),
        "off_min":    float(off_vals.min()),
        "diag_minus_off_mean": float(diag_val - off_vals.mean()),
        "diag_minus_off_max":  float(diag_val - off_vals.max()),
        "diag_rank_desc": rank_desc,
        "is_top1": int(rank_desc == 1),
        "perm_p_value_one_sided": float(perm_p),
        "row_values": {s: float(v) for s, v in zip(subjects, M_row)},
    }


def analyze_atlas_vs_swap_row(M_atlas_row: np.ndarray, M_shuf_row: np.ndarray,
                              subjects, probe_s: str) -> dict:
    """
    同一探针下，臂 1 与臂 2 行的逐对差值（仅非对角线）：
      问：把 A_t^A 换成 Cluster(G_{s*}, Z_t^{A→s*}) 后，匹配分数是否下降？
    """
    s_idx = subjects.index(probe_s)
    diffs_all = M_atlas_row - M_shuf_row
    off_diffs = np.delete(diffs_all, s_idx)

    n = len(off_diffs)
    if n >= 2:
        t_stat, p_val = stats.ttest_1samp(off_diffs, 0.0, alternative="greater")
        d = float(off_diffs.mean() / (off_diffs.std(ddof=1) + 1e-8))
    else:
        t_stat, p_val, d = 0.0, 1.0, 0.0

    n_pos = int((off_diffs > 0).sum())
    if n >= 2:
        try:
            sign_p = float(stats.binomtest(n_pos, n, 0.5, alternative="greater").pvalue)
        except AttributeError:      # scipy < 1.7 兼容
            sign_p = float(stats.binom_test(n_pos, n, 0.5, alternative="greater"))
    else:
        sign_p = 1.0

    return {
        "probe_subject": probe_s,
        "n_off": int(n),
        "mean_diff_off":   float(off_diffs.mean()),
        "median_diff_off": float(np.median(off_diffs)),
        "std_diff_off":    float(off_diffs.std(ddof=1)) if n > 1 else 0.0,
        "n_atlas_above_swap": n_pos,
        "sign_test_p": sign_p,
        "one_sample_t": {"t_stat": float(t_stat), "p_value": float(p_val)},
        "cohen_d": d,
        "per_donor": {s: float(v) for s, v in zip(subjects, diffs_all)},
    }


# ======================================================================
# 5. 导出工具
# ======================================================================

def save_matrices(out_dir: Path, subjects, M_delta, M_within, M_between):
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, M in [("M_delta_FC", M_delta), ("M_FC_within", M_within), ("M_FC_between", M_between)]:
        df = pd.DataFrame(M, index=subjects, columns=subjects)
        df.index.name = "subject_s"
        df.to_csv(out_dir / f"{name}.csv")

    S = len(subjects)
    rows = []
    for i, s in enumerate(subjects):
        for j, t in enumerate(subjects):
            rows.append({
                "subject_s": s,
                "atlas_t": t,
                "is_diagonal": int(i == j),
                "delta_FC":   M_delta[i, j],
                "FC_within":  M_within[i, j],
                "FC_between": M_between[i, j],
            })
    pd.DataFrame(rows).to_csv(out_dir / "specificity_matrix_long.csv", index=False)


def build_per_subject_dataframe(subjects, M_delta, M_within, M_between):
    S = len(subjects)
    off_mask = ~np.eye(S, dtype=bool)
    rank_desc = (-M_delta).argsort(axis=1).argsort(axis=1) + 1

    rows = []
    for i, s in enumerate(subjects):
        off_vals = M_delta[i, off_mask[i]]
        rows.append({
            "subject": s,
            "diag_delta_FC": M_delta[i, i],
            "row_off_mean_delta_FC": float(off_vals.mean()),
            "row_off_max_delta_FC":  float(off_vals.max()),
            "diag_minus_off_mean":   float(M_delta[i, i] - off_vals.mean()),
            "diag_rank_desc":        int(rank_desc[i, i]),
            "is_top1":               int(rank_desc[i, i] == 1),
            "diag_FC_within":        M_within[i, i],
            "diag_FC_between":       M_between[i, i],
        })
    return pd.DataFrame(rows)


# ======================================================================
# 6. 可视化（ARM-1 完整矩阵）
# ======================================================================

def plot_specificity_results(M_delta, subjects, analysis, out_dir: Path,
                             primary_name: str = "ΔFC"):
    sns.set_theme(style="whitegrid")
    S = len(subjects)

    fig = plt.figure(figsize=(18, 10))

    # (1) 原始矩阵热力图
    ax1 = plt.subplot(2, 3, 1)
    vmin = float(np.percentile(M_delta, 5))
    vmax = float(np.percentile(M_delta, 95))
    im1 = ax1.imshow(M_delta, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax1.set_xlabel("Atlas subject $t$  (from scan A)")
    ax1.set_ylabel("Data subject $s$  (from scan B)")
    ax1.set_title(f"$M_{{s,t}} = {primary_name}(A_t^A, X_s^B)$")
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    for i in range(S):
        ax1.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                     edgecolor="red", lw=0.4))

    # (2) 行归一化
    ax2 = plt.subplot(2, 3, 2)
    M_norm = M_delta - M_delta.mean(axis=1, keepdims=True)
    vmax2 = float(np.percentile(np.abs(M_norm), 98)) + 1e-8
    im2 = ax2.imshow(M_norm, cmap="RdBu_r", vmin=-vmax2, vmax=vmax2, aspect="auto")
    ax2.set_xlabel("Atlas subject $t$")
    ax2.set_ylabel("Data subject $s$")
    ax2.set_title("Row-normalized:  $M_{s,t} - \\bar{M}_{s,\\cdot}$")
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    for i in range(S):
        ax2.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                     edgecolor="k", lw=0.4))

    # (3) diag vs off box
    ax3 = plt.subplot(2, 3, 3)
    diag = np.diag(M_delta)
    off = M_delta[~np.eye(S, dtype=bool)]
    df_box = pd.DataFrame({
        "Group": ["Diagonal\n(matched)"] * S + ["Off-diagonal\n(mismatched)"] * len(off),
        "Value": np.concatenate([diag, off]),
    })
    sns.boxplot(data=df_box, x="Group", y="Value", hue="Group",
            palette=["#e74c3c", "#95a5a6"], legend=False,
            ax=ax3, width=0.5, boxprops=dict(alpha=0.8))
    sns.stripplot(data=df_box, x="Group", y="Value", color="black",
                  size=3, alpha=0.35, jitter=0.25, ax=ax3)
    p_perm = analysis["permutation_test"]["p_value_one_sided"]
    t_p = analysis["diag_vs_row_off_mean"]["p_value"]
    d = analysis["diag_vs_row_off_mean"]["cohen_d"]
    ax3.set_title(
        f"Diagonal vs Off-diagonal ({primary_name})\n"
        f"Δ = {analysis['diag_minus_off_mean']:+.4f} | d = {d:.2f} | "
        f"paired p = {t_p:.2e}\nperm p = {p_perm:.4f}",
        fontsize=10, fontweight="bold",
    )
    ax3.set_ylabel(primary_name)
    ax3.set_xlabel("")

    # (4) rank distribution
    ax4 = plt.subplot(2, 3, 4)
    rank_desc = (-M_delta).argsort(axis=1).argsort(axis=1) + 1
    diag_ranks = rank_desc[np.arange(S), np.arange(S)]
    sns.histplot(diag_ranks, bins=np.arange(1, S + 2) - 0.5,
                 stat="probability", color="#9b59b6", ax=ax4, edgecolor="white")
    ax4.axhline(1.0 / S, color="red", linestyle="--",
                label=f"Chance = 1/S = {1.0/S:.3f}")
    ax4.set_xlabel("Rank of own atlas (1 = best)")
    ax4.set_ylabel("Probability")
    ax4.set_title(f"Rank distribution  (Mean = {diag_ranks.mean():.2f}, "
                  f"Chance = {(S+1)/2:.2f})")
    ax4.legend()

    # (5) per-subject diag vs row-off-mean
    ax5 = plt.subplot(2, 3, 5)
    row_off_mean = np.array([M_delta[i, ~np.eye(S, dtype=bool)[i]].mean() for i in range(S)])
    ax5.scatter(row_off_mean, diag, s=25, alpha=0.7, color="#e74c3c",
                edgecolor="k", linewidth=0.3)
    lo = float(min(row_off_mean.min(), diag.min()))
    hi = float(max(row_off_mean.max(), diag.max()))
    ax5.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax5.set_xlabel(f"Row off-diagonal mean {primary_name}")
    ax5.set_ylabel(f"Diagonal {primary_name}")
    n_above = int((diag > row_off_mean).sum())
    ax5.set_title(f"Per-subject:  {n_above}/{S} above y=x")
    ax5.legend()

    # (6) Top-K 曲线
    ax6 = plt.subplot(2, 3, 6)
    ks = sorted([int(k.replace("top", "").replace("_acc", ""))
                 for k in analysis["identification"]["topk_acc"].keys()])
    accs = [analysis["identification"]["topk_acc"][f"top{k}_acc"] for k in ks]
    chance = [k / S for k in ks]
    ax6.plot(ks, accs, "-o", color="#2980b9", lw=2, label="Observed Top-K")
    ax6.plot(ks, chance, "--", color="gray", lw=1.5, label="Chance")
    ax6.set_xlabel("K")
    ax6.set_ylabel("Top-K identification accuracy")
    ax6.set_ylim(0, 1.05)
    ax6.set_title(f"Fingerprint identification\n"
                  f"Top-1 = {accs[0]:.3f} (chance {chance[0]:.3f})")
    ax6.legend()

    fig.tight_layout()
    out_path = out_dir / "specificity_report.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] 保存至: {out_path}")


# ======================================================================
# 7. CLI 与主流程
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Personalized atlas subject-specificity / fingerprinting")
    p.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径")
    p.add_argument("--dataset-dir", default="dataset",
                   help="含 <subject>.pt / <subject>.nodes.npz 的目录")
    p.add_argument("--subject-dir-root", required=True,
                   help="fMRIPrep 被试根目录（含 <subject>/anat/*_sphere.surf.gii）")
    p.add_argument("--result-root", default="results",
                   help="results/<subject>/<scan>/parcellation_result.npz")
    p.add_argument("--out-dir", default="results_diagnostics/specificity")
    p.add_argument("--device", default=None)

    p.add_argument("--atlas-scan", choices=["scan1", "scan2", "full"], default="scan1",
                   help="用于构建 atlas A_s 的 scan")
    p.add_argument("--eval-scan", choices=["scan1", "scan2", "full"], default="scan2",
                   help="用于构建数据 X_s 的 scan")
    p.add_argument("--atlas-key",
                   choices=["cluster_pred_raw", "cluster_pred_post", "cluster_pred"],
                   default="cluster_pred_raw")
    p.add_argument("--sphere-space", default="fsaverage",
                   choices=["fsaverage", "fsLR", "fsLR_msmsulc", "native"])
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--primary-metric", default="delta_FC",
                   choices=["delta_FC", "FC_within"])

    # ---- 实验选择 ----
    p.add_argument("--run-atlas-transfer", action="store_true", default=True,
                   help="臂 1：A_t → s 空间图谱迁移（默认开启）")
    p.add_argument("--run-repr-swap", action="store_true", default=False,
                   help="臂 2：只对单一探针被试计算 M_shuf[s*, :]")
    p.add_argument("--repr-swap-probe-subject", default=None,
                   help="臂 2 的探针被试；None → subjects 列表的第一个")

    # ---- 臂 2 投影参数 ----
    p.add_argument("--shuffle-mode", default="knn_gaussian",
                   choices=["nearest", "knn_mean", "knn_gaussian", "knn_inverse"])
    p.add_argument("--shuffle-k", type=int, default=4)
    p.add_argument("--shuffle-sigma", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) / f"{args.atlas_scan}_atlas__{args.eval_scan}_bold"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, cfg = load_model_from_checkpoint(args.checkpoint, device)
    brain_dataset = get_dataset("brain", pe_dim=cfg["pe_dim"])
    result_root = Path(args.result_root)
    subject_dir_root = Path(args.subject_dir_root)

    print("=" * 80)
    print("Subject-Specificity Experiment")
    print(f"  atlas_scan      = {args.atlas_scan}")
    print(f"  eval_scan       = {args.eval_scan}")
    print(f"  atlas_key       = {args.atlas_key}")
    print(f"  sphere_space    = {args.sphere_space}")
    print(f"  run_atlas_transfer = {args.run_atlas_transfer}")
    print(f"  run_repr_swap      = {args.run_repr_swap}")
    print(f"  split_mode      = {cfg.get('split_mode', 'subject')}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1) 收集被试：atlas / X_norm / Z^A / adj / prior / nodes_paths
    # ------------------------------------------------------------------
    subjects = []
    atlas_dict = {}
    nodes_paths = {}
    X_norm_dict = {}
    Z_A_dict = {}
    adj_dict = {}
    prior_dict = {}

    for idx in range(len(brain_dataset)):
        sample = brain_dataset[idx]
        sub_id = sample["subject"]

        npz_path = result_root / sub_id / args.atlas_scan / "parcellation_result.npz"
        nodes_path = Path(sample["path"]).with_suffix(".nodes.npz")
        if not npz_path.exists():
            print(f"[SKIP] {sub_id}: 缺少 {npz_path}")
            continue
        if not nodes_path.exists():
            print(f"[SKIP] {sub_id}: 缺少 {nodes_path}")
            continue

        try:
            A = load_atlas_npz(npz_path, args.atlas_key)
        except KeyError as e:
            print(f"[SKIP] {sub_id}: {e}")
            continue

        # scan B：评估用 BOLD
        rec_B = extract_node_rep(model, sample, cfg, device, which_scan=args.eval_scan)

        # scan A：臂 2 专用（只在需要时提取）
        if args.run_repr_swap:
            rec_A = extract_node_rep(model, sample, cfg, device, which_scan=args.atlas_scan)
            Z_A_dict[sub_id] = rec_A["node_rep"]

        X_norm_dict[sub_id] = normalize_bold(rec_B["raw_bold"])
        adj_dict[sub_id]    = rec_B["adj"]
        prior_dict[sub_id]  = rec_B["prior_labels"]

        subjects.append(sub_id)
        atlas_dict[sub_id]  = A
        nodes_paths[sub_id] = nodes_path

    if len(subjects) < 2:
        print("[ERROR] 有效被试数 < 2，无法构建特异性矩阵。")
        return

    print(f"[Align] 最终可用被试数: {len(subjects)}")

    # ------------------------------------------------------------------
    # 2) 投影质量 round-trip
    # ------------------------------------------------------------------
    proj_df = compute_projection_quality(
        subjects, atlas_dict, nodes_paths,
        subject_dir_root=subject_dir_root,
        sphere_space=args.sphere_space,
    )
    if not proj_df.empty:
        proj_df.to_csv(out_dir / "projection_quality_roundtrip.csv", index=False)

    # ------------------------------------------------------------------
    # 3) ARM-1：完整矩阵
    # ------------------------------------------------------------------
    report = {
        "_meta": {
            "atlas_scan": args.atlas_scan,
            "eval_scan": args.eval_scan,
            "atlas_key": args.atlas_key,
            "sphere_space": args.sphere_space,
            "n_subjects": int(len(subjects)),
            "primary_metric": args.primary_metric,
            "run_atlas_transfer": bool(args.run_atlas_transfer),
            "run_repr_swap": bool(args.run_repr_swap),
            "shuffle_mode": args.shuffle_mode if args.run_repr_swap else None,
            "shuffle_k": int(args.shuffle_k) if args.run_repr_swap else None,
        },
    }

    if not proj_df.empty:
        report["projection_quality"] = {
            "ari_mean":         float(proj_df["ari"].mean()),
            "ari_min":          float(proj_df["ari"].min()),
            "exact_rate_mean":  float(proj_df["exact_match_rate"].mean()),
            "dice_mean":        float(proj_df["mean_dice"].mean()),
            "n_subjects":       int(len(proj_df)),
        }

    M_delta = M_within = M_between = None
    if args.run_atlas_transfer:
        M_delta, M_within, M_between = build_atlas_transfer_matrix(
            subjects, X_norm_dict, atlas_dict, nodes_paths,
            subject_dir_root=subject_dir_root,
            sphere_space=args.sphere_space,
        )
        report["delta_FC"]  = analyze_specificity_matrix(
            M_delta, subjects, "delta_FC", n_perm=args.n_perm)
        report["FC_within"] = analyze_specificity_matrix(
            M_within, subjects, "FC_within", n_perm=args.n_perm)
        report["FC_between"] = analyze_specificity_matrix(
            M_between, subjects, "FC_between", n_perm=args.n_perm)

        save_matrices(out_dir, subjects, M_delta, M_within, M_between)
        per_subj_df = build_per_subject_dataframe(
            subjects, M_delta, M_within, M_between)
        per_subj_df.to_csv(out_dir / "per_subject_specificity.csv", index=False)

        primary_name = "ΔFC" if args.primary_metric == "delta_FC" else "FC_within"
        plot_specificity_results(
            M_delta, subjects, report["delta_FC"], out_dir,
            primary_name=primary_name,
        )

    # ------------------------------------------------------------------
    # 4) ARM-2：单探针行
    # ------------------------------------------------------------------
    probe_s = None
    if args.run_repr_swap:
        probe_s = args.repr_swap_probe_subject or subjects[0]
        if probe_s not in subjects:
            print(f"[WARN] 探针被试 {probe_s} 不在 subjects 中，回退到 {subjects[0]}")
            probe_s = subjects[0]

        M_shuf_row_delta, M_shuf_row_within, M_shuf_row_between = build_repr_swap_row(
            probe_s          = probe_s,
            subjects         = subjects,
            X_norm_dict      = X_norm_dict,
            Z_A_dict         = Z_A_dict,
            adj_dict         = adj_dict,
            prior_dict       = prior_dict,
            nodes_paths      = nodes_paths,
            atlas_dict       = atlas_dict,
            subject_dir_root = subject_dir_root,
            sphere_space     = args.sphere_space,
            shuffle_mode     = args.shuffle_mode,
            shuffle_k        = args.shuffle_k,
            shuffle_sigma    = args.shuffle_sigma,
        )

        report["shuffled_delta_FC_row"] = analyze_specificity_row(
            M_shuf_row_delta, subjects, probe_s, "shuffled_delta_FC_row")
        report["shuffled_FC_within_row"] = analyze_specificity_row(
            M_shuf_row_within, subjects, probe_s, "shuffled_FC_within_row")

        # 跨臂对比（仅当臂 1 也跑了）
        if args.run_atlas_transfer and M_delta is not None:
            s_idx = subjects.index(probe_s)
            M_atlas_row_delta = M_delta[s_idx, :]
            report["atlas_vs_swap_row_contrast"] = analyze_atlas_vs_swap_row(
                M_atlas_row_delta, M_shuf_row_delta, subjects, probe_s)

            row_df = pd.DataFrame({
                "subject_s": [probe_s] * len(subjects),
                "atlas_t":   subjects,
                "M_atlas_delta_FC":      M_atlas_row_delta,
                "M_shuf_delta_FC":       M_shuf_row_delta,
                "M_shuf_FC_within":      M_shuf_row_within,
                "M_shuf_FC_between":     M_shuf_row_between,
                "diff_atlas_minus_swap": M_atlas_row_delta - M_shuf_row_delta,
                "is_diagonal": [int(t == probe_s) for t in subjects],
            })
        else:
            row_df = pd.DataFrame({
                "subject_s": [probe_s] * len(subjects),
                "atlas_t":   subjects,
                "M_shuf_delta_FC":  M_shuf_row_delta,
                "M_shuf_FC_within": M_shuf_row_within,
                "M_shuf_FC_between":M_shuf_row_between,
                "is_diagonal": [int(t == probe_s) for t in subjects],
            })
        row_df.to_csv(out_dir / f"repr_swap_row_{probe_s}.csv", index=False)
        report["_meta"]["repr_swap_probe"] = probe_s

    # ------------------------------------------------------------------
    # 5) 保存 JSON
    # ------------------------------------------------------------------
    with open(out_dir / "specificity_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 6) 终端汇总
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("           SUBJECT-SPECIFICITY / FINGERPRINT REPORT")
    print(f"           atlas_scan={args.atlas_scan}  eval_scan={args.eval_scan}  "
          f"S={len(subjects)}")
    print("=" * 80)

    if args.run_atlas_transfer:
        for key, label in [("delta_FC",  "ARM-1 ΔFC = FC_within - FC_between"),
                           ("FC_within", "ARM-1 FC_within")]:
            a = report[key]
            if "error" in a:
                continue
            ident = a["identification"]
            ci = a["bootstrap_ci"]
            print(f" ► {label}")
            print(f"   - Diagonal  Mean/Med     : {a['diag']['mean']:.4f} / {a['diag']['median']:.4f}")
            print(f"   - Off-diag  Mean/Med     : {a['off_diag']['mean']:.4f} / {a['off_diag']['median']:.4f}")
            print(f"   - Δ = diag - off_mean    : {a['diag_minus_off_mean']:+.4f}")
            print(f"   - Paired t vs row-mean   : t={a['diag_vs_row_off_mean']['t_stat']:.3f}, "
                  f"p={a['diag_vs_row_off_mean']['p_value']:.4e}, "
                  f"d={a['diag_vs_row_off_mean']['cohen_d']:.2f}")
            print(f"   - Paired t vs row-max    : t={a['diag_vs_row_off_max']['t_stat']:.3f}, "
                  f"p={a['diag_vs_row_off_max']['p_value']:.4e}")
            print(f"   - Permutation test       : obs={a['permutation_test']['obs_diag_mean']:.4f}, "
                  f"null={a['permutation_test']['null_mean']:.4f}±{a['permutation_test']['null_std']:.4f}, "
                  f"p={a['permutation_test']['p_value_one_sided']:.4f}")
            print(f"   - Diagonal Mean (bootstrap 95% CI): {a['diag']['mean']:.4f}  "
                  f"[{ci['diag_mean'][0]:.4f}, {ci['diag_mean'][1]:.4f}]")
            print(f"   - Top-1  identification  : {ident['top1_acc']:.3f}   "
                  f"(chance = {ident['chance_top1']:.3f}, "
                  f"×{ident['top1_acc']/max(ident['chance_top1'],1e-8):.1f} over chance; "
                  f"95% CI = [{ci['top1_acc'][0]:.3f}, {ci['top1_acc'][1]:.3f}])")
            print(f"   - Mean rank of own atlas : {ident['mean_rank']:.2f}   "
                  f"(chance = {ident['chance_mean_rank']:.2f}; "
                  f"95% CI = [{ci['mean_rank'][0]:.2f}, {ci['mean_rank'][1]:.2f}])")
            topk_str = ", ".join([f"{k}={v:.3f}" for k, v in ident["topk_acc"].items()])
            print(f"   - Top-K accuracy         : {topk_str}")
            print("-" * 80)

    if args.run_repr_swap and "shuffled_delta_FC_row" in report:
        r = report["shuffled_delta_FC_row"]
        print(f" ► ARM-2 Repr-Swap single-row null  (probe = {r['probe_subject']})")
        print(f"   - Diag  ΔFC                     : {r['diag_value']:+.4f}")
        print(f"   - Off-diag Mean / Max           : {r['off_mean']:+.4f} / {r['off_max']:+.4f}")
        print(f"   - Diag − Off_mean               : {r['diag_minus_off_mean']:+.4f}")
        print(f"   - Diag − Off_max                : {r['diag_minus_off_max']:+.4f}")
        print(f"   - Diag rank (1=best) / S        : {r['diag_rank_desc']} / {r['n_subjects']}")
        print(f"   - Permutation p (row-internal)  : {r['perm_p_value_one_sided']:.4f}")
        print(f"   - Top-1 hit                     : {'YES' if r['is_top1'] else 'no'}")
        print("-" * 80)

        if "atlas_vs_swap_row_contrast" in report:
            c = report["atlas_vs_swap_row_contrast"]
            print(" ► ARM1 − ARM2  (single row, off-diagonal entries only)")
            print(f"   - Mean (M_atlas − M_swap)       : {c['mean_diff_off']:+.4f}  "
                  f"(±{c['std_diff_off']:.4f})")
            print(f"   - N donors where atlas > swap   : {c['n_atlas_above_swap']}/{c['n_off']}  "
                  f"(sign-test p = {c['sign_test_p']:.4e})")
            print(f"   - One-sample t (H0: mean diff=0): t={c['one_sample_t']['t_stat']:.3f}, "
                  f"p={c['one_sample_t']['p_value']:.4e}, d={c['cohen_d']:.2f}")
            print("   - 解释: 若此差显著 > 0，说明个体特异性来自 atlas 几何本身，")
            print("           而非驱动聚类的表示携带的泛化偏置。")
            print("-" * 80)

    if not proj_df.empty:
        pq = report["projection_quality"]
        print(" ► Projection quality (s→s round-trip, ARI 上界 = 1.0)")
        print(f"   - ARI       Mean / Min : {pq['ari_mean']:.4f} / {pq['ari_min']:.4f}")
        print(f"   - Exact match rate Mean: {pq['exact_rate_mean']:.4f}")
        print(f"   - Parcel Dice Mean     : {pq['dice_mean']:.4f}")
        print(f"   - 解释: ARI < 1 说明球面 NN 投影本身引入噪声，")
        print(f"           off-diagonal 被'额外压低'；observed Δ 因此是 specificity 的保守估计。")
        print("-" * 80)

    print("=" * 80)
    print(f" [DONE] 完整结果已写入: {out_dir.resolve()}")


if __name__ == "__main__":
    main()