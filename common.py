# common.py
# -*- coding: utf-8 -*-
"""
统一公共模块：所有主 pipeline 及分析脚本共享的工具函数。

内容分区：
    A. 通用工具（to_scipy_csr, resolve_device）
    B. 模型（build_model, load_model_from_checkpoint, fast_re_features,
              extract_node_rep(s)）——适配 VarLenBrainTransformer（变长 T）
       B2. BOLD 时间序列切分（与 run.py 训练逻辑对齐）
    C. FC 指标（mean_pairwise_fc, random_fc_baseline）
    D. Embedding 指标（within_z, between_z, random_z_baseline）
    E. 图结构工具（extract_connected_components, count_connected_components,
                    get_spatial_neighbors）
    F. 标签/表面映射（save_label_gii, nodes_to_vertices,
                      scatter_graph_labels_to_full_vertex_space）
    G. 渲染（render_preview）
    H. 跨被试特征映射（基于球面配准）

变长改造说明：
    - 模型侧（model.py）已切换为 VarLenBrainTransformer，T 由运行时决定；
    - 因此 common 里所有 "bold_dim = input_dim - pe_dim" 一律改为
      按当前样本实算：bold_dim = features.size(1) - pe_dim；
    - extract_node_rep 内部不再显式调 fast_re_features，
      因为新模型 extract_node_representation 已内置该传播。

约定：
    - 所有函数不依赖具体数据集路径，只接受 numpy / scipy.sparse / torch.Tensor。
    - 不在此文件里设置线程环境变量；线程数由入口脚本决定。
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import scipy.sparse as sp
import torch

# 延迟导入：nibabel / nilearn / 项目模块在真正调用时才 import，
# 避免脚本只需要其中一部分功能时被拖累。
# import nibabel as nib
# from data import get_dataset
# from model import VarLenBrainTransformer
# import utils


# ======================================================================
# A. 通用工具
# ======================================================================

def to_scipy_csr(adj_matrix) -> sp.csr_matrix:
    """将 PyTorch 稀疏/稠密 Tensor 或任意稀疏矩阵转为 SciPy CSR。"""
    if isinstance(adj_matrix, torch.Tensor):
        if adj_matrix.is_sparse:
            adj_coo = adj_matrix.coalesce()
            row = adj_coo.indices()[0].cpu().numpy()
            col = adj_coo.indices()[1].cpu().numpy()
            val = adj_coo.values().cpu().numpy()
            N = adj_matrix.shape[0]
            return sp.coo_matrix((val, (row, col)), shape=(N, N)).tocsr()
        return sp.csr_matrix(adj_matrix.cpu().numpy())
    if isinstance(adj_matrix, sp.spmatrix):
        return adj_matrix.tocsr()
    return sp.csr_matrix(adj_matrix)


def resolve_device(device_spec: Optional[str] = None) -> torch.device:
    """解析设备字符串；未指定时自动选择 cuda/cpu。"""
    if device_spec is None:
        device_spec = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(str(device_spec))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[Device] CUDA requested but unavailable -> CPU")
        return torch.device("cpu")
    return device


# ======================================================================
# B. 模型
# ======================================================================

def build_model(cfg: dict, device: torch.device, input_dim: int = None):
    """
    按 checkpoint config 重建 VarLenBrainTransformer（变长版本）。

    input_dim：仅为向后兼容旧 checkpoint 保留，变长模型不再使用，可传 None。

    可选超参（只有 cfg 显式给出时才透传，避免旧 checkpoint 里没有这些键
    导致构造参数与 state_dict 结构不匹配）：
        max_timesteps, tok_dim, chunk_size, dec_channels,
        mask_ratio, decoder_layers, dropout_rate, attention_dropout_rate,
        use_grad_ckpt
    """
    from model import VarLenBrainTransformer

    kwargs = dict(
        hops=cfg["K"],
        n_class=1,
        pe_dim=cfg["pe_dim"],
        n_layers=cfg["n_layers"],
        num_heads=cfg["num_heads"],
        hidden_dim=cfg["hidden_dim"],
        ffn_dim=cfg["ffn_dim"],
        ssl_dim=cfg["ssl_dim"],
    )
    for k in ("max_timesteps", "tok_dim", "chunk_size", "dec_channels",
              "mask_ratio", "decoder_layers", "dropout_rate",
              "attention_dropout_rate", "use_grad_ckpt"):
        if k in cfg:
            kwargs[k] = cfg[k]
    return VarLenBrainTransformer(**kwargs).to(device)


def load_model_from_checkpoint(ckpt_path, device):
    """加载 checkpoint，返回 (model, cfg, input_dim)。

    注意：input_dim 只为兼容旧 checkpoint 保留；变长模型不使用该值，
    返回 None 或旧 ckpt 记录的整数均属正常。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    input_dim = ckpt.get("input_dim", None)  # 旧 ckpt 才有；仅供打印/兼容
    model = build_model(cfg, device, input_dim=input_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Checkpoint] {ckpt_path} | device={device} | "
          f"pe_dim={cfg['pe_dim']} | K={cfg['K']} | "
          f"input_dim={input_dim} (informational only)")
    return model, cfg


def fast_re_features(adj: torch.Tensor, features: torch.Tensor, K: int):
    """
    包装 utils.fast_re_features，统一入口，便于日后替换实现。

    注：新模型 VarLenBrainTransformer.extract_node_representation 已内置
    该传播，common.extract_node_rep 不再调用本函数，但保留给其它下游脚本
    独立使用。
    """
    import utils
    return utils.fast_re_features(adj, features, K=K)


# ----------------------------------------------------------------------
# B2. BOLD 时间序列切分（与 run.py 训练逻辑对齐）
# ----------------------------------------------------------------------

def slice_bold_features(features, pe_dim: int, which: str = "scan1"):
    """
    将 features = [BOLD | PE] 按时间维二等分。

    which:
        "scan1" → 前半段 BOLD + 完整 PE
        "scan2" → 后半段 BOLD + 完整 PE（长度与 scan1 对齐，奇数帧丢弃末帧）
        "full"  → 不做切分，原样返回

    返回 (new_features, new_feature_dim, original_bold_dim)。
    features 可以是 torch.Tensor 或 np.ndarray。

    注：bold_dim = total_dim - pe_dim 每次现算，天然兼容任意长度的 BOLD。
    """
    assert which in ("scan1", "scan2", "full")

    is_torch = isinstance(features, torch.Tensor)
    if not is_torch:
        features = torch.as_tensor(features)

    total_dim = features.shape[1]
    bold_dim = total_dim - pe_dim
    if bold_dim < 1:
        raise ValueError(f"Invalid bold_dim={bold_dim} (total={total_dim}, pe={pe_dim})")

    if which == "full":
        new_features = features
        new_dim = total_dim
    else:
        half = bold_dim // 2
        if half < 1:
            raise ValueError(f"BOLD time series too short to split: bold_dim={bold_dim}")
        pe = features[:, bold_dim:]
        if which == "scan1":
            bold = features[:, :half]
        else:  # scan2
            bold = features[:, half : half + half]
        new_features = torch.cat([bold, pe], dim=1)
        new_dim = new_features.shape[1]

    if not is_torch:
        new_features = new_features.numpy()

    return new_features, int(new_dim), int(bold_dim)


def apply_split_to_sample(sample: dict, pe_dim: int, which: str = "scan1") -> dict:
    """
    对单个 sample dict 应用时间序列切分，返回新 dict（不修改原 sample）。
    which 含义同 slice_bold_features。
    """
    new_sample = {k: v for k, v in sample.items()}
    features = sample["features"]
    new_features, new_dim, original_bold_dim = slice_bold_features(
        features, pe_dim=pe_dim, which=which
    )
    new_sample["features"] = new_features
    new_sample["feature_dim"] = new_dim
    new_sample["scan"] = which
    new_sample["original_bold_dim"] = original_bold_dim
    return new_sample


@torch.no_grad()
def extract_node_rep(model, sample: dict, cfg: dict, device: torch.device,
                     which_scan: str = None):
    """
    对单个样本提取：
        node_rep : [N, D]  numpy
        raw_bold : [N, T'] numpy  （已按 split 切分后的 BOLD）
        prior    : [N]     numpy int64
        adj      : CPU tensor
        path / subject / scan / ...

    时间序列兼容：
        - 若 cfg["split_mode"] == "timeseries"：
            默认 which_scan="scan2"（held-out 半段，与训练 scan1 对齐维度）
            可通过 which_scan 显式覆盖为 "scan1" / "scan2" / "full"
        - 否则（subject 或未设置）：使用完整时间序列

    变长适配：
        - bold_dim 按当前样本实算（features.size(1) - pe_dim），
          不再依赖 checkpoint 里记录的全局 input_dim；
        - 内部不再显式调 fast_re_features；
          新模型 extract_node_representation(adj, X_raw, pe) 已内置该传播。
    """
    K = cfg["K"]
    pe_dim = cfg["pe_dim"]
    split_mode = cfg.get("split_mode", "subject")

    # 决定是否切分
    if which_scan is None:
        if split_mode == "timeseries":
            which_scan = "scan2"  # 默认推理用 held-out 半段
        else:
            which_scan = "full"

    if which_scan != "full":
        sample = apply_split_to_sample(sample, pe_dim=pe_dim, which=which_scan)

    adj_dev = sample["adj"].to(device)
    features = sample["features"]
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    features = features.to(device)

    # —— 变长关键：按当前样本现算 bold_dim ——
    bold_dim = features.size(1) - pe_dim
    if bold_dim < 1:
        raise ValueError(
            f"bold_dim={bold_dim} <= 0 "
            f"(feature_dim={features.size(1)}, pe_dim={pe_dim})"
        )

    X_raw = features[:, :bold_dim]
    pe    = features[:, bold_dim:]

    # 新模型内部自带 fast_re_features + hop tokenizer + encoder + readout
    node_rep = model.extract_node_representation(
        adj_dev, X_raw, pe
    ).detach().cpu().numpy()

    raw_bold = X_raw.detach().cpu().numpy()

    labels = sample.get("labels", None)
    if isinstance(labels, torch.Tensor):
        prior = labels.cpu().numpy()
    elif labels is not None:
        prior = np.asarray(labels)
    else:
        prior = None

    # adj 统一放回 CPU，保持原返回类型
    adj_out = sample["adj"]
    if isinstance(adj_out, torch.Tensor):
        adj_out = adj_out.cpu()

    return {
        "subject": sample["subject"],
        "path": sample["path"],
        "node_rep": node_rep,
        "raw_bold": raw_bold,
        "prior_labels": prior,
        "adj": adj_out,
        "feature_dim": int(features.size(1)),   # 当前样本实际宽度（变长）
        "pe_dim": int(pe_dim),
        "K": int(K),
        "scan": which_scan,
        "split_mode": split_mode,
    }


@torch.no_grad()
def extract_node_reps(model, dataset, cfg: dict, device: torch.device,
                      which_scan: str = None):
    """
    对数据集（或 Subset）逐样本提取，返回 list[dict]。
    which_scan 透传给 extract_node_rep；None 时按 cfg["split_mode"] 自动选择
    （timeseries → scan2，否则 full）。
    """
    results = []
    for i in range(len(dataset)):
        sample = dataset[i]
        rec = extract_node_rep(model, sample, cfg, device, which_scan=which_scan)
        results.append(rec)
        scan_tag = rec.get("scan", "full")
        print(f"  [{i+1}/{len(dataset)}] {rec['subject']} "
              f"scan={scan_tag} "
              f"N={rec['node_rep'].shape[0]} D={rec['node_rep'].shape[1]} "
              f"T={rec['raw_bold'].shape[1]}")
    return results


# ======================================================================
# C. FC 指标
# ======================================================================

def mean_pairwise_fc(bold: np.ndarray, nodes_a, nodes_b=None) -> float:
    """
    节点集内部或两集合之间的平均 Pearson 相关。
    - nodes_b is None：nodes_a 内部平均 FC。
    - 否则：nodes_a × nodes_b 交叉平均 FC。
    """
    if nodes_b is None:
        if len(nodes_a) < 2:
            return np.nan
        X = bold[nodes_a]
        X = X - X.mean(1, keepdims=True)
        X = X / (X.std(1, keepdims=True) + 1e-8)
        corr = (X @ X.T) / X.shape[1]
        iu = np.triu_indices(len(nodes_a), k=1)
        return float(np.nanmean(corr[iu]))

    if len(nodes_a) == 0 or len(nodes_b) == 0:
        return np.nan
    Xa = bold[nodes_a] - bold[nodes_a].mean(1, keepdims=True)
    Xb = bold[nodes_b] - bold[nodes_b].mean(1, keepdims=True)
    Xa = Xa / (Xa.std(1, keepdims=True) + 1e-8)
    Xb = Xb / (Xb.std(1, keepdims=True) + 1e-8)
    return float(np.nanmean((Xa @ Xb.T) / Xa.shape[1]))


def random_fc_baseline(bold: np.ndarray, size_a: int, size_b: int,
                       n_samples: int = 200,
                       hemi_mask: Optional[np.ndarray] = None,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """size-matched 随机 FC null：从 hemi_mask 指定池中随机抽 size_a + size_b 个节点。"""
    if rng is None:
        rng = np.random.default_rng(42)
    pool = np.where(hemi_mask)[0] if hemi_mask is not None else np.arange(bold.shape[0])
    vals = []
    for _ in range(n_samples):
        if len(pool) < size_a + size_b:
            break
        chosen = rng.choice(pool, size=size_a + size_b, replace=False)
        vals.append(mean_pairwise_fc(bold, chosen[:size_a], chosen[size_a:]))
    return np.asarray(vals)


# ======================================================================
# D. Embedding 指标
# ======================================================================

def _pairwise_dist(X: np.ndarray) -> np.ndarray:
    """行向量两两欧氏距离。"""
    sq = (X * X).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    return np.sqrt(np.maximum(D2, 0.0))


def within_z(emb: np.ndarray, nodes) -> float:
    """节点集内部平均 embedding 距离。"""
    if len(nodes) < 2:
        return np.nan
    D = _pairwise_dist(emb[nodes])
    iu = np.triu_indices(len(nodes), k=1)
    return float(np.nanmean(D[iu]))


def between_z(emb: np.ndarray, nodes_a, nodes_b) -> float:
    """两集合之间平均 embedding 距离（对称）。"""
    if len(nodes_a) == 0 or len(nodes_b) == 0:
        return np.nan
    Xa = emb[nodes_a]
    Xb = emb[nodes_b]
    D = np.sqrt(np.maximum(
        (Xa * Xa).sum(1)[:, None] + (Xb * Xb).sum(1)[None, :] - 2.0 * Xa @ Xb.T,
        0.0,
    ))
    return float(np.nanmean(D))


def between_z_centroid(emb: np.ndarray, nodes_a, nodes_b) -> float:
    """两组质心之间的 embedding 距离。"""
    if len(nodes_a) == 0 or len(nodes_b) == 0:
        return np.nan
    ca = emb[nodes_a].mean(0)
    cb = emb[nodes_b].mean(0)
    return float(np.linalg.norm(ca - cb))


def random_z_baseline(emb: np.ndarray, size_a: int, size_b: int,
                      n_samples: int = 200,
                      hemi_mask: Optional[np.ndarray] = None,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """size-matched 随机 embedding null。"""
    if rng is None:
        rng = np.random.default_rng(42)
    pool = np.where(hemi_mask)[0] if hemi_mask is not None else np.arange(emb.shape[0])
    vals = []
    for _ in range(n_samples):
        if len(pool) < size_a + size_b:
            break
        chosen = rng.choice(pool, size=size_a + size_b, replace=False)
        vals.append(between_z(emb, chosen[:size_a], chosen[size_a:]))
    return np.asarray(vals)


# ======================================================================
# E. 图结构工具
# ======================================================================

def extract_connected_components(node_indices, adj_csr: sp.csr_matrix):
    """给定节点子集，返回其在 adj_csr 上的连通分量（list[np.ndarray]）。"""
    if len(node_indices) == 0:
        return []
    mask = np.zeros(adj_csr.shape[0], dtype=bool)
    mask[node_indices] = True
    sub_adj = adj_csr[mask][:, mask]
    n_comp, labels = sp.csgraph.connected_components(sub_adj, directed=False)
    return [node_indices[labels == c] for c in range(n_comp)]


def count_connected_components(cluster_labels: np.ndarray,
                               adj_csr: sp.csr_matrix) -> int:
    """统计一组 cluster 标签在其 induced subgraph 上的连通分量总数。"""
    unique_labels = np.unique(cluster_labels[cluster_labels > 0])
    total_components = 0
    for c in unique_labels:
        nodes = np.where(cluster_labels == c)[0]
        if len(nodes) == 0:
            continue
        mask = np.zeros(adj_csr.shape[0], dtype=bool)
        mask[nodes] = True
        subgraph = adj_csr[mask, :][:, mask]
        n_comp, _ = sp.csgraph.connected_components(subgraph, directed=False)
        total_components += n_comp
    return total_components


def count_connected_components_fullmesh(labels_full: np.ndarray,
                                        adj_full_csr: sp.csr_matrix):
    """在完整 mesh 顶点空间上计算 fragmentation index。返回 (total_comp, frag_idx)。"""
    unique_labels = np.unique(labels_full[labels_full > 0])
    total_comp = 0
    for c in unique_labels:
        nodes = np.where(labels_full == c)[0]
        if len(nodes) == 0:
            continue
        mask = np.zeros(adj_full_csr.shape[0], dtype=bool)
        mask[nodes] = True
        subg = adj_full_csr[mask, :][:, mask]
        n_comp, _ = sp.csgraph.connected_components(subg, directed=False)
        total_comp += n_comp
    n_cluster = len(unique_labels)
    frag_idx = total_comp / (n_cluster + 1e-8)
    return total_comp, frag_idx


def get_spatial_neighbors(component_nodes, adj_csr: sp.csr_matrix,
                          exclude_nodes) -> np.ndarray:
    """返回 component_nodes 的一环邻居，去掉自身和 exclude_nodes。"""
    neighbors = set()
    for n in component_nodes:
        s, e = adj_csr.indptr[n], adj_csr.indptr[n + 1]
        neighbors.update(adj_csr.indices[s:e].tolist())
    neighbors -= set(component_nodes)
    neighbors -= set(exclude_nodes)
    return np.array(sorted(neighbors), dtype=int)


def mean_geodesic(adj_csr: sp.csr_matrix, nodes_a, nodes_b,
                  max_dist: int = 60) -> float:
    """BFS 估计 nodes_a → nodes_b 的平均 geodesic 距离（图 hop）。"""
    if len(nodes_a) == 0 or len(nodes_b) == 0:
        return np.nan
    if len(nodes_a) > len(nodes_b):
        nodes_a, nodes_b = nodes_b, nodes_a
    dist = np.full(adj_csr.shape[0], -1, dtype=np.int32)
    q = deque()
    for n in nodes_a:
        dist[n] = 0
        q.append(n)
    while q:
        u = q.popleft()
        if dist[u] >= max_dist:
            continue
        s, e = adj_csr.indptr[u], adj_csr.indptr[u + 1]
        for v in adj_csr.indices[s:e]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    d = dist[nodes_b]
    d = d[d >= 0]
    return float(d.mean()) if len(d) else np.nan


# ======================================================================
# F. 标签 / 表面映射
# ======================================================================

def save_label_gii(values: np.ndarray, output_path) -> None:
    """保存 int32 label GII。"""
    import nibabel as nib
    values = np.asarray(values, dtype=np.int32).ravel()
    darray = nib.gifti.GiftiDataArray(
        data=values,
        intent="NIFTI_INTENT_LABEL",
        datatype="NIFTI_TYPE_INT32",
    )
    img = nib.gifti.GiftiImage(darrays=[darray])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(output_path))


def load_native_vertex_count(subject_dir, subject: str, hemi: str) -> int:
    """读取 pial surf.gii 的顶点数。"""
    import nibabel as nib
    pial = Path(subject_dir) / "anat" / f"{subject}_hemi-{hemi}_pial.surf.gii"
    if not pial.exists():
        raise FileNotFoundError(pial)
    img = nib.load(str(pial))
    for d in img.darrays:
        if d.intent == 1008:
            return d.data.shape[0]
    raise RuntimeError(f"{pial} 找不到 POINTSET")


def nodes_to_vertices(cluster_pred: np.ndarray, hemisphere: np.ndarray,
                      orig_vertex: np.ndarray, n_L: int, n_R: int,
                      offset: int = 0):
    """graph-node 标签 → (LH, RH) 完整表面顶点标签。"""
    label_L = np.zeros(n_L, dtype=np.int32)
    label_R = np.zeros(n_R, dtype=np.int32)
    m_L = hemisphere == "L"
    m_R = ~m_L
    label_L[orig_vertex[m_L]] = (cluster_pred[m_L] + offset).astype(np.int32)
    label_R[orig_vertex[m_R]] = (cluster_pred[m_R] + offset).astype(np.int32)
    return label_L, label_R


def scatter_graph_labels_to_full_vertex_space(pred_labels_graph: np.ndarray,
                                              nodes_npz_path) -> tuple:
    """将 graph-node 预测标签散射回完整 LH / RH 原始表面顶点空间。"""
    mapping = np.load(nodes_npz_path, allow_pickle=True)
    orig_v = mapping["original_surface_vertex"]
    hemi_arr = mapping["hemisphere"]

    graph_lh_mask = hemi_arr == "L"
    graph_rh_mask = hemi_arr == "R"

    lh_orig_vert_ids = orig_v[graph_lh_mask]
    rh_orig_vert_ids = orig_v[graph_rh_mask]

    n_lh_full = int(np.max(lh_orig_vert_ids)) + 1 if len(lh_orig_vert_ids) > 0 else 0
    n_rh_full = int(np.max(rh_orig_vert_ids)) + 1 if len(rh_orig_vert_ids) > 0 else 0

    lh_full_labels = np.zeros(n_lh_full, dtype=np.int64)
    rh_full_labels = np.zeros(n_rh_full, dtype=np.int64)

    lh_full_labels[lh_orig_vert_ids] = pred_labels_graph[graph_lh_mask]
    rh_full_labels[rh_orig_vert_ids] = pred_labels_graph[graph_rh_mask]
    return lh_full_labels, rh_full_labels


def build_full_mesh_adjacency(surf_gii_path) -> tuple:
    """加载完整 pial surf.gii，构建全 mesh 顶点邻接矩阵 (CSR, n_vertices)。"""
    import nibabel as nib
    img = nib.load(str(surf_gii_path))
    faces = None
    coords = None
    for darray in img.darrays:
        intent = darray.intent
        if intent == 1008:
            coords = np.asarray(darray.data)
        elif intent == 1009:
            faces = np.asarray(darray.data, dtype=np.int64)
    n_vertices = coords.shape[0]
    faces = np.asarray(faces, dtype=np.int64)
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    data = np.ones(len(rows), dtype=np.float32)
    A_coo = sp.coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    A_coo.sum_duplicates()
    A_coo.setdiag(0)
    A_coo.eliminate_zeros()
    return A_coo.tocsr(), n_vertices


# ======================================================================
# G. 渲染
# ======================================================================

def render_preview(surf_mesh, roi_vec, out_png, hemi: str,
                   bg_map=None, title: Optional[str] = None) -> None:
    """nilearn 表面 ROI 预览图（lateral + medial）。缺 nilearn 时跳过。"""
    try:
        from nilearn import plotting
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] nilearn / matplotlib 未安装，跳过预览图")
        return

    roi_vec = np.asarray(roi_vec, dtype=np.float32)
    n_neg = int((roi_vec < 0).sum())
    if n_neg > 0:
        print(f"    [fix] roi_vec 含 {n_neg} 个负值 → 归 0")
        roi_vec = np.where(roi_vec < 0, 0, roi_vec)

    if roi_vec.max() <= 0:
        print(f"    [skip] {Path(out_png).name}: roi_vec 全为背景")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), subplot_kw={"projection": "3d"})
    for ax, view in zip(axes, ["lateral", "medial"]):
        plotting.plot_surf_roi(
            surf_mesh=str(surf_mesh),
            roi_map=roi_vec,
            hemi=hemi,
            view=view,
            bg_map=str(bg_map) if bg_map else None,
            colorbar=True,
            title=f"{title} ({view})" if title else view,
            axes=ax,
        )
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    [ok] {out_png.name}")


# ======================================================================
# H. 跨被试特征映射（基于球面配准）
# ======================================================================

import json
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------
# H1. Surface / GIFTI IO
# ----------------------------------------------------------------------

def load_surface_gii(path):
    """读取 surface GIFTI，返回 (coords[N,3] float64, faces[M,3] int64)。"""
    import nibabel as nib
    img = nib.load(str(path))
    coords, faces = None, None
    for d in img.darrays:
        if d.intent == 1008:
            coords = np.asarray(d.data, dtype=np.float64)
        elif d.intent == 1009:
            faces = np.asarray(d.data, dtype=np.int64)
    if coords is None:
        raise RuntimeError(f"{path} 中找不到 POINTSET")
    return coords, faces


def load_gifti_scalar(path) -> np.ndarray:
    """读取 shape / label GIFTI 的标量数组。"""
    import nibabel as nib
    img = nib.load(str(path))
    return np.asarray(img.darrays[0].data)


def save_gifti_scalar(values, output_path,
                      intent: str = "NIFTI_INTENT_LABEL",
                      datatype: str = "NIFTI_TYPE_INT32") -> None:
    """把一维标量保存为 GIFTI（默认 label）。"""
    import nibabel as nib
    values = np.asarray(values).ravel()
    darray = nib.gifti.GiftiDataArray(
        data=values, intent=intent, datatype=datatype,
    )
    img = nib.gifti.GiftiImage(darrays=[darray])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(output_path))


# ----------------------------------------------------------------------
# H2. 球面配准文件探测
# ----------------------------------------------------------------------

_SPHERE_SUFFIXES = {
    "fsaverage":    "space-fsaverage_desc-reg_sphere.surf.gii",
    "fsLR":         "space-fsLR_desc-reg_sphere.surf.gii",
    "fsLR_msmsulc": "space-fsLR_desc-msmsulc_sphere.surf.gii",
    "native":       "sphere.surf.gii",
}


def detect_sphere_reg_file(subject_dir, subject: str, hemi: str,
                           space: str = "fsaverage") -> Path:
    """返回指定空间的配准球面路径。"""
    if space not in _SPHERE_SUFFIXES:
        raise ValueError(f"未知空间: {space}，可选: {list(_SPHERE_SUFFIXES)}")
    anat = Path(subject_dir) / subject / "anat"
    path = anat / f"{subject}_hemi-{hemi}_{_SPHERE_SUFFIXES[space]}"
    if not path.exists():
        raise FileNotFoundError(f"找不到配准球面: {path}")
    return path


# ----------------------------------------------------------------------
# H3. 顶点级球面最近邻投影
# ----------------------------------------------------------------------

def project_vertex_labels_via_sphere(labels_src,
                                     src_sphere_reg_path,
                                     tgt_sphere_reg_path,
                                     mode: str = "nearest",
                                     k: int = 1) -> np.ndarray:
    """
    把 src native 顶点标签映射到 tgt native 顶点集。
    两个 sphere_reg 必须配准到同一公共空间。
    """
    src_coords, _ = load_surface_gii(src_sphere_reg_path)
    tgt_coords, _ = load_surface_gii(tgt_sphere_reg_path)

    labels_src = np.asarray(labels_src)
    if labels_src.shape[0] != src_coords.shape[0]:
        raise ValueError(
            f"labels_src 长度 {labels_src.shape[0]} 与 src sphere "
            f"顶点数 {src_coords.shape[0]} 不一致"
        )

    tree = cKDTree(src_coords)

    if mode == "nearest":
        _, idx = tree.query(tgt_coords, k=1)
        return labels_src[idx]

    if mode == "majority":
        _, idx = tree.query(tgt_coords, k=int(k))
        idx = np.atleast_2d(idx)
        n_tgt = tgt_coords.shape[0]
        labels_tgt = np.full(n_tgt, -1, dtype=labels_src.dtype)
        for i in range(n_tgt):
            neigh = labels_src[idx[i]]
            vals, counts = np.unique(neigh[neigh >= 0], return_counts=True)
            if len(vals) > 0:
                labels_tgt[i] = vals[np.argmax(counts)]
        return labels_tgt

    raise ValueError(f"未知 mode: {mode}")


# ----------------------------------------------------------------------
# H4. Vertex 级 native → native
# ----------------------------------------------------------------------

def remap_atlas_native_to_native(labels_src,
                                 src_subject_dir, src_subject,
                                 tgt_subject_dir, tgt_subject,
                                 hemi: str,
                                 space: str = "fsaverage",
                                 mode: str = "nearest",
                                 k: int = 1) -> np.ndarray:
    """
    把 subject s native 顶点标签映射到 subject t native 顶点。
    """
    src_sphere = detect_sphere_reg_file(src_subject_dir, src_subject, hemi, space)
    tgt_sphere = detect_sphere_reg_file(tgt_subject_dir, tgt_subject, hemi, space)
    return project_vertex_labels_via_sphere(
        labels_src, src_sphere, tgt_sphere, mode=mode, k=k
    )


# ----------------------------------------------------------------------
# H5. Graph-node mapping helpers
# ----------------------------------------------------------------------

def load_nodes_mapping(nodes_npz_path) -> dict:
    """读取 nodes.npz，返回 dict（至少含 hemisphere / original_surface_vertex）。"""
    data = np.load(nodes_npz_path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    if "hemisphere" not in out or "original_surface_vertex" not in out:
        raise KeyError(
            f"{nodes_npz_path} 缺少 hemisphere / original_surface_vertex；"
            f"现有键: {list(out.keys())}"
        )
    return out


def _hemi_mask(mapping: dict, hemi: str) -> np.ndarray:
    target = hemi.upper()
    return np.array([target in str(h).upper() for h in mapping["hemisphere"]])


def graph_labels_to_native_vertices(labels_nodes: np.ndarray,
                                    mapping: dict,
                                    hemi: str,
                                    n_vertices: int,
                                    offset: int = 0) -> np.ndarray:
    """
    完整图节点标签 → 单半球 native 顶点标签。
    返回长度 = n_vertices，非目标半球位置为 0。
    """
    mask = _hemi_mask(mapping, hemi)
    v_ids = mapping["original_surface_vertex"][mask]
    v_labels = labels_nodes[mask]

    out = np.zeros(n_vertices, dtype=np.int64)
    out[v_ids] = (v_labels + offset).astype(np.int64)
    return out


def native_vertices_to_graph_labels(labels_vertices: np.ndarray,
                                    mapping: dict,
                                    hemi: str,
                                    offset: int = 0) -> np.ndarray:
    """
    单半球 native 顶点标签 → 该半球图节点标签（在 orig_v 位置采样）。
    返回长度 = 目标半球节点数。
    """
    mask = _hemi_mask(mapping, hemi)
    v_ids = mapping["original_surface_vertex"][mask]
    labels = labels_vertices[v_ids].astype(np.int64)
    if offset:
        labels = labels - offset
    return labels


# ----------------------------------------------------------------------
# H6. Graph-node 级 native → native（主入口）
# ----------------------------------------------------------------------

def remap_graph_atlas_between_subjects(labels_nodes_src: np.ndarray,
                                       src_nodes_npz_path,
                                       src_subject_dir, src_subject,
                                       tgt_nodes_npz_path,
                                       tgt_subject_dir, tgt_subject,
                                       hemi: str,
                                       space: str = "fsaverage",
                                       offset_src: int = 0,
                                       offset_tgt: int = 0,
                                       mode: str = "nearest") -> np.ndarray:
    """
    subject s 图节点标签（完整数组）→ subject t 图节点标签（该半球）。
    labels_nodes_src 为完整图节点数组；返回仅含目标半球节点。

    流程：
        src nodes --(orig_v)--> src native vertices
                 --(sphere NN)--> tgt native vertices
                 --(orig_v)--> tgt nodes
    """
    src_map = load_nodes_mapping(src_nodes_npz_path)
    tgt_map = load_nodes_mapping(tgt_nodes_npz_path)

    # 源/目标 native 顶点数直接从配准球面读
    src_sphere = detect_sphere_reg_file(src_subject_dir, src_subject, hemi, space)
    tgt_sphere = detect_sphere_reg_file(tgt_subject_dir, tgt_subject, hemi, space)
    src_coords, _ = load_surface_gii(src_sphere)
    tgt_coords, _ = load_surface_gii(tgt_sphere)
    n_src_v = src_coords.shape[0]
    n_tgt_v = tgt_coords.shape[0]

    # 1) src nodes → src vertices
    src_v = graph_labels_to_native_vertices(
        labels_nodes_src, src_map, hemi, n_src_v, offset=offset_src,
    )

    # 2) src vertices → tgt vertices
    tgt_v = project_vertex_labels_via_sphere(
        src_v, src_sphere, tgt_sphere, mode=mode,
    )

    # 3) tgt vertices → tgt nodes（该半球）
    tgt_node_labels = native_vertices_to_graph_labels(
        tgt_v, tgt_map, hemi, offset=offset_tgt,
    )
    return tgt_node_labels


def remap_graph_atlas_full_brain(labels_nodes_src: np.ndarray,
                                 src_nodes_npz_path,
                                 src_subject_dir, src_subject,
                                 tgt_nodes_npz_path,
                                 tgt_subject_dir, tgt_subject,
                                 space: str = "fsaverage",
                                 offset_src: int = 0,
                                 offset_tgt: int = 0,
                                 mode: str = "nearest",
                                 fill_value: int = -1) -> np.ndarray:
    """
    全脑图节点标签映射，返回与目标节点数等长的完整数组。
    LH / RH 分别映射，按目标 mapping 的顺序填充。
    """
    tgt_map = load_nodes_mapping(tgt_nodes_npz_path)
    n_tgt = len(tgt_map["hemisphere"])
    out = np.full(n_tgt, fill_value, dtype=np.int64)

    tgt_hemi = np.array([str(h).upper()[0] for h in tgt_map["hemisphere"]])

    for hemi in ["L", "R"]:
        if not np.any(tgt_hemi == hemi):
            continue
        hemi_labels = remap_graph_atlas_between_subjects(
            labels_nodes_src,
            src_nodes_npz_path, src_subject_dir, src_subject,
            tgt_nodes_npz_path, tgt_subject_dir, tgt_subject,
            hemi=hemi, space=space,
            offset_src=offset_src, offset_tgt=offset_tgt,
            mode=mode,
        )
        out[tgt_hemi == hemi] = hemi_labels

    return out


# ----------------------------------------------------------------------
# H7. 便捷：graph atlas ↔ GII
# ----------------------------------------------------------------------

def save_graph_atlas_as_gii(labels_nodes, nodes_npz_path,
                            subject_dir, subject,
                            out_dir, prefix: str = "atlas",
                            offset: int = 0) -> dict:
    """
    把图节点标签散射为 LH / RH native vertex GII 并保存。
    返回 {"L": path_L, "R": path_R}。
    """
    mapping = load_nodes_mapping(nodes_npz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    for hemi in ["L", "R"]:
        n_v = _native_vertex_count_for_hemi(subject_dir, subject, hemi)
        if n_v is None:
            continue
        v = graph_labels_to_native_vertices(
            labels_nodes, mapping, hemi, n_v, offset=offset,
        )
        path = out_dir / f"{prefix}_{hemi}.label.gii"
        save_gifti_scalar(v, path)
        saved[hemi] = str(path)
    return saved


def _native_vertex_count_for_hemi(subject_dir, subject: str, hemi: str):
    """从 pial surf.gii 读 native 顶点数；找不到返回 None。"""
    import nibabel as nib
    pial = Path(subject_dir) / subject / "anat" / \
        f"{subject}_hemi-{hemi}_pial.surf.gii"
    if not pial.exists():
        return None
    img = nib.load(str(pial))
    for d in img.darrays:
        if d.intent == 1008:
            return d.data.shape[0]
    return None


# ----------------------------------------------------------------------
# H8. 连续节点特征投影（基于球面配准的 k-NN / 加权平均）
#     与 H6（离散图谱标签最近邻）互补：用于 node_rep Z 等连续特征
# ----------------------------------------------------------------------

def _graph_node_sphere_coords(nodes_npz_path,
                              subject_dir, subject: str,
                              hemi: str, space: str = "fsaverage"):
    """
    取某被试某半球的图节点在配准球面上的坐标。

    返回
    ----
    coords  : (n_nodes_hemi, 3) float64   —— 每个图节点在其被试管配准球面上的坐标
    mask    : (n_nodes_total,) bool       —— 该半球节点在完整图节点数组中的布尔掩码
    """
    mapping = load_nodes_mapping(nodes_npz_path)
    mask = _hemi_mask(mapping, hemi)
    if not mask.any():
        return np.zeros((0, 3), dtype=np.float64), mask

    sphere_path = detect_sphere_reg_file(subject_dir, subject, hemi, space)
    sphere_coords, _ = load_surface_gii(sphere_path)

    orig_v = mapping["original_surface_vertex"][mask]
    return sphere_coords[orig_v], mask


def project_node_features_between_subjects(
        features_src: np.ndarray,
        src_nodes_npz_path,
        src_subject_dir, src_subject: str,
        tgt_nodes_npz_path,
        tgt_subject_dir, tgt_subject: str,
        space: str = "fsaverage",
        mode: str = "knn_gaussian",
        k: int = 4,
        sigma: Optional[float] = None,
        fill_value: float = 0.0) -> np.ndarray:
    """
    将源被试的连续图节点特征（如 node_rep Z，shape [N_src, D]）投影到目标被试图节点空间。

    与 `remap_graph_atlas_between_subjects`（离散标签，NN 单点采样）互补：
    本函数专门处理连续向量，支持邻居平均以降低球面 NN 采样抖动。

    参数
    ----
    features_src : (N_src_nodes, D) np.ndarray
        源图节点顺序下的连续特征。第 i 行对应 src 的第 i 个图节点。
    mode : {"nearest", "knn_mean", "knn_gaussian", "knn_inverse"}
        - "nearest"      : 最近邻单点（等价于现有 H6 的做法，供对照）
        - "knn_mean"     : k 个最近邻的简单平均
        - "knn_gaussian" : 高斯核加权平均，带宽 sigma；默认 sigma = 最近邻距离中位数
        - "knn_inverse"  : 1/d 加权平均
    k : int
        k-NN 邻居数（mode="nearest" 时忽略）
    sigma : float or None
        高斯核带宽（球面坐标单位）；None → 该半球所有目标节点最近邻距离的中位数
    fill_value : float
        目标节点若两半球均无覆盖时的填充值（一般不会发生）

    返回
    ----
    projected : (N_tgt_nodes, D) np.ndarray
        按目标图节点顺序填充。

    注意
    ----
    - 坐标取配准球面坐标，而非原生表面坐标（后者跨被试不可比）。
    - 两半球独立投影（L 与 R 配准到不同球面）。
    - 若 tgt 某节点的 k 个邻居中有元素来自不同半球（理论上不会发生），
      因为按半球分别处理，本函数天然规避了这个问题。
    """
    features_src = np.asarray(features_src)
    if not np.isfinite(features_src).all():
        n_bad = int((~np.isfinite(features_src)).sum())
        raise ValueError(f"features_src 含 {n_bad} 个 NaN/Inf，无法投影")
    if features_src.ndim != 2:
        raise ValueError(f"features_src 必须为 2D (N, D)，实际为 {features_src.shape}")

    src_map = load_nodes_mapping(src_nodes_npz_path)
    tgt_map = load_nodes_mapping(tgt_nodes_npz_path)
    n_src = len(src_map["hemisphere"])
    n_tgt = len(tgt_map["hemisphere"])

    if features_src.shape[0] != n_src:
        raise ValueError(
            f"features_src 第一维 {features_src.shape[0]} 与 src nodes {n_src} 不一致"
        )

    D = features_src.shape[1]
    out = np.full((n_tgt, D), fill_value, dtype=features_src.dtype)

    for hemi in ("L", "R"):
        src_coords, src_hemi_mask = _graph_node_sphere_coords(
            src_nodes_npz_path, src_subject_dir, src_subject, hemi, space,
        )
        tgt_coords, tgt_hemi_mask = _graph_node_sphere_coords(
            tgt_nodes_npz_path, tgt_subject_dir, tgt_subject, hemi, space,
        )
        if src_coords.shape[0] == 0 or tgt_coords.shape[0] == 0:
            continue

        src_feat = features_src[src_hemi_mask]

        # ---- nearest: 单点采样，直接走 H3 的老路 ----
        if mode == "nearest":
            tree = cKDTree(src_coords)
            _, idx = tree.query(tgt_coords, k=1)
            out[tgt_hemi_mask] = src_feat[idx]
            continue

        # ---- 邻居平均 ----
        k_eff = min(int(k), src_coords.shape[0])
        tree = cKDTree(src_coords)
        dists, idxs = tree.query(tgt_coords, k=k_eff)
        if k_eff == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]

        # 带宽：默认取该半球所有目标节点的最近邻距离中位数
        if sigma is None:
            sigma_eff = float(np.median(dists[:, 0])) + 1e-8
        else:
            sigma_eff = float(sigma)

        if mode == "knn_mean":
            w = np.ones_like(dists, dtype=np.float64)
        elif mode == "knn_gaussian":
            w = np.exp(-(dists ** 2) / (2.0 * sigma_eff ** 2))
        elif mode == "knn_inverse":
            w = 1.0 / (dists + 1e-8)
        else:
            raise ValueError(f"未知 mode: {mode}")

        denom = w.sum(axis=1, keepdims=True)
        zero_rows = (denom[:, 0] < 1e-12)
        if zero_rows.any():
            # 退化为 1-NN
            w[zero_rows] = 0.0
            w[zero_rows, 0] = 1.0
            denom[zero_rows] = 1.0
        w = w / denom          # (n_tgt_h, k)
        gathered = src_feat[idxs]                     # (n_tgt_h, k, D)
        out[tgt_hemi_mask] = np.einsum("nk,nkd->nd", w, gathered)

    return out