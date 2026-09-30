import os
from xml.parsers.expat import model
# 必须在导入 numpy, torch, sklearn 之前设置！
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import gc

from data import get_dataset
from model import VarLenBrainTransformer
import utils
from common import apply_split_to_sample

# ============================================================
# 全局默认配置
# ============================================================
DEFAULT_CONFIG = {
    "K": 2,
    "pe_dim": 16,
    "hidden_dim": 64,
    "num_heads": 8,
    "n_layers": 1,
    "ffn_dim": 64,
    "ssl_dim": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_epochs": 10,
    "num_subjects_expected": 9,
    "train_ratio": 0.75,          # 仅在 split_mode="subject" 时生效
    "seed": 42,
    # 划分模式：
    #   "timeseries"（默认）：BOLD 时间序列层次划分，将时间维二等分作为 scan1/scan2
    #   "subject"          ：被试层次划分，验证个体间泛化
    "split_mode": "timeseries",
}


# ============================================================
# 工具函数
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg, device):
    return VarLenBrainTransformer(
        hops=cfg["K"],
        n_class=1,
        pe_dim=cfg["pe_dim"],
        n_layers=cfg["n_layers"],
        num_heads=cfg["num_heads"],
        hidden_dim=cfg["hidden_dim"],
        ffn_dim=cfg["ffn_dim"],
        ssl_dim=cfg["ssl_dim"],
        use_grad_ckpt=True,
    ).to(device)


# ============================================================
# 时间序列二等分：构造 scan1 / scan2 视图
# ============================================================
class TimeSeriesSplitDataset(Dataset):
    """
    包装原始 brain_dataset，按时间维二等分。
    训练默认 which="scan1"；推理默认 which="scan2"（见 common.extract_node_rep）。
    """

    def __init__(self, base_dataset, pe_dim: int, which: str = "scan1"):
        assert which in ("scan1", "scan2", "full")
        self.base = base_dataset
        self.pe_dim = pe_dim
        self.which = which

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if self.which == "full":
            return sample
        return apply_split_to_sample(sample, pe_dim=self.pe_dim, which=self.which)


def prepare_data(cfg):
    """
    确定性地构建 dataset / loaders（仅训练用）。

    支持两种划分模式（由 cfg["split_mode"] 控制）：

    1. "timeseries"（默认）
       - 不进行被试层次划分，所有被试用于训练
       - 将每条 BOLD 时间序列二等分，使用 scan1（前半段）进行训练
       - 适用于几乎没有真实重测扫描的临床数据集

    2. "subject"
       - 经典被试层次随机划分（train_ratio），仅使用 train 部分
       - 使用完整时间序列
    """
    set_seed(cfg["seed"])

    brain_dataset = get_dataset("brain", pe_dim=cfg["pe_dim"])
    num_subjects = len(brain_dataset)
    print(f"Number of subjects: {num_subjects}")

    assert num_subjects == cfg["num_subjects_expected"], (
        f"Expected {cfg['num_subjects_expected']} subjects, but got {num_subjects}"
    )

    pe_dim = cfg["pe_dim"]
    split_mode = cfg.get("split_mode", "timeseries")

    if split_mode == "subject":
        # ---------- 被试层次划分（仅取 train 部分） ----------
        indices = list(range(num_subjects))
        random.shuffle(indices)

        train_size = int(cfg["train_ratio"] * num_subjects)
        train_indices = indices[:train_size]

        print("----------------------------------------")
        print("Split mode: SUBJECT (inter-individual)")
        print("----------------------------------------")
        print("Train subjects:", train_indices)

        train_dataset = Subset(brain_dataset, train_indices)
        input_dim = brain_dataset[0]["feature_dim"]

    else:
        # ---------- BOLD 时间序列层次划分（默认） ----------
        # 所有被试使用 scan1（前半段）进行训练
        print("----------------------------------------")
        print("Split mode: TIMESERIES (BOLD scan1 for training)")
        print("----------------------------------------")
        print("All subjects used. Time series bisected: first half = scan1.")

        train_dataset = TimeSeriesSplitDataset(
            brain_dataset, pe_dim=pe_dim, which="scan1"
        )

        sample0 = train_dataset[0]
        input_dim = sample0["feature_dim"]
        original_bold = sample0.get("original_bold_dim", None)
        print(f"Original BOLD dim : {original_bold}")
        print(f"Split BOLD dim    : {input_dim - pe_dim}  (scan1)")
        print(f"PE dim            : {pe_dim}")
        print(f"Model input_dim   : {input_dim}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: x[0],
    )

    print("----------------------------------------")
    print(f"Input feature dimension: {input_dim}")
    print("----------------------------------------")

    return train_loader


# ============================================================
# Checkpoint 存取
# ============================================================
def save_checkpoint(model, cfg, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {k: v for k, v in cfg.items()},
    }
    torch.save(ckpt, path)
    print(f"\n[Checkpoint] Model saved to: {os.path.abspath(path)}")


# ============================================================
# 训练
# ============================================================
def train_brain_branch(args, cfg, device):
    print("========================================")
    print("Brain NAGphormer Training")
    print("========================================")
    print(f"Device      : {device}")
    print(f"K           : {cfg['K']}")
    print(f"PE dim      : {cfg['pe_dim']}")
    print(f"Hidden dim  : {cfg['hidden_dim']}")
    print(f"Num heads   : {cfg['num_heads']}")
    print(f"SSL dim     : {cfg['ssl_dim']}")
    print(f"Epochs      : {cfg['num_epochs']}")
    print(f"Split mode  : {cfg.get('split_mode', 'timeseries')}")
    print(f"Checkpoint  : {args.checkpoint}")
    print("========================================")

    train_loader = prepare_data(cfg)

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    num_epochs = cfg["num_epochs"]
    K = cfg["K"]
    pe_dim = cfg["pe_dim"]

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for step, sample in enumerate(train_loader):
            subject_id = sample["subject"]
            scan_tag = sample.get("scan", "full")

            adj = sample["adj"].to(device)
            features = sample["features"].to(device)

            bold_dim = features.size(1) - pe_dim
            X_raw = features[:, :bold_dim]   # 原始 BOLD（已按模式切分）
            pe = features[:, bold_dim:]      # LPE

            optimizer.zero_grad()

            # 训练只走 MAE 分支即可；forward_mae 内部已经算了 enc/readout
            loss, pred, mask = model.forward_mae(adj, X_raw, pe)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(
                f"Epoch [{epoch + 1:03d}/{num_epochs:03d}] "
                f"| Step [{step + 1}/{len(train_loader)}] "
                f"| Subject: {subject_id} "
                f"| Scan: {scan_tag} "
                f"| Loss: {loss.item():.6f}"
            )

        epoch_loss /= len(train_loader)
        print(f"Epoch [{epoch + 1:03d}] | Mean Loss: {epoch_loss:.6f}")

    # ---- 训练结束：保存 checkpoint ----
    save_checkpoint(model, cfg, args.checkpoint)

    return model


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Brain NAGphormer: training pipeline"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/brain_branch.pt",
        help="Path to save the checkpoint.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEFAULT_CONFIG["num_epochs"],
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG["seed"],
        help="Random seed.",
    )
    parser.add_argument(
        "--split_mode",
        choices=["timeseries", "subject"],
        default=DEFAULT_CONFIG["split_mode"],
        help="Train split strategy. "
             "'timeseries' (default): bisect BOLD time series, use scan1 for training "
             "(suitable for clinical data with almost no real rescans). "
             "'subject': classic subject-level split for inter-individual training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = DEFAULT_CONFIG.copy()
    cfg["num_epochs"] = args.num_epochs
    cfg["seed"] = args.seed
    cfg["split_mode"] = args.split_mode

    train_brain_branch(args, cfg, device)


if __name__ == "__main__":
    # 默认（时间序列二等分，使用 scan1 训练）:
    #   python run.py --num_epochs 10 --checkpoint checkpoints/brain_branch.pt
    #
    # 显式指定时间序列模式:
    #   python run.py --split_mode timeseries ...
    #
    # 被试层次划分:
    #   python run.py --split_mode subject ...
    main()