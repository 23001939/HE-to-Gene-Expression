"""
run_baselines.py -- Pipeline train / predict / eval cho các mô hình baseline trên HER2ST.

Tất cả baseline dùng HER2ST dataset để so sánh fair với LightHGGEP.

Cách dùng:
    python run_baselines.py --mode all          # chạy tất cả 4 baseline liên tiếp
    python run_baselines.py --mode stnet        # chạy riêng 1 model
    python run_baselines.py --mode histogene
    python run_baselines.py --mode uni
    python run_baselines.py --mode wsuni

Tùy chọn:
    --fold        : LOOCV fold (default: 5)
    --n_genes     : số gene dự đoán (default: 785)
    --max_epochs  : số epoch tối đa
    --batch_size  : batch size
    --lr          : learning rate (default: 1e-5)
    --ckpt_dir    : thư mục lưu checkpoint (default: model_ckpts)
    --ckpt_path   : load checkpoint sẵn, bỏ qua train (chỉ dùng khi mode != all)
    --skip_train  : chỉ predict+eval (chỉ dùng khi mode != all)
    --n_gpus      : số GPU dùng (default: 2 nếu có, 1 nếu không)
"""

import argparse
import os
import pathlib
import random
import warnings

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR = str(pathlib.Path(__file__).parent.resolve())
os.chdir(WORKDIR)

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Baseline pipeline cho HER2ST")
parser.add_argument("--mode",       type=str, required=True,
                    choices=["histogene", "stnet", "uni", "all"],
                    help="Model muốn chạy. 'all' = chạy cả 3 model liên tiếp (không bao gồm wsuni).")
parser.add_argument("--fold",       type=int,   default=5)
parser.add_argument("--n_genes",    type=int,   default=785)
parser.add_argument("--max_epochs", type=int,   default=None)
parser.add_argument("--batch_size", type=int,   default=None)
parser.add_argument("--lr",         type=float, default=1e-5)
parser.add_argument("--ckpt_dir",   type=str,   default="model_ckpts")
parser.add_argument("--ckpt_path",  type=str,   default=None,
                    help="Chỉ dùng khi --mode không phải 'all'")
parser.add_argument("--skip_train", action="store_true",
                    help="Chỉ dùng khi --mode không phải 'all'")
parser.add_argument("--n_gpus",     type=int,   default=None,
                    help="Số GPU dùng. Mặc định: dùng hết GPU có sẵn (tối đa 2).")
args = parser.parse_args()

FOLD    = args.fold
N_GENES = args.n_genes
LR      = args.lr

# Số GPU
n_available = torch.cuda.device_count()
if args.n_gpus is not None:
    N_GPUS = min(args.n_gpus, n_available)
else:
    N_GPUS = min(n_available, 2)   # dùng tối đa 2 GPU, tự động detect
N_GPUS = max(N_GPUS, 1)           # ít nhất 1

print("=" * 60)
print(f"BASELINE PIPELINE  mode={args.mode.upper()}  fold={FOLD}")
print("=" * 60)
print(f"  GPU available : {n_available}  →  dùng {N_GPUS} GPU")
print(f"  n_genes       : {N_GENES}")
print(f"  lr            : {LR}")
print("=" * 60)

# ── Imports chung ─────────────────────────────────────────────────────────────
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import CSVLogger

from dataset import HER2ST
from utils import comp_tsne_km
from predict import (
    get_R, get_MSE, get_MAE,
    get_Spearman, get_MoransI_all, cluster_with_nmi,
    stnet_predict, histogene_predict,
)

# ── Callback: log mỗi epoch ra stdout ────────────────────────────────────────
class EpochProgressBar(Callback):
    """In 1 dòng tóm tắt sau mỗi epoch: train_loss, val_loss, lr, thời gian."""
    def on_train_epoch_end(self, trainer, pl_module):
        m        = trainer.callback_metrics
        ep       = trainer.current_epoch + 1
        total    = trainer.max_epochs
        t_loss   = m.get("train_loss_epoch", m.get("train_loss", float("nan")))
        v_loss   = m.get("val_loss",  m.get("valid_loss", float("nan")))
        opt      = pl_module.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        lr = opt.param_groups[0]["lr"]
        # Thời gian epoch
        elapsed = trainer.fit_loop.epoch_loop.batch_progress.total.completed
        print(
            f"[{trainer.logger.name}] "
            f"Epoch {ep:3d}/{total}  "
            f"train_loss={float(t_loss):.4f}  "
            f"val_loss={float(v_loss):.4f}  "
            f"lr={lr:.2e}",
            flush=True,
        )

# ── Helpers ───────────────────────────────────────────────────────────────────
_default_epochs = {"histogene": 100, "stnet": 100, "uni": 50, "wsuni": 50}
_default_bs     = {"histogene": 1,   "stnet": 1,   "uni": 16, "wsuni": 16}

def split_train_val(ds_aug, ds_noaug):
    """
    Tách slide đầu alphabet làm val, còn lại làm train.
    ds_aug   : HER2ST(train=True)  → có augmentation → train_loader
    ds_noaug : HER2ST(train=True) với .train=False → không augment → val_loader
    """
    val_name  = sorted(ds_aug.names)[0]
    name2idx  = {name: i for i, name in ds_aug.id2name.items()}
    val_i     = name2idx[val_name]
    val_start = int(ds_aug.cumlen[val_i - 1]) if val_i > 0 else 0
    val_end   = int(ds_aug.cumlen[val_i])
    val_idx   = list(range(val_start, val_end))
    train_idx = [i for i in range(len(ds_aug)) if i not in set(val_idx)]
    print(f"  Val slide : {val_name} ({len(val_idx)} spots) | "
          f"Train spots: {len(train_idx)}")
    return Subset(ds_aug, train_idx), Subset(ds_noaug, val_idx)


def collate_drop_center(batch):
    """
    HER2ST(train=False) trả về 4 phần tử (patch, loc, exp, center).
    validation_step của HisToGene/STNet unpack 3 phần tử → drop center.
    """
    return default_collate([item[:3] for item in batch])


# ─────────────────────────────────────────────────────────────────────────────
# HÀM CHÍNH: chạy train + predict + eval cho 1 mode
# ─────────────────────────────────────────────────────────────────────────────
def run_one(mode, fold, n_genes, lr, max_epochs, batch_size,
            ckpt_dir, ckpt_path, skip_train, n_gpus):

    from models.HisToGene_model import HisToGene
    from models.STNet_model import STModel
    try:
        from models.UNI import UNI
    except ImportError:
        UNI = None

    max_ep = max_epochs if max_epochs is not None else _default_epochs[mode]
    bs     = batch_size if batch_size is not None else _default_bs[mode]
    ckpt_out_dir = os.path.join(ckpt_dir, mode)
    os.makedirs(ckpt_out_dir, exist_ok=True)
    os.makedirs("figures/kmeans", exist_ok=True)
    os.makedirs("figures/FASN",   exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MODEL : {mode.upper()}")
    print(f"  epochs={max_ep}  batch={bs}  lr={lr}  gpus={n_gpus}")
    print(f"{'='*60}")

    # ── Strategy cho multi-GPU ────────────────────────────────────────────────
    # DataParallel (dp): chạy trên 1 process, chia batch sang các GPU.
    # Đơn giản, không cần spawn, không conflict với num_workers=0.
    # DDP sẽ nhanh hơn nhưng cần multi-process → phức tạp hơn khi chạy từ script.
    if n_gpus > 1:
        strategy = "dp"
        accelerator = "gpu"
        devices = n_gpus
    elif torch.cuda.is_available():
        strategy = "auto"
        accelerator = "gpu"
        devices = 1
    else:
        strategy = "auto"
        accelerator = "cpu"
        devices = 1

    # ── Monitor key ───────────────────────────────────────────────────────────
    _monitor = {
        "histogene": "val_loss",
        "stnet":     "valid_loss",
        "uni":       "val_loss",
        "wsuni":     "val_loss",
    }
    monitor = _monitor[mode]

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    if not skip_train:
        logger = CSVLogger("logs", name=f"baseline_{mode}")

        checkpoint_cb = ModelCheckpoint(
            dirpath=ckpt_out_dir,
            filename=f"{mode}_fold{fold}_" + "{epoch:02d}",
            save_top_k=3,
            monitor=monitor,
            mode="min",
            save_last=True,
        )
        early_stop_cb = EarlyStopping(
            monitor=monitor,
            patience=15,
            mode="min",
            verbose=False,
        )

        ds_aug   = HER2ST(train=True, fold=fold)
        ds_noaug = HER2ST(train=True, fold=fold)
        ds_noaug.train = False
        train_subset, val_subset = split_train_val(ds_aug, ds_noaug)

        train_loader = DataLoader(train_subset, batch_size=bs,
                                  num_workers=0, shuffle=True)
        val_loader   = DataLoader(val_subset,   batch_size=bs,
                                  num_workers=0, shuffle=False,
                                  collate_fn=collate_drop_center)

        if mode == "histogene":
            model = HisToGene(n_layers=8, n_genes=n_genes, learning_rate=lr)
        elif mode == "stnet":
            model = STModel(n_genes=n_genes, learning_rate=lr)
        elif mode == "uni":
            if UNI is None:
                print("[SKIP] models.UNI không import được.")
                return None
            model = UNI(n_genes=n_genes, learning_rate=lr, max_epochs=max_ep)
            model.enable_lora_training()
        elif mode == "wsuni":
            if WSUNI is None:
                print("[SKIP] models.WSUNI không import được.")
                return None
            model = WSUNI(n_genes=n_genes, learning_rate=lr, max_epochs=max_ep)

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            max_epochs=max_ep,
            logger=logger,
            log_every_n_steps=10,
            gradient_clip_val=1.0,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[checkpoint_cb, early_stop_cb, EpochProgressBar()],
        )
        trainer.fit(model, train_loader, val_loader)
        ckpt_path = checkpoint_cb.best_model_path
        print(f"  Best checkpoint: {ckpt_path}")
        print(f"  Best val loss  : {checkpoint_callback.best_model_score:.4f}"
              if hasattr(checkpoint_cb, "best_model_score") else "")

    else:
        if ckpt_path is None:
            raise ValueError(f"--skip_train yêu cầu --ckpt_path cho mode={mode}")

    # ── PREDICT ───────────────────────────────────────────────────────────────
    print(f"\n  [PREDICT] Load: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = HER2ST(train=False, fold=fold)
    label        = test_dataset.label[test_dataset.names[0]]

    if mode == "histogene":
        m = HisToGene.load_from_checkpoint(
            ckpt_path, n_layers=8, n_genes=n_genes, learning_rate=lr)
        test_loader = DataLoader(test_dataset, batch_size=1,
                                 num_workers=0, shuffle=False)
        adata_pred, adata_gt = histogene_predict(m, test_loader, device=device)

    elif mode == "stnet":
        m = STModel.load_from_checkpoint(
            ckpt_path, n_genes=n_genes, learning_rate=lr)
        test_loader = DataLoader(test_dataset, batch_size=bs,
                                 num_workers=0, shuffle=False)
        adata_pred, adata_gt = stnet_predict(m, test_loader, device=device)

    elif mode == "uni":
        m = UNI.load_from_checkpoint(
            ckpt_path, n_genes=n_genes, learning_rate=lr, max_epochs=max_ep)
        test_loader = DataLoader(test_dataset, batch_size=bs,
                                 num_workers=0, shuffle=False)
        adata_pred, adata_gt = stnet_predict(m, test_loader, device=device)

    elif mode == "wsuni":
        m = WSUNI.load_from_checkpoint(
            ckpt_path, n_genes=n_genes, learning_rate=lr, max_epochs=max_ep)
        test_loader = DataLoader(test_dataset, batch_size=bs,
                                 num_workers=0, shuffle=False)
        adata_pred, adata_gt = stnet_predict(m, test_loader, device=device)

    # ── Post-processing ───────────────────────────────────────────────────────
    g = list(np.load("data/her_hvg_cut_1000.npy", allow_pickle=True))
    adata_pred.var_names = g
    sc.pp.scale(adata_pred)
    adata_pred = comp_tsne_km(adata_pred, 4)

    # ── METRICS ───────────────────────────────────────────────────────────────
    print(f"\n  [EVAL] {mode.upper()} fold={fold}")
    R,        p_values         = get_R(adata_pred, adata_gt)
    Spearman, spearman_pvalues = get_Spearman(adata_pred, adata_gt)
    MSE                        = get_MSE(adata_pred, adata_gt)
    MAE                        = get_MAE(adata_pred, adata_gt)
    RMSE                       = np.sqrt(MSE)
    morans                     = get_MoransI_all(adata_pred, adata_gt, top_k=50)

    mean_pcc      = np.nanmean(R)
    median_pcc    = np.nanmedian(R)
    mean_spearman = np.nanmean(Spearman)
    mean_rmse     = np.nanmean(RMSE)
    mean_mae      = np.nanmean(MAE)
    mean_mi_pred  = np.nanmean(morans["pred"])
    mean_mi_gt    = np.nanmean(morans["gt"])

    if label is not None:
        _, ARI, NMI = cluster_with_nmi(adata_pred, label)
    else:
        ARI = NMI = float("nan")

    print(f"  PCC={mean_pcc:.4f}  Spearman={mean_spearman:.4f}  "
          f"ARI={ARI:.4f}  NMI={NMI:.4f}")
    print(f"  RMSE={mean_rmse:.4f}  MAE={mean_mae:.4f}")
    print(f"  Moran's I pred={mean_mi_pred:.4f}  gt={mean_mi_gt:.4f}")

    # ── VISUALIZE ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(R, bins=30, color="tab:blue", alpha=0.75, edgecolor="black")
    ax.axvline(mean_pcc,   color="red",  ls="--", lw=2,
               label=f"Mean={mean_pcc:.3f}")
    ax.axvline(median_pcc, color="blue", ls="-.", lw=2,
               label=f"Median={median_pcc:.3f}")
    ax.set_xlabel("PCC"); ax.set_ylabel("Gene count")
    ax.set_title(f"{mode.upper()} fold{fold} PCC distribution")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/{mode.upper()}_PCC_fold{fold}.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    sc.pl.spatial(adata_pred, img=None, color="kmeans", spot_size=112,
                  frameon=False, legend_loc=None, title=None, show=False)
    plt.gca().set_title("")
    plt.savefig(f"figures/kmeans/{mode.upper()}_kmeans_fold{fold}.png",
                dpi=300, bbox_inches="tight", transparent=True)
    plt.clf(); plt.close()

    sc.pl.spatial(adata_pred, img=None, color="FASN", spot_size=112,
                  color_map="magma", frameon=False, legend_loc=None,
                  title=None, show=False)
    plt.gca().set_title("")
    plt.savefig(f"figures/FASN/{mode.upper()}_FASN_fold{fold}.png",
                dpi=300, bbox_inches="tight", transparent=True)
    plt.clf(); plt.close()

    # ── LƯU KẾT QUẢ ──────────────────────────────────────────────────────────
    gene_stats = pd.DataFrame({
        "gene":          g,
        "pcc":           R,
        "pcc_pvalue":    p_values,
        "spearman":      Spearman,
        "spearman_pval": spearman_pvalues,
        "mse":           MSE,
        "rmse":          RMSE,
        "mae":           MAE,
    })
    gene_stats.to_csv(f"{mode}_gene_stats_fold{fold}.csv", index=False)

    total_params = sum(p.numel() for p in m.parameters())
    result = {
        "model":         mode.upper(),
        "fold":          fold,
        "pearson":       mean_pcc,
        "spearman":      mean_spearman,
        "ari":           ARI,
        "nmi":           NMI,
        "rmse":          mean_rmse,
        "mae":           mean_mae,
        "morans_i_pred": mean_mi_pred,
        "morans_i_gt":   mean_mi_gt,
        "params":        total_params,
        "ckpt":          ckpt_path,
    }

    summary_csv = "baselines_results.csv"
    new_row = pd.DataFrame([result])
    if os.path.isfile(summary_csv):
        existing = pd.read_csv(summary_csv)
        existing = existing[~((existing["model"] == mode.upper()) &
                               (existing["fold"]  == fold))]
        summary = pd.concat([existing, new_row], ignore_index=True)
    else:
        summary = new_row
    summary.to_csv(summary_csv, index=False)
    print(f"  Saved summary → {summary_csv}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
ALL_MODES = ["histogene", "stnet", "uni"]   # wsuni bị loại vì không tương thích HER2ST raw image
modes_to_run = ALL_MODES if args.mode == "all" else [args.mode]

all_results = []
for mode in modes_to_run:
    max_ep = args.max_epochs
    bs     = args.batch_size
    # Khi chạy all, ckpt_path và skip_train không áp dụng
    ckpt_p     = args.ckpt_path  if args.mode != "all" else None
    skip_train = args.skip_train if args.mode != "all" else False

    result = run_one(
        mode       = mode,
        fold       = FOLD,
        n_genes    = N_GENES,
        lr         = LR,
        max_epochs = max_ep,
        batch_size = bs,
        ckpt_dir   = args.ckpt_dir,
        ckpt_path  = ckpt_p,
        skip_train = skip_train,
        n_gpus     = N_GPUS,
    )
    if result is not None:
        all_results.append(result)

# ── Bảng tổng kết cuối ───────────────────────────────────────────────────────
if all_results:
    df = pd.DataFrame(all_results)
    print(f"\n{'='*70}")
    print("TỔNG KẾT TẤT CẢ BASELINE")
    print(f"{'='*70}")
    cols = ["model", "pearson", "spearman", "ari", "nmi", "rmse", "mae",
            "morans_i_pred", "params"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].sort_values("pearson", ascending=False).to_string(index=False))
    print(f"{'='*70}")
