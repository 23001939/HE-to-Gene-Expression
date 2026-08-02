"""
run_baselines.py -- Pipeline train / predict / eval cho các mô hình baseline trên HER2ST.

Baseline được hỗ trợ:
    - histogene   : HisToGene  (ViT-based, patch-level)
    - stnet       : ST-Net     (ResNet-based, patch-level)
    - uni         : UNI        (ViT foundation model + LoRA, patch-level)
    - wsuni       : WSUNI      (Whole-slide UNI + adaptive K-NN)

Cách dùng:
    python run_baselines.py --mode histogene
    python run_baselines.py --mode stnet
    python run_baselines.py --mode uni
    python run_baselines.py --mode wsuni

Tùy chọn:
    --fold        : LOOCV fold (default: 5)
    --n_genes     : số gene dự đoán (default: 785)
    --max_epochs  : số epoch tối đa (default: 50 cho UNI/WSUNI, 100 cho HisToGene/STNet)
    --batch_size  : batch size (default: theo từng model)
    --lr          : learning rate (default: 1e-5)
    --ckpt_dir    : thư mục lưu checkpoint (default: model_ckpts)
    --ckpt_path   : đường dẫn checkpoint để predict (bỏ qua train nếu được cung cấp)
    --skip_train  : chỉ chạy predict+eval, bỏ qua train (cần --ckpt_path)
    --topk        : top-K láng giềng cho WSUNI (default: 40)
    --cache_dir_train : thư mục cache feature train cho WSUNI (default: cache_features_train)
    --cache_dir_test  : thư mục cache feature test  cho WSUNI (default: cache_features_test_saved)

File này giữ đúng logic train/predict của ST_train.py + ST_predict.py gốc,
chỉ chuẩn hoá lại cấu trúc, thêm argparse, bộ metric đầy đủ (PCC, Spearman,
RMSE, MAE, Moran's I, ARI, NMI) và lưu kết quả ra CSV để so sánh với LightHGGEP.
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

# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Working directory (giống run_pipeline.py) ────────────────────────────────
if os.path.isdir("/kaggle/working"):
    WORKDIR = "/kaggle/working"
else:
    WORKDIR = str(pathlib.Path(__file__).parent.resolve())
os.chdir(WORKDIR)

# ── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Baseline pipeline cho HER2ST")
parser.add_argument("--mode",       type=str,   required=True,
                    choices=["histogene", "stnet", "uni", "wsuni"],
                    help="Tên baseline model")
parser.add_argument("--fold",       type=int,   default=5)
parser.add_argument("--n_genes",    type=int,   default=785)
parser.add_argument("--max_epochs", type=int,   default=None,
                    help="Mặc định: 100 (histogene/stnet), 50 (uni/wsuni)")
parser.add_argument("--batch_size", type=int,   default=None,
                    help="Mặc định: 1 (histogene/stnet/ws_uni), 16 (uni/wsuni)")
parser.add_argument("--lr",         type=float, default=1e-5)
parser.add_argument("--ckpt_dir",   type=str,   default="model_ckpts")
parser.add_argument("--ckpt_path",  type=str,   default=None,
                    help="Load checkpoint sẵn, bỏ qua train")
parser.add_argument("--skip_train", action="store_true",
                    help="Bỏ qua bước train, chỉ predict+eval")
parser.add_argument("--topk",       type=int,   default=40,
                    help="Top-K láng giềng cho WSUNI")
parser.add_argument("--cache_dir_train", type=str, default="cache_features_train")
parser.add_argument("--cache_dir_test",  type=str, default="cache_features_test_saved")
args = parser.parse_args()

MODE = args.mode
FOLD = args.fold
N_GENES = args.n_genes
LR = args.lr
CKPT_DIR = os.path.join(args.ckpt_dir, MODE)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs("figures/kmeans", exist_ok=True)
os.makedirs("figures/FASN",   exist_ok=True)

# Defaults theo từng model
_default_epochs = {"histogene": 100, "stnet": 100, "uni": 50, "wsuni": 50}
_default_bs     = {"histogene": 1,   "stnet": 1,   "uni": 16, "wsuni": 16}
MAX_EPOCHS  = args.max_epochs  if args.max_epochs  is not None else _default_epochs[MODE]
BATCH_SIZE  = args.batch_size  if args.batch_size  is not None else _default_bs[MODE]

print("=" * 60)
print(f"BASELINE PIPELINE: {MODE.upper()}")
print("=" * 60)
print(f"  fold        = {FOLD}")
print(f"  n_genes     = {N_GENES}")
print(f"  max_epochs  = {MAX_EPOCHS}")
print(f"  batch_size  = {BATCH_SIZE}")
print(f"  lr          = {LR}")
print(f"  ckpt_dir    = {CKPT_DIR}")
print(f"  skip_train  = {args.skip_train}")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ── Imports theo mode ─────────────────────────────────────────────────────────
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from utils import comp_tsne_km
from predict import (
    get_R, get_MSE, get_MAE,
    get_Spearman, get_MoransI_all, cluster_with_nmi,
)

if MODE == "histogene":
    from dataset import ViT_HER2ST
    from models.HisToGene_model import HisToGene
    from predict import model_predict

elif MODE == "stnet":
    from dataset import HER2ST
    from models.STNet_model import STModel
    from predict import model_predict

elif MODE == "uni":
    from dataset import UNI_HER2ST
    from models.UNI import UNI
    from predict import uni_predict

elif MODE == "wsuni":
    from dataset import WSUNI_HER2ST
    from models.WSUNI import WSUNI
    from predict import wsuni_predict

# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 1: TRAIN
# ─────────────────────────────────────────────────────────────────────────────
ckpt_path = args.ckpt_path

if not args.skip_train:
    print(f"\n{'='*60}")
    print("BƯỚC 1: TRAIN")
    print(f"{'='*60}")

    logger = CSVLogger("logs", name=f"baseline_{MODE}")

    checkpoint_cb = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename=f"{MODE}_fold{FOLD}_" + "{epoch:02d}_{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True,
    )

    # ── Shared trainer kwargs (áp dụng cho tất cả mode) ──────────────────────
    # gradient_clip_val=1.0 đồng nhất với run_pipeline.py (LightHGGEP).
    trainer_kwargs = dict(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    # ── Helper: tách val subset từ train dataset ──────────────────────────────
    # Giống chiến lược của LightHGGEP: lấy slide đầu tiên theo alphabet của
    # tập train làm validation -- KHÔNG dùng test slide, tránh data leakage.
    #
    # Vấn đề với Subset(full_train, val_idx):
    #   1. full_train.train=True → augmentation vẫn áp dụng trên val → val loss
    #      noisy → EarlyStopping kém ổn định.
    #   2. dataset.names = list(set(...)) có thứ tự không xác định → val_i
    #      trỏ sai slide nếu chỉ dùng index.
    # Giải pháp: load 2 bản dataset cùng fold -- 1 bản train=True (augment,
    # dùng làm train), 1 bản train=True nhưng PATCH qua wrapper tắt augment
    # (dùng làm val). Slide val được xác định bằng TÊN, không phải index.
    from torch.utils.data import Subset
    from torch.utils.data.dataloader import default_collate

    def split_train_val(dataset_train, dataset_noaug):
        """
        Trả về (train_subset, val_subset) từ 2 bản dataset cùng fold:
          - dataset_train  : train=True (có augmentation) → dùng cho train_loader
          - dataset_noaug  : train=True nhưng augmentation đã bị disable → val_loader
        Val slide = slide đầu tiên theo alphabet trong tập train.
        Index được tính từ cumlen theo TÊN để tránh lỗi thứ tự set().
        """
        # Tên slide val (alphabet) -- cả 2 dataset phải có cùng names
        val_name = sorted(dataset_train.names)[0]

        # Tính index theo tên, không theo vị trí list (tránh set-order bug)
        name2idx = {name: i for i, name in dataset_train.id2name.items()}
        val_i     = name2idx[val_name]
        val_start = int(dataset_train.cumlen[val_i - 1]) if val_i > 0 else 0
        val_end   = int(dataset_train.cumlen[val_i])
        val_idx   = list(range(val_start, val_end))
        train_idx = [i for i in range(len(dataset_train)) if i not in set(val_idx)]

        print(f"  Val slide : {val_name} ({len(val_idx)} spots) | "
              f"Train spots: {len(train_idx)}")

        return Subset(dataset_train, train_idx), Subset(dataset_noaug, val_idx)

    # collate_fn cho val_loader của HisToGene / STModel:
    # dataset.train=False trả về (patch, loc, exp, center) -- 4 phần tử,
    # nhưng validation_step của 2 model này unpack (patch, loc, exp) -- 3 phần tử.
    # Drop phần tử thứ 4 (center) để khớp format train.
    def collate_drop_center(batch):
        return default_collate([item[:3] for item in batch])

    if MODE == "histogene":
        ds_aug    = ViT_HER2ST(train=True, fold=FOLD)
        ds_noaug  = ViT_HER2ST(train=True, fold=FOLD)
        ds_noaug.train = False   # tắt augmentation
        train_subset, val_subset = split_train_val(ds_aug, ds_noaug)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                                  num_workers=4, shuffle=True)
        val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE,
                                  num_workers=4, shuffle=False,
                                  collate_fn=collate_drop_center)
        model = HisToGene(n_layers=8, n_genes=N_GENES, learning_rate=LR)

        trainer = pl.Trainer(
            callbacks=[checkpoint_cb, early_stop_cb],
            **trainer_kwargs,
        )
        trainer.fit(model, train_loader, val_loader)
        ckpt_path = checkpoint_cb.best_model_path
        print(f"Best checkpoint: {ckpt_path}")

    elif MODE == "stnet":
        ds_aug   = HER2ST(train=True, fold=FOLD)
        ds_noaug = HER2ST(train=True, fold=FOLD)
        ds_noaug.train = False
        train_subset, val_subset = split_train_val(ds_aug, ds_noaug)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                                  num_workers=4, shuffle=True)
        val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE,
                                  num_workers=4, shuffle=False,
                                  collate_fn=collate_drop_center)
        model = STModel(n_genes=N_GENES, learning_rate=LR)

        trainer = pl.Trainer(
            callbacks=[checkpoint_cb, early_stop_cb],
            **trainer_kwargs,
        )
        trainer.fit(model, train_loader, val_loader)
        ckpt_path = checkpoint_cb.best_model_path
        print(f"Best checkpoint: {ckpt_path}")

    elif MODE == "uni":
        ds_aug   = UNI_HER2ST(train=True, fold=FOLD)
        ds_noaug = UNI_HER2ST(train=True, fold=FOLD)
        ds_noaug.train = False
        train_subset, val_subset = split_train_val(ds_aug, ds_noaug)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                                  num_workers=1, shuffle=True)
        val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE,
                                  num_workers=1, shuffle=False)
        model = UNI(n_genes=N_GENES, learning_rate=LR, max_epochs=MAX_EPOCHS)
        model.enable_lora_training()

        trainer = pl.Trainer(
            callbacks=[checkpoint_cb, early_stop_cb],
            **trainer_kwargs,
        )
        trainer.fit(model, train_loader, val_loader)
        ckpt_path = checkpoint_cb.best_model_path
        print(f"\nBest checkpoint: {ckpt_path}")

    elif MODE == "wsuni":
        ds_aug = WSUNI_HER2ST(
            train=True, fold=FOLD,
            cache_dir=args.cache_dir_train, topk=args.topk,
        )
        ds_noaug = WSUNI_HER2ST(
            train=True, fold=FOLD,
            cache_dir=args.cache_dir_train, topk=args.topk,
        )
        ds_noaug.train = False
        train_subset, val_subset = split_train_val(ds_aug, ds_noaug)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                                  num_workers=1, shuffle=True)
        val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE,
                                  num_workers=1, shuffle=False)
        model = WSUNI(n_genes=N_GENES, learning_rate=LR, max_epochs=MAX_EPOCHS)

        trainer = pl.Trainer(
            callbacks=[checkpoint_cb, early_stop_cb],
            **trainer_kwargs,
        )
        trainer.fit(model, train_loader, val_loader)
        ckpt_path = checkpoint_cb.best_model_path
        print(f"\nBest checkpoint: {ckpt_path}")

else:
    print("\n[skip_train=True] Bỏ qua bước train.")
    if ckpt_path is None:
        raise ValueError("Khi --skip_train, phải cung cấp --ckpt_path.")

# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 2: PREDICT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("BƯỚC 2: PREDICT")
print(f"{'='*60}")
print(f"Load checkpoint: {ckpt_path}")

if MODE == "histogene":
    model = HisToGene.load_from_checkpoint(
        ckpt_path, n_layers=8, n_genes=N_GENES, learning_rate=LR,
    )
    test_dataset = ViT_HER2ST(train=False, fold=FOLD)
    test_loader  = DataLoader(test_dataset, batch_size=1, num_workers=4)
    label        = test_dataset.label[test_dataset.names[0]]
    adata_pred, adata_gt = model_predict(model, test_loader,
                                         attention=False, device=device)

elif MODE == "stnet":
    model = STModel.load_from_checkpoint(
        ckpt_path, n_genes=N_GENES, learning_rate=LR,
    )
    test_dataset = HER2ST(train=False, fold=FOLD)
    test_loader  = DataLoader(test_dataset, batch_size=1, num_workers=4)
    label        = test_dataset.label[test_dataset.names[0]]
    adata_pred, adata_gt = model_predict(model, test_loader,
                                         attention=False, device=device)

elif MODE == "uni":
    model = UNI.load_from_checkpoint(
        ckpt_path, n_genes=N_GENES, learning_rate=LR, max_epochs=MAX_EPOCHS,
    )
    test_dataset = UNI_HER2ST(train=False, fold=FOLD)
    test_loader  = DataLoader(test_dataset, batch_size=1, num_workers=1)
    label        = test_dataset.label[test_dataset.names[0]]
    adata_pred, adata_gt = uni_predict(model, test_loader,
                                       attention=False, device=device)

elif MODE == "wsuni":
    model = WSUNI.load_from_checkpoint(
        ckpt_path, n_genes=N_GENES, learning_rate=LR, max_epochs=MAX_EPOCHS,
    )
    test_dataset = WSUNI_HER2ST(
        train=False, fold=FOLD,
        cache_dir=args.cache_dir_test, topk=args.topk,
    )
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=1)
    label        = test_dataset.label[test_dataset.names[0]]
    adata_pred, adata_gt = wsuni_predict(model, test_loader,
                                         attention=False, device=device)

# ── Post-processing (cùng thứ tự với run_pipeline.py đã sửa) ─────────────────
g = list(np.load("data/her_hvg_cut_1000.npy", allow_pickle=True))
adata_pred.var_names = g
sc.pp.scale(adata_pred)            # scale TRƯỚC cluster để nhất quán
adata_pred = comp_tsne_km(adata_pred, 4)

# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 3: ĐÁNH GIÁ (cùng bộ metric với LightHGGEP)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("BƯỚC 3: ĐÁNH GIÁ")
print(f"{'='*60}")

R,        p_values         = get_R(adata_pred, adata_gt)
Spearman, spearman_pvalues = get_Spearman(adata_pred, adata_gt)
MSE                        = get_MSE(adata_pred, adata_gt)
MAE                        = get_MAE(adata_pred, adata_gt)
RMSE                       = np.sqrt(MSE)
morans                     = get_MoransI_all(adata_pred, adata_gt, top_k=50)

mean_pcc      = np.nanmean(R)
median_pcc    = np.nanmedian(R)
std_pcc       = np.nanstd(R)
mean_spearman = np.nanmean(Spearman)
mean_rmse     = np.nanmean(RMSE)
mean_mae      = np.nanmean(MAE)
mean_mi_pred  = np.nanmean(morans["pred"])
mean_mi_gt    = np.nanmean(morans["gt"])

n_spots = adata_pred.shape[0]

print(f"  Model                      : {MODE.upper()}")
print(f"  Fold                       : {FOLD}")
print(f"  Số spot test               : {n_spots}")
print(f"  Số gene                    : {len(R)}")
print()
print(f"  [Correlation]")
print(f"  Mean Gene-wise PCC         : {mean_pcc:.4f}")
print(f"  Median Gene-wise PCC       : {median_pcc:.4f}")
print(f"  Std Gene-wise PCC          : {std_pcc:.4f}")
print(f"  Mean Gene-wise Spearman    : {mean_spearman:.4f}")
print()
print(f"  [Error]")
print(f"  Mean RMSE                  : {mean_rmse:.4f}")
print(f"  Mean MAE                   : {mean_mae:.4f}")
print()
print(f"  [Spatial structure - top-50 high-var genes]")
print(f"  Mean Moran's I (pred)      : {mean_mi_pred:.4f}")
print(f"  Mean Moran's I (gt)        : {mean_mi_gt:.4f}")

# ARI + NMI
if label is not None:
    clus, ARI, NMI = cluster_with_nmi(adata_pred, label)
    print()
    print(f"  [Global structure]")
    print(f"  ARI (Adjusted Rand Index)  : {ARI:.4f}")
    print(f"  NMI (Norm. Mutual Info)    : {NMI:.4f}")
else:
    ARI = float("nan")
    NMI = float("nan")
    print("\nARI/NMI: N/A (section này không có ground-truth label)")

print("=" * 60)

# ── Top/Bottom gene stats ─────────────────────────────────────────────────────
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

print(f"\nTop-10 gen TỐT NHẤT (PCC cao nhất):")
print("  " + gene_stats.nlargest(10, "pcc")[["gene", "pcc", "spearman"]]
      .to_string(index=False).replace("\n", "\n  "))

print(f"\nTop-10 gen KÉM NHẤT (PCC thấp nhất):")
print("  " + gene_stats.nsmallest(10, "pcc")[["gene", "pcc", "spearman"]]
      .to_string(index=False).replace("\n", "\n  "))

# ── Thống kê PCC ─────────────────────────────────────────────────────────────
print(f"\n  Số gene có PCC > 0  : {np.sum(R > 0):,}/{len(R)} ({100*np.sum(R > 0)/len(R):.1f}%)")
print(f"  Số gene có PCC > 0.2: {np.sum(R > 0.2):,}/{len(R)} ({100*np.sum(R > 0.2)/len(R):.1f}%)")
print(f"  Số gene có PCC > 0.3: {np.sum(R > 0.3):,}/{len(R)} ({100*np.sum(R > 0.3)/len(R):.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 4: VISUALIZE
# ─────────────────────────────────────────────────────────────────────────────
# Histogram PCC
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(R, bins=30, color="tab:blue", alpha=0.75, edgecolor="black")
ax.axvline(mean_pcc,   color="red",  linestyle="--", linewidth=2,
           label=f"Mean PCC = {mean_pcc:.3f}")
ax.axvline(median_pcc, color="blue", linestyle="-.", linewidth=2,
           label=f"Median PCC = {median_pcc:.3f}")
ax.set_xlabel("PCC (Pearson Correlation Coefficient)")
ax.set_ylabel("Số lượng gen")
ax.set_title(f"PCC distribution - {MODE.upper()} fold{FOLD} ({len(R)} genes)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
pcc_fig = f"figures/{MODE.upper()}_PCC_distribution_fold{FOLD}.png"
plt.savefig(pcc_fig, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nĐã lưu PCC distribution → {pcc_fig}")

# K-means spatial
sc.pl.spatial(adata_pred, img=None, color="kmeans", spot_size=112,
              frameon=False, legend_loc=None, title=None, show=False)
plt.gca().set_title("")
kmeans_fig = f"figures/kmeans/{MODE.upper()}_kmeans_fold{FOLD}.png"
plt.savefig(kmeans_fig, dpi=300, bbox_inches="tight", transparent=True)
plt.clf(); plt.close()
print(f"Saved: {kmeans_fig}")

# FASN expression
sc.pl.spatial(adata_pred, img=None, color="FASN", spot_size=112,
              color_map="magma", frameon=False, legend_loc=None, title=None,
              show=False)
plt.gca().set_title("")
fasn_fig = f"figures/FASN/{MODE.upper()}_FASN_fold{FOLD}.png"
plt.savefig(fasn_fig, dpi=300, bbox_inches="tight", transparent=True)
plt.clf(); plt.close()
print(f"Saved: {fasn_fig}")

# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 5: LƯU KẾT QUẢ
# ─────────────────────────────────────────────────────────────────────────────
# Per-gene stats
gene_csv = f"{MODE}_gene_stats_fold{FOLD}.csv"
gene_stats.to_csv(gene_csv, index=False)
print(f"\nĐã lưu per-gene stats → {gene_csv}")

# Summary (có thể append nhiều fold/model để so sánh)
summary_csv = "baselines_results.csv"
total_params = sum(p.numel() for p in model.parameters())

new_row = pd.DataFrame([{
    "model":          MODE.upper(),
    "fold":           FOLD,
    "pearson":        mean_pcc,
    "spearman":       mean_spearman,
    "ari":            ARI,
    "nmi":            NMI,
    "rmse":           mean_rmse,
    "mae":            mean_mae,
    "morans_i_pred":  mean_mi_pred,
    "morans_i_gt":    mean_mi_gt,
    "params":         total_params,
    "ckpt":           ckpt_path,
}])

if os.path.isfile(summary_csv):
    existing = pd.read_csv(summary_csv)
    # Xoá dòng cũ nếu cùng model+fold để tránh trùng lặp khi chạy lại
    existing = existing[~((existing["model"] == MODE.upper()) &
                           (existing["fold"]  == FOLD))]
    summary = pd.concat([existing, new_row], ignore_index=True)
else:
    summary = new_row

summary.to_csv(summary_csv, index=False)
print(f"Đã lưu/cập nhật summary → {summary_csv}")

print(f"\n{'='*60}")
print(f"HOÀN TẤT: {MODE.upper()} fold={FOLD}")
print(f"  PCC={mean_pcc:.4f}  Spearman={mean_spearman:.4f}  "
      f"ARI={ARI:.4f}  NMI={NMI:.4f}")
print(f"  RMSE={mean_rmse:.4f}  MAE={mean_mae:.4f}")
print(f"  Moran's I pred={mean_mi_pred:.4f}  gt={mean_mi_gt:.4f}")
print(f"{'='*60}")
