"""

run_pipeline.py -- Toàn bộ pipeline train/predict/eval/visualize Light-HGGEP trên HER2ST.

File này được TÁCH RA NGUYÊN VẸN từ các cell code của LightHGGEP.ipynb (PHẦN 1, 2, 4, 5,
6, 7, 8, 9 -- KHÔNG bao gồm PHẦN 3, vốn đã tách thành utils.py / predict.py / dataset.py /
models/LightHGGEP.py / models/__init__.py riêng), giữ NGUYÊN THỨ TỰ và NỘI DUNG từng cell.

Thay đổi DUY NHẤT so với notebook gốc: các dòng lệnh IPython bắt đầu bằng "!" (không phải
cú pháp Python hợp lệ trong file .py) được dịch sang subprocess.run(..., shell=True) --
CÙNG một câu lệnh shell, cùng hành vi, không đổi logic. Mỗi vị trí dịch đều có comment
"[DỊCH TỪ IPYTHON]" đánh dấu, kèm câu lệnh gốc để đối chiếu.

Cách chạy: xem notebook mỏng đi kèm (chỉ gồm !pip install + !python run_pipeline.py).
"""
import subprocess  # [MỚI - chỉ để dịch các dòng "!..." của notebook, xem docstring trên]


# ============================================================================
# ---- Cell 3 (notebook gốc) ----
# ============================================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================================
# ---- Cell 4 (notebook gốc) ----
# ============================================================================
# ============================================================
# (Tuy chon) Dang nhap Weights & Biases
# ============================================================
USE_WANDB = False  # doi thanh True neu ban muon dung W&B cua rieng minh

wandb_logger = None
if USE_WANDB:
    import wandb
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb_key = user_secrets.get_secret("WANDB_API_KEY")
        wandb.login(key=wandb_key)
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(project="ST-her2st-kaggle", name="lighthggep")
        print("Da bat W&B logging.")
    except Exception as e:
        print("Khong the bat W&B (thieu secret WANDB_API_KEY?), tiep tuc voi CSVLogger.", e)
        USE_WANDB = False

from pytorch_lightning.loggers import CSVLogger
default_logger = wandb_logger if (USE_WANDB and wandb_logger is not None) else CSVLogger("logs", name="lighthggep")
print("Logger:", default_logger)

# ============================================================================
# ---- Cell 6 (notebook gốc) ----
# ============================================================================
import os
import pathlib

WORKDIR = "/kaggle/working"
os.chdir(WORKDIR)

# Cac thu muc se duoc tao trong qua trinh chay:
os.makedirs("data", exist_ok=True)
os.makedirs("model_ckpts", exist_ok=True)
os.makedirs("cache_features_train", exist_ok=True)
os.makedirs("cache_features_test", exist_ok=True)
os.makedirs("figures/kmeans", exist_ok=True)
os.makedirs("figures/FASN", exist_ok=True)
os.makedirs("models", exist_ok=True)
print("Working dir:", os.getcwd())

# ============================================================================
# ---- Cell 7 (notebook gốc) ----
# ============================================================================
# ----------------------------------------------------------
# 2.1 Clone du lieu HER2ST (chi can chay 1 LAN)
# ----------------------------------------------------------
if not os.path.isdir("data/her2st/.git"):
    subprocess.run("cd data && git clone https://github.com/almaan/her2st.git", shell=True)  # [DỊCH TỪ IPYTHON] gốc: !cd data && git clone https://github.com/almaan/her2st.git
else:
    print("data/her2st da ton tai, bo qua clone.")

# ============================================================================
# ---- Cell 8 (notebook gốc) ----
# ============================================================================
# ----------------------------------------------------------
# 2.2 Giai nen cac file .tsv.gz trong ST-cnts (chi can chay 1 lan)
# ----------------------------------------------------------
cnt_dir = "data/her2st/data/ST-cnts"
gz_files = [f for f in os.listdir(cnt_dir) if f.endswith(".gz")]
if gz_files:
    subprocess.run(f"cd {cnt_dir} && gunzip -f *.gz", shell=True)  # [DỊCH TỪ IPYTHON] gốc: !cd {cnt_dir} && gunzip -f *.gz
    print(f"Da giai nen {len(gz_files)} file.")
else:
    print("Khong con file .gz (co the da giai nen roi).")

print("So file .tsv trong ST-cnts:", len([f for f in os.listdir(cnt_dir) if f.endswith(".tsv")]))

# ============================================================================
# ---- Cell 9 (notebook gốc) ----
# ============================================================================
# ----------------------------------------------------------
# 2.3 Copy file her_hvg_cut_1000.npy tu Kaggle Dataset
# ----------------------------------------------------------
import shutil, glob

candidates = glob.glob("/kaggle/input/*/her_hvg_cut_1000.npy") + glob.glob("/kaggle/input/*/**/her_hvg_cut_1000.npy", recursive=True)
if candidates:
    shutil.copy(candidates[0], "data/her_hvg_cut_1000.npy")
    print("Da copy her_hvg_cut_1000.npy tu:", candidates[0])
elif os.path.isfile("data/her_hvg_cut_1000.npy"):
    print("data/her_hvg_cut_1000.npy da ton tai.")
else:
    raise FileNotFoundError(
        "KHONG TIM THAY her_hvg_cut_1000.npy trong /kaggle/input/. "
        "Hay upload file nay len 1 Kaggle Dataset (vi du 'her2st-extra') va Add Input truoc khi chay tiep."
    )

# ============================================================================
# ---- Cell 10 (notebook gốc) ----
# ============================================================================
# ----------------------------------------------------------
# 2.4 Khoi phuc model_ckpts tu Kaggle Dataset cua session truoc (neu co)
# ----------------------------------------------------------
def restore_dir_from_input(dirname):
    matches = glob.glob(f"/kaggle/input/*/{dirname}") 
    if matches:
        src = matches[0]
        dst = os.path.join(WORKDIR, dirname)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"Da khoi phuc {dirname} tu {src}")
        return True
    return False

for d in ["model_ckpts"]:
    if not restore_dir_from_input(d):
        print(f"Khong tim thay {d} trong /kaggle/input (binh thuong neu day la lan chay dau tien).")

# ============================================================================
# ---- Cell 17 (notebook gốc) ----
# ============================================================================
# Tao file __init__.py
# with open("models/__init__.py", "w") as f:
#     f.write("from .LightHGGEP import LightHGGEP\n")

# import sys
# if WORKDIR not in sys.path:
#     sys.path.insert(0, WORKDIR)

# print("Da ghi xong toan bo module. Cau truc thu muc hien tai:")
# subprocess.run('''find . -maxdepth 2 -name "*.py" | sort''', shell=True)  # [DỊCH TỪ IPYTHON] gốc: !find . -maxdepth 2 -name "*.py" | sort

# ============================================================================
# ---- Cell 19 (notebook gốc) ----
# ============================================================================
FOLD = 5
N_GENES = 785
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-4
K_NEIGHBORS = 4
BATCH_SIZE = 32  # Light-HGGEP rat nhe nen co the tang batch size

CKPT_DIR = "model_ckpts"
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"Configuration:")
print(f"  FOLD = {FOLD}")
print(f"  N_GENES = {N_GENES}")
print(f"  MAX_EPOCHS = {MAX_EPOCHS}")
print(f"  PATIENCE = {PATIENCE}")
print(f"  LEARNING_RATE = {LEARNING_RATE}")
print(f"  BATCH_SIZE = {BATCH_SIZE}")

# ============================================================================
# ---- Cell 22 (notebook gốc) ----
# ============================================================================
# ==== [MOI - vá 2 lỗi đã phát hiện khi rà soát] ====
# Lỗi 1: DataLoader mac dinh (shuffle=True, khong Sampler/collate_fn rieng) tron lan cac
# spot tu NHIEU section khac nhau vao 1 batch -- vi pham dieu kien bat buoc cua Spatial SGC
# (Eq.2): A_norm_full[local_indices][:, local_indices] chi co y nghia khi TOAN BO
# local_indices trong 1 batch thuoc CUNG 1 do thi/section. Ngoai ra default_collate cua
# PyTorch LUON goi truong section_name (kieu str) thanh 1 LIST (ke ca batch_size=1), khien
# `section_name in self.A_norm_cache` (trong forward()) nem TypeError: unhashable type
# 'list' -- crash ngay batch dau tien, ca luc train LAN luc predict (test_loader).
#
# Lỗi 2: trainer.fit(model, train_loader) khong truyen val_dataloader nao, trong khi
# EarlyStopping/ModelCheckpoint lai theo doi 'val_loss' -- chi so nay khong bao gio duoc
# log vi validation_step() khong bao gio duoc goi.
#
# Cach va: KHONG doi Dataset (dataset.py)/Model (models/LightHGGEP.py) -- chi them 1
# Sampler dam bao moi batch CHI chua 1 section, va 1 collate_fn giu section_name la 1
# CHUOI DUY NHAT thay vi list. Vi Model doc dung 1 chuoi section_name tu batch (khong doi
# forward()/training_step()/validation_step()/test_step()), day la cach va dung o dung lop
# DataLoader, khong dung vao logic model/dataset.
import random
import math
from torch.utils.data import Sampler


class SectionBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True,
                 include_sections=None, exclude_sections=None):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.section_indices = {}
        start = 0
        for i, length in enumerate(dataset.lengths):
            name = dataset.id2name[i]
            self.section_indices[name] = list(range(start, start + length))
            start += length

        if include_sections is not None:
            self.section_indices = {k: v for k, v in self.section_indices.items()
                                     if k in set(include_sections)}
        if exclude_sections is not None:
            self.section_indices = {k: v for k, v in self.section_indices.items()
                                     if k not in set(exclude_sections)}
        
        # Lưu danh sách section names để shuffle
        self.section_names = list(self.section_indices.keys())

    def __iter__(self):
        section_names = self.section_names.copy()
        if self.shuffle:
            random.shuffle(section_names)
        for name in section_names:
            idxs = list(self.section_indices[name])
            if self.shuffle:
                random.shuffle(idxs)
            # Cắt thành các batch nhỏ theo BATCH_SIZE
            for s in range(0, len(idxs), self.batch_size):
                yield idxs[s:s + self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(v) / self.batch_size) for v in self.section_indices.values())
        
def section_collate_fn(batch):
    """Thay the default_collate CHI cho truong section_name (str -> giu nguyen 1 chuoi
    thay vi bi goi thanh list). Moi truong khac (patch_3ch/loc/exp/center/local_idx) duoc
    torch.stack() giong het hanh vi mac dinh cua default_collate cho tensor cung shape --
    KHONG doi gia tri/kieu du lieu nao khac ngoai section_name."""
    is_train = (len(batch[0]) == 5)   # train: 5 phan tu; test: 6 phan tu (co them center)
    sec_pos = 3 if is_train else 4

    section_names = [b[sec_pos] for b in batch]
    assert len(set(section_names)) == 1, (
        f"SectionBatchSampler lỗi: 1 batch chứa nhiều section khác nhau {set(section_names)} "
        f"-- Spatial SGC yêu cầu mọi spot trong batch phải cùng 1 section."
    )
    section_name = section_names[0]

    patch_3ch = torch.stack([b[0] for b in batch])
    loc = torch.stack([b[1] for b in batch])
    exp = torch.stack([b[2] for b in batch])

    if is_train:
        local_idx = torch.tensor([b[4] for b in batch], dtype=torch.long)
        return patch_3ch, loc, exp, section_name, local_idx
    else:
        center = torch.stack([b[3] for b in batch])
        local_idx = torch.tensor([b[5] for b in batch], dtype=torch.long)
        return patch_3ch, loc, exp, center, section_name, local_idx


print("Đã định nghĩa SectionBatchSampler / section_collate_fn (vá lỗi batching + section_name).")


# ============================================================================
# ---- Cell 25 (notebook gốc) ----
# ============================================================================
from dataset import LightHGGEP_HER2ST
from models.LightHGGEP import LightHGGEP
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl

# Dataset
train_dataset = LightHGGEP_HER2ST(train=True, fold=FOLD, k_neighbors=K_NEIGHBORS)
# [SỬA - vá lỗi 1+2] Tách 1 slide CỐ ĐỊNH trong 31 slide train làm validation (KHÔNG
# đụng test_dataset -- giữ đúng nguyên tắc LOOCV: test chỉ dùng 1 lần duy nhất lúc
# đánh giá cuối, xem PHẦN 6). Chọn theo alphabet cho tái lập được, có thể đổi thủ công
# nếu muốn slide khác.
VAL_SECTION = sorted(train_dataset.names)[0]
print(f"Slide dùng làm validation (tách từ tập train, KHÔNG phải test_dataset): {VAL_SECTION}")

train_sampler = SectionBatchSampler(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                     exclude_sections=[VAL_SECTION])
val_sampler = SectionBatchSampler(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                   include_sections=[VAL_SECTION])
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0,
                           collate_fn=section_collate_fn)
val_loader = DataLoader(train_dataset, batch_sampler=val_sampler, num_workers=0,
                         collate_fn=section_collate_fn)

# Model
model = LightHGGEP(
    n_genes=N_GENES,
    k_neighbors=K_NEIGHBORS,
    learning_rate=LEARNING_RATE,
    max_epochs=MAX_EPOCHS,
    cnn_chunk=BATCH_SIZE,
)

# Set graph cho model
for section, A_norm in train_dataset.A_norm_cache.items():
    model.set_graph(section, torch.from_numpy(A_norm).float())

# Tinh so tham so
total_params = sum(p.numel() for p in model.parameters())
print(f"Light-HGGEP total parameters: {total_params:,}")
assert total_params < 300000, f"Light-HGGEP should have <300K params, got {total_params:,}"

# Callbacks
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    mode='min',
    verbose=True
)

checkpoint_callback = ModelCheckpoint(
    dirpath=CKPT_DIR,
    filename='lighthggep_fold' + str(FOLD) + '_{epoch:02d}_{val_loss:.4f}',
    save_top_k=3,
    monitor='val_loss',
    mode='min',
    save_last=True
)

# Trainer
trainer = pl.Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,              # <-- ép 1 GPU, tắt hẳn DDP, không cần use_distributed_sampler nữa
    max_epochs=MAX_EPOCHS,
    callbacks=[early_stop_callback, checkpoint_callback],
    logger=default_logger,
    log_every_n_steps=100,   
    gradient_clip_val=1.0,
)

# Train
trainer.fit(model, train_loader, val_loader)

# Load best checkpoint
best_ckpt_path = checkpoint_callback.best_model_path
print(f"\nBest checkpoint: {best_ckpt_path}")
print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")


# ============================================================================
# ---- Cell 27 (notebook gốc) ----
# ============================================================================
from predict import lighthggep_predict, get_R, get_MSE, get_MAE, cluster
from utils import comp_tsne_km
import scanpy as sc
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load best model
best_model = LightHGGEP.load_from_checkpoint(
    best_ckpt_path,
    n_genes=N_GENES,
    k_neighbors=K_NEIGHBORS,
    learning_rate=LEARNING_RATE,
    max_epochs=MAX_EPOCHS,
    cnn_chunk=BATCH_SIZE
)

# Set graph cho model (cho test set)
test_dataset = LightHGGEP_HER2ST(train=False, fold=FOLD, k_neighbors=K_NEIGHBORS)
for section, A_norm in test_dataset.A_norm_cache.items():
    best_model.set_graph(section, torch.from_numpy(A_norm).float())

test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, collate_fn=section_collate_fn)

# Predict
label = test_dataset.label[test_dataset.names[0]]
adata_pred, adata_gt = lighthggep_predict(best_model, test_loader, device=device)

# Post-processing
adata_pred = comp_tsne_km(adata_pred, 4)
g = list(np.load('data/her_hvg_cut_1000.npy', allow_pickle=True))
adata_pred.var_names = g
sc.pp.scale(adata_pred)

# ==================== TÍNH TOÁN METRICS ====================

# Pearson Correlation cho từng gene
R, p_values = get_R(adata_pred, adata_gt)

# MSE cho từng gene
MSE = get_MSE(adata_pred, adata_gt)

# MAE cho từng gene
MAE = get_MAE(adata_pred, adata_gt)

# ==================== IN KẾT QUẢ CHI TIẾT ====================

print("="*70)
print("KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG - Light-HGGEP trên HER2ST (tập test)")
print("="*70)

# Thông tin tổng quan
n_spots = adata_pred.shape[0]
mean_pcc = np.nanmean(R)
median_pcc = np.nanmedian(R)
std_pcc = np.nanstd(R)

print(f"  Số spot test đã đánh giá : {n_spots}")
print(f"  Số gene đánh giá        : {len(R)}")
print(f"  MSE tổng thể             : {np.nanmean(MSE):.4f}")
print(f"  PCC trung bình (gen)     : {mean_pcc:.4f}")
print(f"  PCC trung vị   (gen)     : {median_pcc:.4f}")
print(f"  PCC std        (gen)     : {std_pcc:.4f}")

# Tạo DataFrame với thông tin các gene
gene_stats = pd.DataFrame({
    'gene': g,
    'pcc': R,
    'p_value': p_values,
    'mse': MSE,
    'mae': MAE
})

# Top-10 gene tốt nhất (PCC cao nhất)
print("\nTop-10 gen dự đoán TỐT NHẤT (PCC cao nhất):")
top10_best = gene_stats.nlargest(10, 'pcc')[['gene', 'pcc']]
print("  " + top10_best.to_string(index=False).replace('\n', '\n  '))

# Top-10 gene kém nhất (PCC thấp nhất)
print("\nTop-10 gen dự đoán KÉM NHẤT (PCC thấp nhất):")
top10_worst = gene_stats.nsmallest(10, 'pcc')[['gene', 'pcc']]
print("  " + top10_worst.to_string(index=False).replace('\n', '\n  '))

# Thống kê bổ sung
print("\n" + "="*70)
print("THỐNG KÊ BỔ SUNG")
print("="*70)
print(f"  Số gene có PCC > 0:  {np.sum(R > 0):,}/{len(R)} ({100*np.sum(R > 0)/len(R):.1f}%)")
print(f"  Số gene có PCC > 0.2: {np.sum(R > 0.2):,}/{len(R)} ({100*np.sum(R > 0.2)/len(R):.1f}%)")
print(f"  Số gene có PCC > 0.3: {np.sum(R > 0.3):,}/{len(R)} ({100*np.sum(R > 0.3)/len(R):.1f}%)")

# ARI
clus, ARI = cluster(adata_pred, label)
print(f"\nARI (Adjusted Rand Index): {ARI:.4f}")
print("="*70)

# ==================== VẼ HISTOGRAM PCC ====================

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(R, bins=30, color="tab:purple", alpha=0.75, edgecolor="black")
ax.axvline(mean_pcc, color="red", linestyle="--", linewidth=2, label=f"Mean PCC = {mean_pcc:.3f}")
ax.axvline(median_pcc, color="blue", linestyle="-.", linewidth=2, label=f"Median PCC = {median_pcc:.3f}")
ax.set_xlabel("PCC (Pearson Correlation Coefficient)")
ax.set_ylabel("Số lượng gen")
ax.set_title(f"Phân bố PCC của Light-HGGEP trên {len(R)} gene (HER2ST test set)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"figures/Light-HGGEP_PCC_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nĐã lưu biểu đồ PCC distribution vào figures/Light-HGGEP_PCC_distribution.png")

# Lưu toàn bộ kết quả gene stats vào CSV
gene_stats.to_csv(f"gene_predictions_stats.csv", index=False)
print(f"Đã lưu thống kê chi tiết từng gene vào gene_predictions_stats.csv")

# ============================================================================
# ---- Cell 29 (notebook gốc) ----
# ============================================================================
import matplotlib.pyplot as plt

# K-means clusters
sc.pl.spatial(adata_pred, img=None, color="kmeans", spot_size=112, 
              frameon=False, legend_loc=None, title=None, show=False)
plt.gca().set_title("")
plt.savefig(f"figures/kmeans/Light-HGGEP_kmeans_fold{FOLD}.png", dpi=300, bbox_inches="tight", transparent=True)
plt.clf()
plt.close()
print(f"Saved: figures/kmeans/Light-HGGEP_kmeans_fold{FOLD}.png")

# FASN gene expression
sc.pl.spatial(adata_pred, img=None, color="FASN", spot_size=112, 
              color_map="magma", frameon=False, legend_loc=None, title=None, show=False)
plt.gca().set_title("")
plt.savefig(f"figures/FASN/Light-HGGEP_FASN_fold{FOLD}.png", dpi=300, bbox_inches="tight", transparent=True)
plt.clf()
plt.close()
print(f"Saved: figures/FASN/Light-HGGEP_FASN_fold{FOLD}.png")

# ============================================================================
# ---- Cell 31 (notebook gốc) ----
# ============================================================================
import pandas as pd

results = pd.DataFrame([{
    'model': 'Light-HGGEP',
    'fold': FOLD,
    'pearson': np.nanmean(R),
    'ari': ARI,
    'mse': np.nanmean(MSE),
    'mae': np.nanmean(MAE),
    'params': total_params,
    'best_epoch': checkpoint_callback.best_model_score,
}])

print("\n" + "="*60)
print("KET QUA LIGHT-HGGEP")
print("="*60)
print(results.to_string(index=False))
print("="*60)

# Luu ket qua
results.to_csv("Light-HGGEP_results.csv", index=False)
print("\nDa luu ket qua vao Light-HGGEP_results.csv")

# ============================================================================
# ---- Cell 33 (notebook gốc) ----
# ============================================================================
print("Checkpoints da luu trong:")
subprocess.run(f"ls -la {CKPT_DIR}", shell=True)  # [DỊCH TỪ IPYTHON] gốc: !ls -la {CKPT_DIR}
print(f"\nBest checkpoint: {best_ckpt_path}")
print("\nDe su dung lai session sau, vao tab Output > New Dataset tu thu muc model_ckpts/")
