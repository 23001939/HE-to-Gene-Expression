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
import time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
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
N_GPUS = 1  # LightHGGEP is I/O-bound and faster/more stable on one T4.
print("GPUs used for Light-HGGEP:", N_GPUS)

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

# [SỬA lỗi #6] WORKDIR luôn = thư mục chứa file này (repo root) -- đúng trên mọi môi
# trường (Kaggle, local, Colab). Trước đây Kaggle dùng /kaggle/working nhưng clone repo
# vào /kaggle/working/HE-to-Gene-Expression/ nên path data/... resolve sai.
WORKDIR = str(pathlib.Path(__file__).parent.resolve())
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
    # [SỬA lỗi #7] Dùng tham số cwd thay vì "cd data && ..." -- lệnh cd trong subprocess
    # shell=True chạy trong tiến trình con riêng nên không ảnh hưởng thư mục làm việc
    # của Python; dùng cwd= là cách đúng và nhất quán trên cả Linux lẫn Windows.
    # [DỊCH TỪ IPYTHON] gốc: !cd data && git clone https://github.com/almaan/her2st.git
    subprocess.run("git clone https://github.com/almaan/her2st.git", shell=True, cwd="data")
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
# ---- Cell 10 (notebook gốc) ----
# ============================================================================
# ----------------------------------------------------------
# 2.4 Khoi phuc model_ckpts tu Kaggle Dataset cua session truoc (neu co)
# ----------------------------------------------------------
import glob
import shutil

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
import argparse

# [K-FOLD] cac fold chay nam trong FOLD_LIST (tinh tu --fold-start/--fold-end o phan args).
# Chon dataset qua CLI: 'her2st' (785 gen her_hvg_cut_1000) hoac 'her2st_top250'
# (250 gen co muc bieu hien trung binh cao nhat, chon tu count matrix truoc LOOCV split).
_p = argparse.ArgumentParser()
_p.add_argument('--datasets', choices=['her2st', 'her2st_top250', 'brainst'], default='her2st',
                help="Dataset dung cho training/eval (mac dinh: her2st)")
_p.add_argument('--use_sgc', action='store_true',
                help="BAT Spatial SGC (GCN nhe 2-hop, phai feed nguyen section -> "
                     "BATCH_SIZE=max(section)). Mac dinh TAT (CNN thuan, PCC cao hon"
                     " tren HER2ST: 0.259 vs 0.189 voi SGC). Dung de thu nghiem SGC.")
_p.add_argument('--no_sgc_nonlinear', action='store_true',
                help="Chi voi --use_sgc: tat phi tuyen (SGC thuan tuyen goc). Mac dinh "
                     "co phi tuyen ReLU.")
_p.add_argument('--sgc_alpha', type=float, default=0.3,
                help="Trong so residual cua SGC (chi voi --use_sgc). CNN da co cau "
                     "truc khong gian san, SGC full (1.0) lam tron qua da -> PCC tut. "
                     "Mac dinh 0.3: SGC chi hieu chinh nhe, giu PCC cao + giu kien "
                     "truc SGC. 0.0 ~ tat SGC, 1.0 = SGC goc.")
_p.add_argument('--sgc_hops', type=int, default=2,
                help="So lan lan truyen (hop) tren do thi KNN khong gian cua SGC "
                     "(chi voi --use_sgc). Mac dinh 2 (paper). Tang de mo rong tam "
                     "anh huong khong gian, 1 = chi hang xom truc tiep. Toi thieu 1.")
# [K-FOLD] chay tu FOLD_START den FOLD_END (ca 2 dau deu duoc chay).
# Mac dinh chi chay 1 fold = 5 (nhu cu). fold_end >= fold_start.
_p.add_argument('--fold-start', type=int, default=5,
                help="Fold bat dau (mac dinh 5). Dung cung --fold-end de chay k-fold.")
_p.add_argument('--fold-end', type=int, default=None,
                help="Fold ket thuc (mac dinh = fold-start, nghia la 1 fold). Ca 2 "
                     "dau deu duoc chay. VD --fold-start 1 --fold-end 5 chay fold 1..5.")
_args = _p.parse_args()
FOLD_START = _args.fold_start
FOLD_END = _args.fold_end if _args.fold_end is not None else _args.fold_start
assert FOLD_END >= FOLD_START, f"--fold-end ({FOLD_END}) phai >= --fold-start ({FOLD_START})"
FOLD_LIST = list(range(FOLD_START, FOLD_END + 1))
DATASET = _args.datasets
# [SGC] AB chung to SGC full lam giam PCC tren HER2ST (0.189 vs 0.259 CNN thuan).
# Mac dinh TAT SGC (CNN thuan, batch 32, nhanh + tot nhat). --use_sgc de bat
# SGC voi alpha nhe (0.3) giu kien truc paper ma khong tut PCC.
USE_SGC = _args.use_sgc
SGC_ALPHA = _args.sgc_alpha
SGC_NONLINEAR = not _args.no_sgc_nonlinear
SGC_HOPS = max(int(_args.sgc_hops), 1)
N_GENES = None  # tu dong lay tu dataset gene_set neu de None
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-4
K_NEIGHBORS = 4
BATCH_SIZE = 32  # Light-HGGEP rat nhe nen co the tang batch size
NUM_WORKERS = 2  # per DDP rank (4 loader workers total with 2 GPUs)

print(f"Configuration:")
print(f"  FOLDS = {FOLD_LIST}")
print(f"  DATASET = {DATASET}")
print(f"  N_GENES = {N_GENES}")
print(f"  MAX_EPOCHS = {MAX_EPOCHS}")
print(f"  PATIENCE = {PATIENCE}")
print(f"  LEARNING_RATE = {LEARNING_RATE}")
print(f"  BATCH_SIZE = {BATCH_SIZE}")
print(f"  NUM_WORKERS = {NUM_WORKERS} per DDP rank")
print(f"  USE_SGC = {USE_SGC}")
if USE_SGC:
    print(f"  SGC_HOPS = {SGC_HOPS}")
    print(f"  SGC_NONLINEAR = {SGC_NONLINEAR}")
    print(f"  SGC_ALPHA = {SGC_ALPHA}")

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
                 include_sections=None, exclude_sections=None,
                 rank=0, num_replicas=1, shard_sections=True,
                 section_indices_override=None):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.section_indices = {}
        start = 0
        for i, length in enumerate(dataset.lengths):
            name = dataset.id2name[i]
            self.section_indices[name] = list(range(start, start + length))
            start += length

        # [BRAIN-ST] override cho phep chia spot-level train/val trong 1 section
        # duy nhat (dataset chi co 1 section -> khong the exclude ca section).
        # section_indices_override: dict section_name -> list global indices.
        if section_indices_override is not None:
            self.section_indices = {k: list(v)
                                    for k, v in section_indices_override.items()}

        if include_sections is not None:
            self.section_indices = {k: v for k, v in self.section_indices.items()
                                     if k in set(include_sections)}
        if exclude_sections is not None:
            self.section_indices = {k: v for k, v in self.section_indices.items()
                                     if k not in set(exclude_sections)}
        
        # A rank owns complete sections: spatial graphs are never split across
        # GPUs.  Greedy bin-packing balances *batch counts*, not merely section
        # counts, because HER2ST sections have very different spot counts.
        self.section_names = list(self.section_indices.keys())
        self.steps_per_epoch = sum(math.ceil(len(v) / self.batch_size)
                                   for v in self.section_indices.values())
        if shard_sections and num_replicas > 1:
            bins = [([], 0) for _ in range(num_replicas)]
            names_by_size = sorted(
                self.section_names,
                key=lambda name: math.ceil(len(self.section_indices[name]) / self.batch_size),
                reverse=True,
            )
            for name in names_by_size:
                target = min(range(num_replicas), key=lambda i: bins[i][1])
                batch_count = math.ceil(len(self.section_indices[name]) / self.batch_size)
                bins[target][0].append(name)
                bins[target] = (bins[target][0], bins[target][1] + batch_count)
            # DDP needs an identical number of optimizer steps on every rank.
            # The very small excess is dropped after shuffling, so it rotates
            # between sections across epochs instead of permanently omitting one.
            self.section_names = bins[rank][0]
            self.steps_per_epoch = min(total for _, total in bins)

    def __iter__(self):
        section_names = self.section_names.copy()
        if self.shuffle:
            random.shuffle(section_names)
        emitted = 0
        for name in section_names:
            idxs = list(self.section_indices[name])
            if self.shuffle:
                random.shuffle(idxs)
            # Cắt thành các batch nhỏ theo BATCH_SIZE
            for s in range(0, len(idxs), self.batch_size):
                if emitted >= self.steps_per_epoch:
                    return
                yield idxs[s:s + self.batch_size]
                emitted += 1

    def __len__(self):
        return self.steps_per_epoch

from pytorch_lightning.callbacks import Callback

class SimpleProgressBar(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_started_at = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module):
        # Validation runs after training in each epoch, so this reports the
        # current epoch's MSE/PCC rather than stale validation metrics.
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        train_mse = trainer.callback_metrics.get('train_mse', trainer.callback_metrics.get('train_loss', 0.0))
        val_mse = trainer.callback_metrics.get('val_mse', trainer.callback_metrics.get('val_loss', 0.0))
        train_pcc = trainer.callback_metrics.get('train_pcc', float('nan'))
        val_pcc = trainer.callback_metrics.get('val_pcc', float('nan'))
        current_epoch = trainer.current_epoch
        total_epochs = trainer.max_epochs
        # [SỬA lỗi #10] trainer.optimizers có thể là [] trong một số phiên bản PL vì
        # optimizers được lazy-init; pl_module.optimizers() luôn trả về optimizer đang
        # hoạt động một cách an toàn hơn.
        opt = pl_module.optimizers()
        # optimizers() có thể trả về list hoặc optimizer đơn tùy phiên bản PL
        if isinstance(opt, list):
            opt = opt[0]
        lr = opt.param_groups[0]['lr']
        elapsed = time.perf_counter() - getattr(self, 'epoch_started_at', time.perf_counter())
        remaining = max(total_epochs - (current_epoch + 1), 0) * elapsed
        
        # In đúng format bạn muốn
        print(f"[ep {current_epoch + 1}/{total_epochs}] "
              f"train_mse={train_mse:.4f} train_pcc={train_pcc:.4f} "
              f"val_mse={val_mse:.4f} val_pcc={val_pcc:.4f} lr={lr:.4e} "
              f"epoch_time={elapsed:.1f}s eta={remaining / 60:.1f}m")


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
def run_fold(fold):
    """Chay toan bo pipeline (train -> eval -> visualize) cho 1 fold.
    Tra ve DataFrame ket qua 1 dong cua fold do."""
    # [K-FOLD] cac bien config co the bi gan lai ben trong ham (N_GENES, USE_SGC,
    # BATCH_SIZE) -> khai bao global de khong bi UnboundLocalError va de lai gia
    # tri cho cac fold sau (datasets brainst bat buoc USE_SGC=True, va SGC bat
    # thi BATCH_SIZE bi ghi de thanh max(section lengths)).
    global N_GENES, USE_SGC, BATCH_SIZE, VAL_SECTION
    # ---- Cell 25 (notebook gốc) ----
    # ============================================================================
    from dataset import LightHGGEP_HER2ST, LightHGGEP_HER2ST_Top250, LightHGGEP_BRAINST
    from models.LightHGGEP import LightHGGEP
    from torch.utils.data import DataLoader
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    import pytorch_lightning as pl

    # [K-FOLD] moi fold co checkpoint dir rieng de khong ghi de nhau
    CKPT_DIR = os.path.join("model_ckpts", f"fold{fold}")
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Chon class dataset theo DATASET
    if DATASET == 'her2st':
        DATASET_CLASS = LightHGGEP_HER2ST
    elif DATASET == 'her2st_top250':
        DATASET_CLASS = LightHGGEP_HER2ST_Top250
    elif DATASET == 'brainst':
        DATASET_CLASS = LightHGGEP_BRAINST
    else:
        raise ValueError(f"Unknown --datasets: {DATASET}")
    if N_GENES is None:
        N_GENES = len(DATASET_CLASS(train=True, fold=fold, k_neighbors=K_NEIGHBORS).gene_set)
        print(f"  N_GENES auto = {N_GENES}")

    # Dataset
    train_dataset = DATASET_CLASS(train=True, fold=fold, k_neighbors=K_NEIGHBORS)
    # [SỬA - vá lỗi 1+2] Tách 1 slide CỐ ĐỊNH trong 31 slide train làm validation (KHÔNG
    # đụng test_dataset -- giữ đúng nguyên tắc LOOCV: test chỉ dùng 1 lần duy nhất lúc
    # đánh giá cuối, xem PHẦN 6). Chọn theo alphabet cho tái lập được, có thể đổi thủ công
    # nếu muốn slide khác.
    VAL_SECTION = sorted(train_dataset.names)[0]
    print(f"Slide dùng làm validation (tách từ tập train, KHÔNG phải test_dataset): {VAL_SECTION}")

    # [HƯỚNG A] dataset chia 1 section thành 4 tile, mỗi tile = 1 "section" riêng
    # (xem LightHGGEP_BRAINST.__init__). Validation = 1 tile cố định (val_tile,
    # ~25%), train = 3 tile còn lại (~75%). Mỗi batch = NGUYÊN 1 tile (~676 spot)
    # -> SectionBatchSampler chỉ cắt theo BATCH_SIZE, tile nhỏ hơn BATCH_SIZE thì
    # thành đúng 1 batch. SGC chạy ĐÚNG trên ma trận kề tile (676x676), không OOM.
    train_override = None
    val_override = None
    if DATASET == 'brainst':
        # BRAIN-ST chi co the chay SGC (1 batch = nguyen tile) -> bat SGC bat buoc.
        USE_SGC = True
        VAL_SECTION = train_dataset.val_tile
        _n_train = sum(train_dataset.lengths) - train_dataset.lengths[train_dataset.names.index(VAL_SECTION)]
        print(f"[BRAIN-ST] tile split: train={_n_train} val={train_dataset.lengths[train_dataset.names.index(VAL_SECTION)]} "
              f"(val_tile={VAL_SECTION})")

    # [SGC] de SGC THUC SU chay, moi batch phai = NGUYEN 1 section (z_spot.shape[0]
    # == A_norm.shape[0]). Nen BATCH_SIZE phai >= section dai nhat -> moi section
    # thanh dung 1 batch (SectionBatchSampler cat theo BATCH_SIZE, section nho hon
    # BATCH_SIZE se thanh 1 batch). Chi ap dung khi USE_SGC bat (HER2ST mac dinh
    # TAT -> batch 32 nhanh + PCC cao hon; brainst bat buoc SGC). chunk(64)+
    # checkpoint da chung minh 712/697 spot khong OOM tren T4.
    if USE_SGC:
        BATCH_SIZE = max(train_dataset.lengths)
        print(f"[SGC] BATCH_SIZE = {BATCH_SIZE} (full section/batch -> SGC enabled, "
              f"chunk+checkpoint tranh OOM)")

    DDP_RANK = int(os.environ.get("LOCAL_RANK", 0))
    # Kaggle's parent DDP process can construct the rank-0 loader before it
    # exports WORLD_SIZE.  Fall back to the configured device count so rank 0 also
    # receives only its own section shard rather than processing the full dataset.
    DDP_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", N_GPUS))
    train_sampler = SectionBatchSampler(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         exclude_sections=[VAL_SECTION],
                                         section_indices_override=train_override,
                                         rank=DDP_RANK,
                                         num_replicas=DDP_WORLD_SIZE, shard_sections=True)
    val_sampler = SectionBatchSampler(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                       include_sections=[VAL_SECTION],
                                       section_indices_override=val_override,
                                       rank=DDP_RANK,
                                       num_replicas=DDP_WORLD_SIZE, shard_sections=False)
    print(f"DDP data shard: rank {DDP_RANK}/{DDP_WORLD_SIZE}; "
          f"train sections={len(train_sampler.section_names)}, "
          f"train batches={len(train_sampler)}", flush=True)
    loader_options = dict(num_workers=NUM_WORKERS,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=NUM_WORKERS > 0,
                          timeout=180)
    eval_loader_options = dict(num_workers=NUM_WORKERS,
                               pin_memory=torch.cuda.is_available(),
                               # Avoid keeping train and validation worker caches
                               # alive simultaneously on every DDP rank.
                               persistent_workers=False,
                               timeout=180)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                               collate_fn=section_collate_fn, **loader_options)
    val_loader = DataLoader(train_dataset, batch_sampler=val_sampler,
                             collate_fn=section_collate_fn, **eval_loader_options)

    # Model
    # [HƯỚNG A] cnn_chunk GIỮ NHỎ (không = BATCH_SIZE) để forward luôn chunk input
    # (tránh OOM kể cả khi 1 batch = 676 patch). Chunk KHÔNG ảnh hưởng SGC vì z_spot
    # được ghép lại đủ cả tile trước khi lan truyền.
    model = LightHGGEP(
        n_genes=N_GENES,
        k_neighbors=K_NEIGHBORS,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        cnn_chunk=64,
        use_sgc=USE_SGC,
        sgc_nonlinear=SGC_NONLINEAR,
        sgc_alpha=SGC_ALPHA,
        sgc_hops=SGC_HOPS,
    )

    # Set graph cho model
    for section, A_norm in train_dataset.A_norm_cache.items():
        model.set_graph(section, torch.from_numpy(A_norm).float())

    # Tinh so tham so
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Light-HGGEP total parameters: {total_params:,}")
    assert total_params < 300000, f"Light-HGGEP should have <300K params, got {total_params:,}"

    # Callbacks
    # [BRAIN-ST] val_loss plateau nhanh hon HER2ST -> tang patience de model co
    # thoi gian hoc (train van dang giam du val dung yen). HER2ST giu PATIENCE=15.
    _es_patience = 40 if DATASET == 'brainst' else PATIENCE
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=_es_patience,
        mode='min',
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename='lighthggep_fold' + str(fold) + '_{epoch:02d}_{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=N_GPUS,
        # LightHGGEP uses every parameter in its forward loss path; plain DDP avoids
        # the unnecessary autograd traversal warned about by find_unused_parameters.
        strategy='ddp' if N_GPUS > 1 else 'auto',
        # SectionBatchSampler shards whole spatial sections itself.
        use_distributed_sampler=False,
        max_epochs=MAX_EPOCHS,
        callbacks=[early_stop_callback, checkpoint_callback, SimpleProgressBar()],
        logger=default_logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        precision='16-mixed' if torch.cuda.is_available() else '32-true',
        enable_progress_bar=False,
        enable_model_summary=False,     # Tắt bảng tóm tắt model
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Only rank zero performs the single canonical test evaluation and writes files.
    trainer.strategy.barrier()
    if not trainer.is_global_zero:
        trainer.strategy.barrier()
        raise SystemExit(0)

    # Load best checkpoint
    best_ckpt_path = checkpoint_callback.best_model_path
    print(f"\nBest checkpoint: {best_ckpt_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")


    # ============================================================================
    # ---- Cell 27 (notebook gốc) ----
    # ============================================================================
    from predict import lighthggep_predict
    from evaluation import PROTOCOL_NAME, evaluate_her2st_predictions
    import scanpy as sc

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load best model
    best_model = LightHGGEP.load_from_checkpoint(
        best_ckpt_path,
        n_genes=N_GENES,
        k_neighbors=K_NEIGHBORS,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        cnn_chunk=64,
        use_sgc=USE_SGC,
        sgc_nonlinear=SGC_NONLINEAR,
        sgc_alpha=SGC_ALPHA,
        sgc_hops=SGC_HOPS
    )

    # Set graph cho model (cho test set)
    test_dataset = DATASET_CLASS(train=False, fold=fold, k_neighbors=K_NEIGHBORS)
    for section, A_norm in test_dataset.A_norm_cache.items():
        best_model.set_graph(section, torch.from_numpy(A_norm).float())

    # [SỬA lỗi #4] test_loader phải đưa TOÀN BỘ spot của 1 section vào cùng 1 batch (hoặc
    # ít nhất các batch đủ lớn từ cùng 1 section) để Spatial SGC có thể lấy đúng
    # A_norm_full[local_indices][:, local_indices] với đầy đủ thông tin lân cận.
    # Dùng batch_size=1 trước đây → A_norm_batch = (1×1) → SGC không thấy láng giềng nào,
    # hoàn toàn vô nghĩa về mặt không gian.
    # SectionBatchSampler với shuffle=False đảm bảo mỗi batch CHỈ chứa 1 section và
    # duyệt tuần tự, phù hợp cho inference.
    test_sampler = SectionBatchSampler(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler,
                             collate_fn=section_collate_fn, **eval_loader_options)

    # Predict
    label = test_dataset.label[test_dataset.names[0]]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _t0 = time.perf_counter()
    adata_pred, adata_gt = lighthggep_predict(best_model, test_loader, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time_total_s = time.perf_counter() - _t0

    # Common fair evaluation: metrics are always computed on raw log-normalised
    # expression.  Only the visualisation/clustering copy is standardised.
    g = test_dataset.gene_set  # dung dung bo gen cua dataset da chon (785 hoac 250)
    adata_pred, metrics = evaluate_her2st_predictions(
        adata_pred, adata_gt, g, label=label, n_clusters=4)
    R, p_values = metrics['R'], metrics['p_values']
    Spearman, spearman_pvalues = metrics['Spearman'], metrics['spearman_pvalues']
    MSE, MAE, RMSE, morans = metrics['MSE'], metrics['MAE'], metrics['RMSE'], metrics['morans']
    ARI, NMI = metrics['ARI'], metrics['NMI']

    # ==================== IN KẾT QUẢ CHI TIẾT ====================

    print("="*70)
    print("KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG - Light-HGGEP trên HER2ST (tập test)")
    print("="*70)

    # Thông tin tổng quan
    n_spots = adata_pred.shape[0]
    inference_time_per_spot_ms = 1000.0 * inference_time_total_s / max(n_spots, 1)
    print(f"  [INFER TIME] total={inference_time_total_s:.3f}s "
          f"({n_spots} spot) -> {inference_time_per_spot_ms:.3f} ms/spot")
    mean_pcc      = metrics['pearson']
    median_pcc    = metrics['median_pearson']
    std_pcc       = np.nanstd(R)
    mean_spearman = metrics['spearman']
    mean_rmse     = metrics['rmse']
    mean_mae      = metrics['mae']
    mean_mi_pred  = metrics['morans_i_pred']
    mean_mi_gt    = metrics['morans_i_gt']

    print(f"  Số spot test đã đánh giá  : {n_spots}")
    print(f"  Số gene đánh giá          : {len(R)}")
    print(f"")
    print(f"  [Correlation]")
    print(f"  Mean Gene-wise PCC        : {mean_pcc:.4f}")
    print(f"  Median Gene-wise PCC      : {median_pcc:.4f}")
    print(f"  Std Gene-wise PCC         : {std_pcc:.4f}")
    print(f"  Mean Gene-wise Spearman   : {mean_spearman:.4f}")
    print(f"")
    print(f"  [Error]")
    print(f"  Mean RMSE                 : {mean_rmse:.4f}")
    print(f"  Mean MAE                  : {mean_mae:.4f}")
    print(f"")
    print(f"  [Spatial structure - top-50 high-var genes]")
    print(f"  Mean Moran's I (pred)     : {mean_mi_pred:.4f}")
    print(f"  Mean Moran's I (gt)       : {mean_mi_gt:.4f}")

    # Tạo DataFrame với thông tin các gene
    gene_stats = pd.DataFrame({
        'gene':          g,
        'pcc':           R,
        'pcc_pvalue':    p_values,
        'spearman':      Spearman,
        'spearman_pval': spearman_pvalues,
        'mse':           MSE,
        'rmse':          RMSE,
        'mae':           MAE,
    })

    # Top-10 gene tốt nhất (PCC cao nhất)
    print("\nTop-10 gen dự đoán TỐT NHẤT (PCC cao nhất):")
    top10_best = gene_stats.nlargest(10, 'pcc')[['gene', 'pcc', 'spearman']]
    print("  " + top10_best.to_string(index=False).replace('\n', '\n  '))

    # Top-10 gene kém nhất (PCC thấp nhất)
    print("\nTop-10 gen dự đoán KÉM NHẤT (PCC thấp nhất):")
    top10_worst = gene_stats.nsmallest(10, 'pcc')[['gene', 'pcc', 'spearman']]
    print("  " + top10_worst.to_string(index=False).replace('\n', '\n  '))

    # Thống kê bổ sung
    print("\n" + "="*70)
    print("THỐNG KÊ BỔ SUNG")
    print("="*70)
    print(f"  Số gene có PCC > 0:  {np.sum(R > 0):,}/{len(R)} ({100*np.sum(R > 0)/len(R):.1f}%)")
    print(f"  Số gene có PCC > 0.2: {np.sum(R > 0.2):,}/{len(R)} ({100*np.sum(R > 0.2)/len(R):.1f}%)")
    print(f"  Số gene có PCC > 0.3: {np.sum(R > 0.3):,}/{len(R)} ({100*np.sum(R > 0.3)/len(R):.1f}%)")

    # ARI + NMI (already computed by the shared protocol on the visualisation copy).
    if label is not None:
        print(f"\n  [Global structure]")
        print(f"  ARI (Adjusted Rand Index) : {ARI:.4f}")
        print(f"  NMI (Norm. Mutual Info)   : {NMI:.4f}")
    else:
        ARI = float('nan')
        NMI = float('nan')
        print("\nARI/NMI: N/A (section này không có ground-truth label)")
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
    plt.savefig(f"figures/Light-HGGEP_PCC_distribution_fold{fold}.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nĐã lưu biểu đồ PCC distribution vào figures/Light-HGGEP_PCC_distribution.png")

    # Lưu toàn bộ kết quả gene stats vào CSV
    gene_stats.to_csv(f"gene_predictions_stats_fold{fold}.csv", index=False)
    print(f"Đã lưu thống kê chi tiết từng gene vào gene_predictions_stats.csv")

    # ============================================================================
    # ---- Cell 29 (notebook gốc) ----
    # ============================================================================
    # K-means clusters
    sc.pl.spatial(adata_pred, img=None, color="kmeans", spot_size=112,
                  frameon=False, legend_loc=None, title=None, show=False)
    plt.gca().set_title("")
    plt.savefig(f"figures/kmeans/Light-HGGEP_kmeans_fold{fold}.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.clf()
    plt.close()
    print(f"Saved: figures/kmeans/Light-HGGEP_kmeans_fold{fold}.png")

    # FASN gene expression (chỉ khi FASN nằm trong bộ gen đang dùng)
    if "FASN" in adata_pred.var_names:
        sc.pl.spatial(adata_pred, img=None, color="FASN", spot_size=112,
                      color_map="magma", frameon=False, legend_loc=None, title=None, show=False)
        plt.gca().set_title("")
        plt.savefig(f"figures/FASN/Light-HGGEP_FASN_fold{fold}.png", dpi=300, bbox_inches="tight", transparent=True)
        plt.clf()
        plt.close()
        print(f"Saved: figures/FASN/Light-HGGEP_FASN_fold{fold}.png")

    # ============================================================================
    # ---- Cell 31 (notebook gốc) ----
    # ============================================================================

    results = pd.DataFrame([{
        'model':          'Light-HGGEP',
        'fold':           fold,
        'pearson':        np.nanmean(R),
        'spearman':       np.nanmean(Spearman),
        'ari':            ARI,
        'nmi':            NMI,
        'rmse':           np.nanmean(RMSE),
        'mae':            np.nanmean(MAE),
        'morans_i_pred':  np.nanmean(morans['pred']),
        'morans_i_gt':    np.nanmean(morans['gt']),
        'params':         total_params,
        'inference_time_total_s':     inference_time_total_s,
        'inference_time_per_spot_ms': inference_time_per_spot_ms,
        'n_test_spots':   n_spots,
        'best_val_loss':  float(checkpoint_callback.best_model_score),
        'eval_protocol':  PROTOCOL_NAME,
        'split_rule':     'LOOCV test=fold; validation=first alphabetical train slide',
        'n_genes':        N_GENES,
        'max_epochs':     MAX_EPOCHS,
        'learning_rate':  LEARNING_RATE,
        'optimizer':      'AdamW(weight_decay=1e-4)',
        'scheduler':      'CosineAnnealingLR(T_max=max_epochs,eta_min=1e-6)',
        'batch_size':     BATCH_SIZE,
        'seed':           42,
        'n_gpus':         N_GPUS,
        'precision':      '16-mixed' if torch.cuda.is_available() else '32-true',
    }])

    print("\n" + "="*60)
    print("KET QUA LIGHT-HGGEP")
    print("="*60)
    print(results.to_string(index=False))
    print("="*60)

    # Luu ket qua
    results.to_csv(f"Light-HGGEP_results_fold{fold}.csv", index=False)
    print("\nDa luu ket qua vao Light-HGGEP_results.csv")

    # ============================================================================
    # ---- Cell 33 (notebook gốc) ----
    # ============================================================================
    print("Checkpoints da luu trong:")
    subprocess.run(f"ls -la {CKPT_DIR}", shell=True)  # [DỊCH TỪ IPYTHON] gốc: !ls -la {CKPT_DIR}
    print(f"\nBest checkpoint: {best_ckpt_path}")
    print("\nDe su dung lai session sau, vao tab Output > New Dataset tu thu muc model_ckpts/")
    trainer.strategy.barrier()
    return results


# ============================================================================
# ---- [K-FOLD] Driver: loop qua FOLD_LIST, gop ket qua ----
# ============================================================================
ALL_RESULTS = []
for _fold in FOLD_LIST:
    print("\n" + "=" * 70)
    print(f"===== BAT DAU FOLD {_fold} ({_fold - FOLD_START + 1}/{len(FOLD_LIST)}) =====")
    print("=" * 70)
    ALL_RESULTS.append(run_fold(_fold))

combined = pd.concat(ALL_RESULTS, ignore_index=True)
combined.to_csv("Light-HGGEP_results.csv", index=False)
print("\n" + "=" * 70)
print("TONG KET QUA TAT CA FOLD (trung binh)")
print("=" * 70)
for col in ["pearson", "spearman", "ari", "nmi", "rmse", "mae",
            "morans_i_pred", "morans_i_gt"]:
    if col in combined.columns:
        print(f"  mean {col:>14} = {combined[col].mean():.4f}")
print(f"  so fold da chay     = {len(combined)}")
print("=" * 70)
print("\nDa luu ket qua tong hop vao Light-HGGEP_results.csv")

