"""
run_baselines.py -- Pipeline train / predict / eval cho các mô hình baseline trên HER2ST.

Tất cả baseline dùng HER2ST dataset để so sánh fair với LightHGGEP.

Cách dùng:
    python run_baselines.py --mode all          # chạy các baseline tương thích liên tiếp
    python run_baselines.py --mode stnet        # chạy riêng 1 model
    python run_baselines.py --mode histogene

Tùy chọn:
    --fold        : LOOCV fold (default: 5)
    --n_genes     : số gene dự đoán (default: 785)
    --max_epochs  : số epoch tối đa
    --batch_size  : batch size
    --lr          : learning rate (default: 1e-4, shared budget)
    --ckpt_dir    : thư mục lưu checkpoint (default: model_ckpts)
    --ckpt_path   : load checkpoint sẵn, bỏ qua train (chỉ dùng khi mode != all)
    --skip_train  : chỉ predict+eval (chỉ dùng khi mode != all)
    --n_gpus      : số GPU dùng (default: tối đa 2, giống Light-HGGEP)
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
                    choices=["histogene", "stnet", "all"],
                    help="Model muốn chạy. 'all' = chạy các baseline tương thích giao thức chung.")
parser.add_argument("--fold",       type=int,   default=5)
parser.add_argument("--n_genes",    type=int,   default=785)
parser.add_argument("--max_epochs", type=int,   default=None)
parser.add_argument("--batch_size", type=int,   default=None)
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--ckpt_dir",   type=str,   default="model_ckpts")
parser.add_argument("--ckpt_path",  type=str,   default=None,
                    help="Chỉ dùng khi --mode không phải 'all'")
parser.add_argument("--skip_train", action="store_true",
                    help="Chỉ dùng khi --mode không phải 'all'")
parser.add_argument("--n_gpus",     type=int,   default=None,
                    help="Số GPU dùng. Mặc định: tối đa 2, giống Light-HGGEP.")
args = parser.parse_args()

FOLD    = args.fold
N_GENES = args.n_genes
LR      = args.lr

# Số GPU
n_available = torch.cuda.device_count()
if args.n_gpus is not None:
    N_GPUS = min(args.n_gpus, n_available)
else:
    N_GPUS = min(n_available, 2)
N_GPUS = max(N_GPUS, 1)           # ít nhất 1

print("=" * 60)
print(f"BASELINE PIPELINE  mode={args.mode.upper()}  fold={FOLD}")
print("=" * 60)
print(f"  GPU available : {n_available}  →  dùng {N_GPUS} GPU")
print(f"  n_genes       : {N_GENES}")
print(f"  lr            : {LR}")
print("=" * 60)

# ── Imports chung ─────────────────────────────────────────────────────────────
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataloader import default_collate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import CSVLogger

from dataset import HER2ST
from evaluation import PROTOCOL_NAME, evaluate_her2st_predictions
from predict import stnet_predict, histogene_predict

# ── Callback: log mỗi epoch ra stdout ────────────────────────────────────────
class EpochProgressBar(Callback):
    """In MSE/PCC train-validation và learning rate sau mỗi epoch."""
    def on_train_epoch_end(self, trainer, pl_module):
        m        = trainer.callback_metrics
        ep       = trainer.current_epoch + 1
        total    = trainer.max_epochs
        t_mse    = m.get("train_mse", m.get("train_loss_epoch", m.get("train_loss", float("nan"))))
        v_mse    = m.get("val_mse", m.get("val_loss", m.get("valid_loss", float("nan"))))
        t_pcc    = m.get("train_pcc", float("nan"))
        v_pcc    = m.get("val_pcc", float("nan"))
        opt      = pl_module.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        lr = opt.param_groups[0]["lr"]
        # Thời gian epoch
        elapsed = trainer.fit_loop.epoch_loop.batch_progress.total.completed
        print(
            f"[{trainer.logger.name}] "
            f"Epoch {ep:3d}/{total}  "
            f"train_mse={float(t_mse):.4f}  train_pcc={float(t_pcc):.4f}  "
            f"val_mse={float(v_mse):.4f}  val_pcc={float(v_pcc):.4f}  "
            f"lr={lr:.2e}",
            flush=True,
        )

# ── Helpers ───────────────────────────────────────────────────────────────────
_default_epochs = {"histogene": 100, "stnet": 100, "uni": 50, "wsuni": 50}
_default_bs     = {"histogene": 1,   "stnet": 32,  "uni": 16, "wsuni": 16}


class HisToGeneSlideDataset(Dataset):
    """Adapt ``HER2ST`` from spot-level samples to HisToGene slide samples.

    HisToGene applies self-attention across all spots in a section, so one dataset
    item must represent one section.  The original HER2ST loader returns a 224 px
    patch per spot, whereas this implementation of HisToGene was built with
    112 px patches.  We take the centred 112 px crop before flattening it.
    """
    def __init__(self, spot_dataset, section_indices):
        self.spot_dataset = spot_dataset
        self.section_indices = section_indices

    def __len__(self):
        return len(self.section_indices)

    def __getitem__(self, index):
        patches, locations, expressions = [], [], []
        for spot_index in self.section_indices[index]:
            item = self.spot_dataset[spot_index]
            patch, location, expression = item[:3]
            # HER2ST patches are (3, 224, 224); HisToGene expects 3 * 112 * 112.
            h, w = patch.shape[-2:]
            top, left = (h - 112) // 2, (w - 112) // 2
            patch = patch[:, top:top + 112, left:left + 112]
            patches.append(patch.flatten())
            locations.append(location.long().clamp(0, 63))
            expressions.append(expression)
        return torch.stack(patches), torch.stack(locations), torch.stack(expressions)

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


def split_histo_train_val(ds_aug, ds_noaug):
    """Return slide-level train/validation datasets for HisToGene."""
    val_name = sorted(ds_aug.names)[0]
    sections = []
    start = 0
    for i, end in enumerate(ds_aug.cumlen):
        sections.append((ds_aug.id2name[i], list(range(start, int(end)))))
        start = int(end)
    train_sections = [indices for name, indices in sections if name != val_name]
    val_sections = [indices for name, indices in sections if name == val_name]
    print(f"  Val slide : {val_name} ({len(val_sections[0])} spots) | "
          f"Train slides: {len(train_sections)}")
    return (HisToGeneSlideDataset(ds_aug, train_sections),
            HisToGeneSlideDataset(ds_noaug, val_sections))


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
    if n_gpus > 1 and not skip_train:
        # HisToGene contains an intentionally unused normalisation module;
        # enabling this DDP mode is required to train it across two GPUs.
        strategy = "ddp_find_unused_parameters_true"
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
        "histogene": "valid_loss",
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
        if mode == "histogene":
            train_subset, val_subset = split_histo_train_val(ds_aug, ds_noaug)
            # Sections have different spot counts, therefore they cannot be
            # stacked together.  One complete section is one training sample.
            train_loader = DataLoader(train_subset, batch_size=1,
                                      num_workers=0, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=1,
                                    num_workers=0, shuffle=False)
        else:
            train_subset, val_subset = split_train_val(ds_aug, ds_noaug)
            train_loader = DataLoader(train_subset, batch_size=bs,
                                      num_workers=0, shuffle=True)
            val_loader   = DataLoader(val_subset, batch_size=bs,
                                      num_workers=0, shuffle=False,
                                      collate_fn=collate_drop_center)

        if mode == "histogene":
            model = HisToGene(patch_size=112, n_layers=8, n_genes=n_genes,
                              learning_rate=lr, max_epochs=max_ep)
        elif mode == "stnet":
            model = STModel(n_genes=n_genes, learning_rate=lr, max_epochs=max_ep)

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            max_epochs=max_ep,
            logger=logger,
            log_every_n_steps=10,
            gradient_clip_val=1.0,
            precision="16-mixed" if accelerator == "gpu" else "32-true",
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[checkpoint_cb, early_stop_cb, EpochProgressBar()],
        )
        trainer.fit(model, train_loader, val_loader)
        ckpt_path = checkpoint_cb.best_model_path
        print(f"  Best checkpoint: {ckpt_path}")
        print(f"  Best val loss  : {checkpoint_cb.best_model_score:.4f}"
              if hasattr(checkpoint_cb, "best_model_score") else "")

    else:
        if ckpt_path is None:
            raise ValueError(f"--skip_train yêu cầu --ckpt_path cho mode={mode}")

    # Every DDP rank trains; only rank zero may perform the canonical inference
    # and write CSV/figures.  Other ranks wait so modes stay in lockstep.
    if n_gpus > 1 and not skip_train:
        trainer.strategy.barrier()
        if not trainer.is_global_zero:
            trainer.strategy.barrier()
            return None

    # ── PREDICT ───────────────────────────────────────────────────────────────
    print(f"\n  [PREDICT] Load: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = HER2ST(train=False, fold=fold)
    label        = test_dataset.label[test_dataset.names[0]]

    if mode == "histogene":
        m = HisToGene.load_from_checkpoint(
            ckpt_path, patch_size=112, n_layers=8, n_genes=n_genes,
            learning_rate=lr, max_epochs=max_ep)
        test_loader = DataLoader(test_dataset, batch_size=1,
                                 num_workers=0, shuffle=False)
        adata_pred, adata_gt = histogene_predict(m, test_loader, device=device)

    elif mode == "stnet":
        m = STModel.load_from_checkpoint(
            ckpt_path, n_genes=n_genes, learning_rate=lr, max_epochs=max_ep)
        test_loader = DataLoader(test_dataset, batch_size=bs,
                                 num_workers=0, shuffle=False)
        adata_pred, adata_gt = stnet_predict(m, test_loader, device=device)

    # ── Common, fair evaluation ───────────────────────────────────────────────
    g = list(np.load("data/her_hvg_cut_1000.npy", allow_pickle=True))
    adata_visual, metrics = evaluate_her2st_predictions(
        adata_pred, adata_gt, g, label=label, n_clusters=4)
    R, p_values = metrics["R"], metrics["p_values"]
    Spearman, spearman_pvalues = metrics["Spearman"], metrics["spearman_pvalues"]
    MSE, MAE, RMSE, morans = metrics["MSE"], metrics["MAE"], metrics["RMSE"], metrics["morans"]
    mean_pcc, median_pcc = metrics["pearson"], metrics["median_pearson"]
    mean_spearman, mean_rmse, mean_mae = metrics["spearman"], metrics["rmse"], metrics["mae"]
    mean_mi_pred, mean_mi_gt = metrics["morans_i_pred"], metrics["morans_i_gt"]
    ARI, NMI = metrics["ARI"], metrics["NMI"]

    print(f"\n  [EVAL] {mode.upper()} fold={fold}")

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

    sc.pl.spatial(adata_visual, img=None, color="kmeans", spot_size=112,
                  frameon=False, legend_loc=None, title=None, show=False)
    plt.gca().set_title("")
    plt.savefig(f"figures/kmeans/{mode.upper()}_kmeans_fold{fold}.png",
                dpi=300, bbox_inches="tight", transparent=True)
    plt.clf(); plt.close()

    sc.pl.spatial(adata_visual, img=None, color="FASN", spot_size=112,
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
        "eval_protocol": PROTOCOL_NAME,
        "split_rule":    "LOOCV test=fold; validation=first alphabetical train slide",
        "n_genes":       n_genes,
        "max_epochs":    max_ep,
        "learning_rate": lr,
        "optimizer":     "AdamW(weight_decay=1e-4)",
        "scheduler":     "CosineAnnealingLR(T_max=max_epochs,eta_min=1e-6)",
        "batch_size":    1 if mode == "histogene" else bs,
        "seed":          42,
        "n_gpus":        n_gpus,
        "precision":     "16-mixed" if torch.cuda.is_available() else "32-true",
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

    if n_gpus > 1 and not skip_train:
        trainer.strategy.barrier()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
ALL_MODES = ["histogene", "stnet"]  # UNI/WSUNI require a distinct multi-scale cached dataset.
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
