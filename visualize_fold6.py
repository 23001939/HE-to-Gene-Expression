"""
visualize_fold6.py -- Load checkpoint Light-HGGEP fold 6 và vẽ 3 cột cho 1 section test:
  Cột 1: ảnh đầu vào (ghép các patch H&E của từng spot theo tọa độ grid)
  Cột 2: ảnh dự đoán biểu hiện gene (scatter spot, màu = giá trị pred)
  Cột 3: ảnh ground-truth biểu hiện gene (scatter spot, màu = giá trị gt)

Dùng riêng với checkpoint:
  model_ckpts/lighthggep_fold6_epoch=86_val_loss=0.6360.ckpt

Chạy:
  python visualize_fold6.py
  python visualize_fold6.py --gene FASN
  python visualize_fold6.py --gene 0 --section A6
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import scanpy as sc

import sys
WORKDIR = os.path.dirname(os.path.abspath(__file__))
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)

from dataset import LightHGGEP_HER2ST
from models.LightHGGEP import LightHGGEP
# [SỬA] KHÔNG import từ run_pipeline (nó chạy train/clone khi import). Định nghĩa
# local các hằng số + helper cần thiết cho test loader.
BATCH_SIZE = 32
K_NEIGHBORS = 4
NUM_WORKERS = 0

from torch.utils.data import Sampler
import math

class SectionBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.section_indices = {}
        start = 0
        for i, length in enumerate(dataset.lengths):
            name = dataset.id2name[i]
            self.section_indices[name] = list(range(start, start + length))
            start += length
        self.section_names = list(self.section_indices.keys())
        self.steps_per_epoch = sum(math.ceil(len(v) / self.batch_size)
                                    for v in self.section_indices.values())
    def __iter__(self):
        section_names = self.section_names.copy()
        if self.shuffle:
            import random; random.shuffle(section_names)
        for name in section_names:
            idxs = list(self.section_indices[name])
            if self.shuffle:
                import random; random.shuffle(idxs)
            for s in range(0, len(idxs), self.batch_size):
                yield idxs[s:s + self.batch_size]
    def __len__(self):
        return self.steps_per_epoch

def section_collate_fn(batch):
    is_train = (len(batch[0]) == 5)
    sec_pos = 3 if is_train else 4
    section_names = [b[sec_pos] for b in batch]
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

CKPT = "model_ckpts/lighthggep_fold6_epoch=86_val_loss=0.6360.ckpt"
FOLD = 6  # checkpoint này train/eval trên LOOCV fold 6


def main():
    _p = argparse.ArgumentParser()
    _p.add_argument("--gene", default="FASN",
                    help="Tên gene hoặc chỉ số int. Mặc định FASN (nếu có).")
    _p.add_argument("--section", default=None,
                    help="Section test cụ thể (vd A6). Mặc định lấy test section của fold.")
    _p.add_argument("--ckpt", default=CKPT)
    args = _p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- Dataset test fold 6 (cung cấp A_norm cache + spot coords + gt) ---
    test_dataset = LightHGGEP_HER2ST(train=False, fold=FOLD, k_neighbors=K_NEIGHBORS)
    n_genes = len(test_dataset.gene_set)
    print(f"  n_genes = {n_genes}, test_section = {test_dataset.names}")

    # --- Load model từ checkpoint ---
    model = LightHGGEP.load_from_checkpoint(
        args.ckpt,
        n_genes=n_genes,
        k_neighbors=K_NEIGHBORS,
        learning_rate=1e-4,
        max_epochs=100,
        cnn_chunk=BATCH_SIZE,
    )
    model = model.to(device)
    model.eval()

    # Set graph cho model (Spatial SGC cần A_norm của test section)
    for section, A_norm in test_dataset.A_norm_cache.items():
        model.set_graph(section, torch.from_numpy(A_norm).float())

    # --- Predict ---
    sampler = SectionBatchSampler(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    loader = DataLoader(test_dataset, batch_sampler=sampler,
                       collate_fn=section_collate_fn,
                       num_workers=NUM_WORKERS,
                       pin_memory=torch.cuda.is_available(),
                       persistent_workers=False,
                       timeout=(180 if NUM_WORKERS > 0 else 0))

    from predict import lighthggep_predict
    adata_pred, adata_gt = lighthggep_predict(model, loader, device=device)

    # --- Lấy patches đầu vào (để ghép ảnh cột 1) ---
    # Dùng lại loader, thu thập patch + loc (grid) + center
    all_patches, all_loc, all_centers = [], [], []
    for batch in loader:
        patch_3ch, positions, exp, centers, section_name, local_indices = batch
        all_patches.append(patch_3ch.cpu())
        all_loc.append(positions.cpu())
        all_centers.append(centers.cpu())
    patches = torch.cat(all_patches, dim=0).numpy()   # (N, 3, H, W)
    locs = torch.cat(all_loc, dim=0).numpy()          # (N, 2) grid coords
    centers = torch.cat(all_centers, dim=0).numpy()   # (N, 2) pixel coords

    # --- Chọn gene ---
    gene_names = list(test_dataset.gene_set)
    if isinstance(args.gene, str) and args.gene in gene_names:
        gidx = gene_names.index(args.gene)
        gname = args.gene
    else:
        try:
            gidx = int(args.gene)
        except ValueError:
            gidx = 0
        gname = gene_names[gidx] if 0 <= gidx < len(gene_names) else f"gene{gidx}"

    pred_vals = adata_pred.X[:, gidx].astype(float)
    gt_vals = adata_gt.X[:, gidx].astype(float)

    # --- Cột 1: ảnh đầu vào (ghép patch H&E theo grid loc) ---
    # loc là tọa độ grid float; nhân tỉ lệ để vừa canvas.
    locs_xy = locs - locs.min(axis=0)
    span = locs_xy.max(axis=0)
    span = np.where(span == 0, 1, span)
    H, W = patches.shape[-2:]
    # scale grid sang pixel canvas (mỗi đơn vị grid ~ 1.2 patch)
    grid_scale = 1.2
    canvas_h = int(span[1] * grid_scale * H / max(span) * 2 + H)
    canvas_w = int(span[0] * grid_scale * W / max(span) * 2 + W)
    canvas_h, canvas_w = max(canvas_h, H), max(canvas_w, W)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.float32)

    for i in range(patches.shape[0]):
        gx, gy = locs_xy[i]
        px = int(gx / span[0] * (canvas_w - W)) if span[0] > 0 else canvas_w // 2
        py = int(gy / span[1] * (canvas_h - H)) if span[1] > 0 else canvas_h // 2
        # unnormalize patch (mean/std ImageNet) để hiển thị tự nhiên
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img = patches[i] * std + mean
        img = np.clip(img, 0, 1).transpose(1, 2, 0)
        canvas[py:py + H, px:px + W] = img

    # --- Vẽ 3 cột ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(canvas)
    axes[0].set_title(f"(1) Input H&E patches\n{patches.shape[0]} spots")
    axes[0].axis("off")

    # Cột 2: prediction (scatter spot tại center, màu = pred)
    sc = axes[1].scatter(centers[:, 0], -centers[:, 1], c=pred_vals, cmap="magma",
                         s=40, edgecolors="k", linewidths=0.3)
    axes[1].set_title(f"(2) Prediction: {gname}\nmean={pred_vals.mean():.3f}")
    axes[1].invert_yaxis()
    axes[1].axis("equal")
    axes[1].axis("off")
    plt.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)

    # Cột 3: ground truth
    sc2 = axes[2].scatter(centers[:, 0], -centers[:, 1], c=gt_vals, cmap="magma",
                          s=40, edgecolors="k", linewidths=0.3)
    axes[2].set_title(f"(3) Ground Truth: {gname}\nmean={gt_vals.mean():.3f}")
    axes[2].invert_yaxis()
    axes[2].axis("equal")
    axes[2].axis("off")
    plt.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f"Light-HGGEP fold {FOLD} | section {test_dataset.names[0]} | gene {gname}",
                 fontsize=14)
    plt.tight_layout()
    out = f"figures/fold{FOLD}_viz_{gname}.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")
    print(f"  PCC ({gname}): pred vs gt = {np.corrcoef(pred_vals, gt_vals)[0,1]:.4f}")


if __name__ == "__main__":
    main()
