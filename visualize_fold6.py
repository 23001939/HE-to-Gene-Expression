import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Sampler
import math
import sys

WORKDIR = os.path.dirname(os.path.abspath(__file__))
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)

from dataset import LightHGGEP_HER2ST
from models.LightHGGEP import LightHGGEP
from predict import lighthggep_predict

BATCH_SIZE = 32
K_NEIGHBORS = 4
NUM_WORKERS = 0
CKPT = "model_ckpts/lighthggep_fold6_epoch=86_val_loss=0.6360.ckpt"
FOLD = 6

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

def main():
    _p = argparse.ArgumentParser()
    _p.add_argument("--gene", default="FASN")
    _p.add_argument("--section", default=None, help="Section test cụ thể (vd A6).")
    _p.add_argument("--ckpt", default=CKPT)
    args = _p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- Dataset ---
    test_dataset = LightHGGEP_HER2ST(train=False, fold=FOLD, k_neighbors=K_NEIGHBORS)
    n_genes = len(test_dataset.gene_set)
    
    target_sections = [args.section] if args.section else test_dataset.names
    print(f"  n_genes = {n_genes}, target_sections = {target_sections}")

    # --- Model ---
    model = LightHGGEP.load_from_checkpoint(
        args.ckpt, n_genes=n_genes, k_neighbors=K_NEIGHBORS,
        learning_rate=1e-4, max_epochs=100, cnn_chunk=BATCH_SIZE,
    )
    model = model.to(device)
    model.eval()

    for section, A_norm in test_dataset.A_norm_cache.items():
        if section in target_sections:
            model.set_graph(section, torch.from_numpy(A_norm).float())

    sampler = SectionBatchSampler(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    loader = DataLoader(test_dataset, batch_sampler=sampler,
                       collate_fn=section_collate_fn,
                       num_workers=NUM_WORKERS,
                       pin_memory=torch.cuda.is_available())

    # --- Predict ---
    adata_pred, adata_gt = lighthggep_predict(model, loader, device=device)

    # --- Lấy patch của section cần vẽ ---
    all_patches, all_centers = [], []
    for batch in loader:
        patch_3ch, positions, exp, centers, section_name, local_indices = batch
        if section_name in target_sections:
            all_patches.append(patch_3ch.cpu())
            all_centers.append(centers.cpu())
            
    if not all_patches:
        print(f"Không tìm thấy dữ liệu cho section {target_sections}. Thoát.")
        return

    patches = torch.cat(all_patches, dim=0).numpy()
    centers = torch.cat(all_centers, dim=0).numpy()  # Sử dụng TỌA ĐỘ PIXEL THỰC TẾ
    n_spots = patches.shape[0]
    
    # --- Chọn gene ---
    gene_names = list(test_dataset.gene_set)
    if isinstance(args.gene, str) and args.gene in gene_names:
        gidx = gene_names.index(args.gene)
        gname = args.gene
    else:
        try:
            gidx = int(args.gene)
            gname = gene_names[gidx]
        except (ValueError, IndexError):
            gidx, gname = 0, gene_names[0]

    pred_vals = adata_pred.X[:n_spots, gidx].astype(float)
    gt_vals = adata_gt.X[:n_spots, gidx].astype(float)

    # =========================================================================
    # [FIX CỐT LÕI]: Cột 1 - Ghép ảnh đầu vào dựa trên tọa độ PIXEL (centers)
    # =========================================================================
    H, W = patches.shape[-2:]
    
    # `centers` đang là điểm trung tâm của patch. 
    # Ta suy ra góc trên-trái (top-left) của patch bằng cách trừ đi nửa chiều dài/rộng.
    top_lefts = centers.copy()
    top_lefts[:, 0] -= W / 2.0
    top_lefts[:, 1] -= H / 2.0
    
    # Chuẩn hóa để tọa độ nhỏ nhất bắt đầu từ (0,0) trên Canvas
    top_lefts -= top_lefts.min(axis=0)
    
    # Khởi tạo Canvas trắng (Tương đương kích thước bounding box bao trọn toàn bộ patch)
    canvas_w = int(np.ceil(top_lefts[:, 0].max() + W))
    canvas_h = int(np.ceil(top_lefts[:, 1].max() + H))
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.float32)

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    for i in range(n_spots):
        px = int(top_lefts[i, 0])
        py = int(top_lefts[i, 1])
        
        # Unnormalize ImageNet -> RGB [0, 1]
        img = patches[i] * std + mean
        img = np.clip(img, 0, 1).transpose(1, 2, 0)
        
        # Đổ ảnh vào đúng vị trí tọa độ
        canvas[py:py + H, px:px + W] = img

    # =========================================================================
    # VẼ 3 CỘT
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(canvas)
    axes[0].set_title(f"(1) Input H&E patches\n{n_spots} spots")
    axes[0].axis("off")

    # Cột 2
    sc = axes[1].scatter(centers[:, 0], centers[:, 1], c=pred_vals, cmap="magma",
                         s=40, edgecolors="k", linewidths=0.3)
    axes[1].set_title(f"(2) Prediction: {gname}\nmean={pred_vals.mean():.3f}")
    axes[1].invert_yaxis() 
    axes[1].axis("equal")
    axes[1].axis("off")
    plt.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)

    # Cột 3
    sc2 = axes[2].scatter(centers[:, 0], centers[:, 1], c=gt_vals, cmap="magma",
                          s=40, edgecolors="k", linewidths=0.3)
    axes[2].set_title(f"(3) Ground Truth: {gname}\nmean={gt_vals.mean():.3f}")
    axes[2].invert_yaxis()
    axes[2].axis("equal")
    axes[2].axis("off")
    plt.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

    sec_title = target_sections[0] if len(target_sections) == 1 else "All Test Sections"
    plt.suptitle(f"Light-HGGEP fold {FOLD} | section {sec_title} | gene {gname}", fontsize=14)
    plt.tight_layout()
    
    out = f"figures/fold{FOLD}_{sec_title}_viz_{gname}.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")
    print(f"  PCC ({gname}): pred vs gt = {np.corrcoef(pred_vals, gt_vals)[0,1]:.4f}")

if __name__ == "__main__":
    main()
