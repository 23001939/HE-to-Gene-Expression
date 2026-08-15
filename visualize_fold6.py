import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Sampler
import math
import sys
from PIL import Image

# Bỏ qua giới hạn kích thước ảnh của PIL (phòng trường hợp ảnh WSI quá lớn)
Image.MAX_IMAGE_PIXELS = None

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

def find_image_path(img_dir, section_name):
    """Hàm hỗ trợ tìm file ảnh gốc với các định dạng phổ biến."""
    for ext in ['.jpg', '.png', '.tif', '.jpeg']:
        path = os.path.join(img_dir, f"{section_name}{ext}")
        if os.path.exists(path):
            return path
        # Thử trường hợp viết hoa
        path_upper = os.path.join(img_dir, f"{section_name}{ext.upper()}")
        if os.path.exists(path_upper):
            return path_upper
    return None

def main():
    _p = argparse.ArgumentParser()
    _p.add_argument("--gene", default="FASN")
    _p.add_argument("--section", default=None, help="Section test cụ thể (vd A6).")
    _p.add_argument("--ckpt", default=CKPT)
    # THÊM THAM SỐ: Thư mục chứa ảnh gốc
    _p.add_argument("--img_dir", default="data/ST-imgs", help="Đường dẫn đến thư mục chứa ảnh H&E gốc")
    _p.add_argument("--img_path", default=None, help="Đường dẫn TRỰC TIẾP đến file ảnh")
    args = _p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- Dataset ---
    test_dataset = LightHGGEP_HER2ST(train=False, fold=FOLD, k_neighbors=K_NEIGHBORS)
    n_genes = len(test_dataset.gene_set)
    
    target_sections = [args.section] if args.section else test_dataset.names
    current_section = target_sections[0]
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

    # --- Lấy tọa độ centers của section ---
    all_centers = []
    for batch in loader:
        _, _, _, centers, section_name, _ = batch
        if section_name in target_sections:
            all_centers.append(centers.cpu())
            
    if not all_centers:
        print(f"Không tìm thấy dữ liệu cho section {target_sections}. Thoát.")
        return

    centers = torch.cat(all_centers, dim=0).numpy()
    n_spots = centers.shape[0]
    
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
    # CHUẨN BỊ ẢNH GỐC VÀ TỌA ĐỘ VẼ
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Tính toán Bounding Box của mô dựa trên tọa độ spots (có thêm padding)
    pad = 200 # Mở rộng khung viền thêm 200 pixels để ảnh không bị cắt quá sát
    min_x, max_x = centers[:, 0].min() - pad, centers[:, 0].max() + pad
    min_y, max_y = centers[:, 1].min() - pad, centers[:, 1].max() + pad

    # Load ảnh gốc
    img_path = find_image_path(args.img_dir, current_section)
    if img_path:
        print(f"Loaded original image: {img_path}")
        orig_img = Image.open(img_path)
        axes[0].imshow(orig_img)
    else:
        print(f"Warning: Không tìm thấy file ảnh cho '{current_section}' trong '{args.img_dir}'.")
        axes[0].text(0.5, 0.5, f"Không tìm thấy ảnh gốc:\nThư mục {args.img_dir}", 
                     ha='center', va='center', fontsize=12)
        axes[0].set_facecolor('#f0f0f0')

    axes[0].set_title(f"(1) Input H&E Original Image\n{n_spots} spots")
    axes[0].axis("off")

    # =========================================================================
    # VẼ CỘT 2 VÀ CỘT 3
    # =========================================================================
    # Cột 2
    sc = axes[1].scatter(centers[:, 0], centers[:, 1], c=pred_vals, cmap="magma",
                         s=40, edgecolors="k", linewidths=0.3)
    axes[1].set_title(f"(2) Prediction: {gname}\nmean={pred_vals.mean():.3f}")
    axes[1].axis("off")
    plt.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)

    # Cột 3
    sc2 = axes[2].scatter(centers[:, 0], centers[:, 1], c=gt_vals, cmap="magma",
                          s=40, edgecolors="k", linewidths=0.3)
    axes[2].set_title(f"(3) Ground Truth: {gname}\nmean={gt_vals.mean():.3f}")
    axes[2].axis("off")
    plt.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

    # ĐỒNG BỘ GIỚI HẠN HIỂN THỊ (BOUNDING BOX) CHO CẢ 3 CỘT
    for ax in axes:
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(max_y, min_y) # Trục Y của hình ảnh luôn đi từ trên xuống dưới (invert)
        ax.set_aspect('equal')    # Đảm bảo tỉ lệ khung hình chuẩn, không bị bóp méo

    sec_title = current_section
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
