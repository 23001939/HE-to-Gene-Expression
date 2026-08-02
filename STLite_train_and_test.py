from dataset import HER2ST
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
import numpy as np
from STLite import STLiteConfig, STLite

# ==========================================
# 1. KHỞI TẠO DATASET VÀ MÔ HÌNH
# ==========================================
train_dataset = HER2ST(train=True)

# total_genes=785: gene list thực tế sau khi lọc từ her_hvg_cut_1000.npy
# gcn_distance_sigma=500: phù hợp với pixel coordinates (thường 200–2000px giữa các spot)
cfg = STLiteConfig(
    micro_patch_size=224,
    in_channels=3,
    token_dim=64,
    num_pathways=32,
    total_genes=785,
    gcn_k_neighbors=6,
    gcn_num_layers=2,
    gcn_distance_sigma=500.0,   # pixel coords scale, ~khoảng cách trung bình giữa các spot
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = STLite(cfg).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ==========================================
# 2. CHUẨN BỊ PATHWAY TARGET (PCA fit trên tập train)
# ==========================================
print("Đang tạo Pathway Target bằng PCA cho 785 gen...")
pca = PCA(n_components=cfg.num_pathways)
all_exp = np.concatenate([exp for exp in train_dataset.exp_dict.values()], axis=0)
pca.fit(all_exp)

# ==========================================
# 3. VÒNG LẶP HUẤN LUYỆN (Gom theo từng Slide)
# ==========================================
epochs = 50
best_loss = float('inf')
model.train()

for epoch in range(epochs):
    epoch_loss = 0.0

    for i in range(len(train_dataset.names)):
        start_idx = 0 if i == 0 else train_dataset.cumlen[i - 1]
        end_idx = train_dataset.cumlen[i]

        slide_patches, slide_centers, slide_exps = [], [], []

        for idx in range(start_idx, end_idx):
            # HER2ST trả về (patch, loc, exp, center) cho cả train lẫn test
            patch, loc, exp, center = train_dataset[idx]
            slide_patches.append(patch)
            slide_centers.append(center)   # pixel coords — dùng cho GCN
            slide_exps.append(exp)

        # Ghép thành tensor (N, ...)
        images   = torch.stack(slide_patches, dim=0)    # (N, 3, 224, 224)
        coords   = torch.stack(slide_centers, dim=0)    # (N, 2)  — pixel coords
        genes_gt = torch.stack(slide_exps, dim=0)       # (N, 785)

        # Thêm chiều Batch (B=1)
        images   = images.unsqueeze(0).to(device)       # (1, N, 3, 224, 224)
        coords   = coords.unsqueeze(0).to(device)       # (1, N, 2)
        genes_gt = genes_gt.unsqueeze(0).to(device)     # (1, N, 785)

        # Tạo Pathway Target bằng PCA đã fit
        flat_genes    = genes_gt.view(-1, cfg.total_genes).cpu().numpy()
        flat_pathways = pca.transform(flat_genes)
        pathways_gt   = torch.tensor(flat_pathways, dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, 32)

        # --- Forward / Backward ---
        optimizer.zero_grad()
        outputs = model(images, coords)
        loss, loss_gene, loss_pathway = model.compute_loss(outputs, genes_gt, pathways_gt)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_dataset.names)
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

    # Lưu checkpoint tốt nhất
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, 'stlite_best.pth')

# Lưu checkpoint cuối cùng
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
}, 'stlite_last.pth')
print("Đã lưu model: stlite_best.pth và stlite_last.pth")

# ==========================================
# 4. KHỞI TẠO TẬP TEST
# ==========================================
import pandas as pd

test_dataset = HER2ST(train=False)

# Load checkpoint tốt nhất trước khi đánh giá
checkpoint = torch.load('stlite_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Đã load model tốt nhất (epoch {checkpoint['epoch']}, loss={checkpoint['loss']:.4f})")

model.eval()

all_preds = []
all_trues = []

print(f"Bắt đầu đánh giá trên {len(test_dataset.names)} slide test...")

# ==========================================
# 5. VÒNG LẶP ĐÁNH GIÁ (INFERENCE)
# ==========================================
with torch.no_grad():
    for i in range(len(test_dataset.names)):
        start_idx = 0 if i == 0 else test_dataset.cumlen[i - 1]
        end_idx = test_dataset.cumlen[i]

        slide_patches, slide_centers, slide_exps = [], [], []

        for idx in range(start_idx, end_idx):
            # HER2ST trả về (patch, loc, exp, center) — nhất quán với train
            patch, loc, exp, center = test_dataset[idx]
            slide_patches.append(patch)
            slide_centers.append(center)   # pixel coords — dùng cho GCN
            slide_exps.append(exp)

        images   = torch.stack(slide_patches, dim=0).unsqueeze(0).to(device)   # (1, N, 3, 224, 224)
        coords   = torch.stack(slide_centers, dim=0).unsqueeze(0).to(device)   # (1, N, 2)
        genes_gt = torch.stack(slide_exps, dim=0).numpy()                      # (N, 785) CPU

        outputs = model(images, coords)
        preds   = outputs["gene_expression"].squeeze(0).cpu().numpy()           # (N, 785)

        all_preds.append(preds)
        all_trues.append(genes_gt)

        print(f"Đã xử lý xong slide {test_dataset.names[i]} ({end_idx - start_idx} spots)")

# ==========================================
# 6. TÍNH TOÁN METRICS (MSE & PCC)
# ==========================================
all_preds_np = np.concatenate(all_preds, axis=0)
all_trues_np = np.concatenate(all_trues, axis=0)

# MSE
mse = np.mean((all_preds_np - all_trues_np) ** 2)

# PCC per-gene: Covariance(pred, true) / (std_pred * std_true)
vx = all_preds_np - all_preds_np.mean(axis=0, keepdims=True)
vy = all_trues_np - all_trues_np.mean(axis=0, keepdims=True)
num = (vx * vy).sum(axis=0)
den = np.sqrt((vx ** 2).sum(axis=0)) * np.sqrt((vy ** 2).sum(axis=0)) + 1e-8
pcc_per_gene = num / den

mean_pcc   = np.nanmean(pcc_per_gene)
median_pcc = np.nanmedian(pcc_per_gene)

# ==========================================
# 7. HIỂN THỊ KẾT QUẢ
# ==========================================
print("\n" + "=" * 50)
print("KẾT QUẢ ĐÁNH GIÁ TẬP TEST - ST-Lite")
print("=" * 50)
print(f"Tổng số spot test đã đánh giá: {all_trues_np.shape[0]}")
print(f"MSE tổng thể                 : {mse:.4f}")
print(f"PCC trung bình (785 gen)     : {mean_pcc:.4f}")
print(f"PCC trung vị  (785 gen)      : {median_pcc:.4f}")

best_genes_indices = np.argsort(pcc_per_gene)[::-1][:5]
print("\nTop 5 gen dự đoán tốt nhất (theo index):")
for idx in best_genes_indices:
    print(f" - Gen index {idx}: PCC = {pcc_per_gene[idx]:.4f}")
