from dataset import HER2ST
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
import numpy as np
from STLite import STLiteConfig, STLite

# ==========================================
# 1. KHỞI TẠO DATASET VÀ MÔ HÌNH
# ==========================================
# Khởi tạo dataset train
train_dataset = HER2ST(train=True)

# Cập nhật Config khớp với dataset (1000 gen)
cfg = STLiteConfig(
    micro_patch_size=224, 
    in_channels=3,
    token_dim=64,
    num_pathways=32,       # 32 pathway (hoặc tùy ý)
    total_genes=785,      # QUAN TRỌNG: Cập nhật thành 1000 theo file của bạn
    gcn_k_neighbors=6,
    gcn_num_layers=2
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = STLite(cfg).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ==========================================
# 2. CHUẨN BỊ PATHWAY TARGET (Dùng PCA trực tiếp trên Dataset)
# ==========================================
print("Đang tạo Pathway Target bằng PCA cho 785 gen...")
pca = PCA(n_components=cfg.num_pathways)
# Lấy toàn bộ ma trận biểu hiện gen từ exp_dict của bạn để fit PCA
all_exp = np.concatenate([exp for exp in train_dataset.exp_dict.values()], axis=0)
pca.fit(all_exp)

# ==========================================
# 3. VÒNG LẶP HUẤN LUYỆN (Gom theo từng Slide)
# ==========================================
epochs = 50
model.train()

for epoch in range(epochs):
    epoch_loss = 0.0
    
    # Duyệt qua từng slide dựa vào mảng cumlen của bạn
    for i in range(len(train_dataset.names)):
        # Xác định chỉ số (index) bắt đầu và kết thúc của slide thứ i
        start_idx = 0 if i == 0 else train_dataset.cumlen[i-1]
        end_idx = train_dataset.cumlen[i]
        
        slide_patches, slide_locs, slide_exps = [], [], []
        
        # Lấy toàn bộ spot của slide này
        for idx in range(start_idx, end_idx):
            patch, loc, exp = train_dataset[idx]
            slide_patches.append(patch)
            slide_locs.append(loc)
            slide_exps.append(exp)
            
        # Ghép thành các tensor có shape (N, ...)
        images = torch.stack(slide_patches, dim=0)   # (N, 3, 224, 224)
        coords = torch.stack(slide_locs, dim=0)      # (N, 2)
        genes_gt = torch.stack(slide_exps, dim=0)    # (N, 1000)
        
        # Thêm chiều Batch (B=1) để đưa vào ST-Lite
        images = images.unsqueeze(0).to(device)      # (1, N, 3, 224, 224)
        coords = coords.unsqueeze(0).to(device)      # (1, N, 2)
        genes_gt = genes_gt.unsqueeze(0).to(device)  # (1, N, 1000)
        
        # Tạo Pathway Target cho slide hiện tại bằng PCA đã fit ở trên
        flat_genes = genes_gt.view(-1, cfg.total_genes).cpu().numpy()
        flat_pathways = pca.transform(flat_genes)
        pathways_gt = torch.tensor(flat_pathways, dtype=torch.float32).unsqueeze(0).to(device) # (1, N, 32)
        
        # --- Huấn luyện ---
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, coords)
        
        # Tính Loss
        loss, loss_gene, loss_pathway = model.compute_loss(outputs, genes_gt, pathways_gt)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss/len(train_dataset.names):.4f}")

import torch
import numpy as np
import pandas as pd

# ==========================================
# 1. KHỞI TẠO TẬP TEST
# ==========================================
# Khởi tạo dataset test (train=False)
test_dataset = HER2ST(train=False)

# Chuyển mô hình sang chế độ đánh giá (tắt Dropout, BatchNorm tĩnh...)
model.eval()

# Danh sách lưu trữ dự đoán và nhãn gốc của toàn bộ các spot trong tập test
all_preds = []
all_trues = []

print(f"Bắt đầu đánh giá trên {len(test_dataset.names)} slide test...")

# ==========================================
# 2. VÒNG LẶP ĐÁNH GIÁ (INFERENCE)
# ==========================================
# Tắt tính toán gradient để tiết kiệm VRAM và tăng tốc
with torch.no_grad():
    for i in range(len(test_dataset.names)):
        start_idx = 0 if i == 0 else test_dataset.cumlen[i-1]
        end_idx = test_dataset.cumlen[i]
        
        slide_patches, slide_locs, slide_exps = [], [], []
        
        for idx in range(start_idx, end_idx):
            # Khi train=False, Dataset trả về 4 giá trị (patch, loc, exp, center)
            patch, loc, exp, center = test_dataset[idx] 
            
            slide_patches.append(patch)
            slide_locs.append(loc)
            slide_exps.append(exp)
            
        # Ghép thành các tensor có shape (N, ...)
        images = torch.stack(slide_patches, dim=0).unsqueeze(0).to(device)  # (1, N, 3, 224, 224)
        coords = torch.stack(slide_locs, dim=0).unsqueeze(0).to(device)     # (1, N, 2)
        genes_gt = torch.stack(slide_exps, dim=0).numpy()                   # Giữ lại trên CPU (N, 785)
        
        # Đưa qua mô hình
        outputs = model(images, coords)
        
        # Lấy dự đoán gen, bỏ chiều Batch (1, N, 785) -> (N, 785), chuyển về Numpy
        preds = outputs["gene_expression"].squeeze(0).cpu().numpy()
        
        all_preds.append(preds)
        all_trues.append(genes_gt)
        
        print(f"Đã xử lý xong slide {test_dataset.names[i]} ({end_idx - start_idx} spots)")

# ==========================================
# 3. TÍNH TOÁN METRICS (MSE & PCC)
# ==========================================
# Gộp tất cả các điểm (spots) từ tất cả các slide test lại với nhau
all_preds_np = np.concatenate(all_preds, axis=0)
all_trues_np = np.concatenate(all_trues, axis=0)

# Tính MSE (Mean Squared Error)
mse = np.mean((all_preds_np - all_trues_np) ** 2)

# Tính PCC (Pearson Correlation Coefficient) cho từng gen
# Công thức: Covariance(x, y) / (StdDev(x) * StdDev(y))
vx = all_preds_np - all_preds_np.mean(axis=0, keepdims=True)
vy = all_trues_np - all_trues_np.mean(axis=0, keepdims=True)
num = (vx * vy).sum(axis=0)
den = np.sqrt((vx ** 2).sum(axis=0)) * np.sqrt((vy ** 2).sum(axis=0)) + 1e-8
pcc_per_gene = num / den

mean_pcc = np.nanmean(pcc_per_gene)
median_pcc = np.nanmedian(pcc_per_gene)

# ==========================================
# 4. HIỂN THỊ KẾT QUẢ
# ==========================================
print("\n" + "="*50)
print("KẾT QUẢ ĐÁNH GIÁ TẬP TEST - ST-Lite")
print("="*50)
print(f"Tổng số spot test đã đánh giá: {all_trues_np.shape[0]}")
print(f"MSE tổng thể                 : {mse:.4f}")
print(f"PCC trung bình (785 gen)     : {mean_pcc:.4f}")
print(f"PCC trung vị (785 gen)       : {median_pcc:.4f}")

# (Tùy chọn) Tìm ra 5 gen dự đoán tốt nhất
best_genes_indices = np.argsort(pcc_per_gene)[::-1][:5]
print("\nTop 5 gen dự đoán tốt nhất (theo index):")
for idx in best_genes_indices:
    print(f" - Gen index {idx}: PCC = {pcc_per_gene[idx]:.4f}")
