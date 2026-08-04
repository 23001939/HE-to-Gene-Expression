import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import pytorch_lightning as pl
from training_metrics import mean_gene_pearson

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LightHGGEP(pl.LightningModule):
    """
    Light-HGGEP: Kien truc sieu nhe cho Spatial Transcriptomics
    
    Cac thanh phan:
    1. Input: 4 kenh (RGB + Sobel Gradient)
    2. Backbone: 3 khoi Depthwise Separable Conv + GAP
    3. Cross-Scale Fusion: Linear(64*3 -> 128)
    4. Spatial SGC: K-NN graph + D^(-1/2) A D^(-1/2) + 2 layers
    5. Prediction Head: Linear(128 -> n_genes)
    """
    def __init__(self, n_genes=785, k_neighbors=4, learning_rate=1e-4, max_epochs=100, cnn_chunk=64):
        super().__init__()
        self.save_hyperparameters()
        
        self.n_genes = n_genes
        self.k_neighbors = k_neighbors
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.cnn_chunk = cnn_chunk  # sẽ được set lại ngay dưới nếu bạn truyền vào
        # Stage 1: Low-level (nuclei features)
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(3, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )
        
        # Stage 2: Mid-level (tissue structure)
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )
        
        # Stage 3: High-level (micro-environment)
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Cross-Scale Fusion (Eq. 1)
        self.cross_scale_fusion = nn.Linear(64 * 3, 128)
        
        # Spatial SGC Weight (Eq. 2)
        self.sgc_weight = nn.Linear(128, 128, bias=False)
        
        # Prediction Head (Eq. 3)
        self.pred_head = nn.Linear(128, n_genes)
        
        # Cache cho ma tran ke cua tung section
        self.A_norm_cache = {}
        
        self._initialize_weights()
    
    def set_graph(self, section_name, A_norm):
        """
        Set pre-computed normalized adjacency matrix cho mot section
        A_norm = D^(-1/2) A D^(-1/2) voi A co self-loops
        """
        self.A_norm_cache[section_name] = A_norm
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def compute_sobel_gradient(self, rgb_image):
        """
        Tinh Sobel gradient tu anh RGB (batch)
        Input: (B, 3, H, W)
        Output: (B, 1, H, W)
        """
        # Chuyen sang grayscale
        gray = 0.299 * rgb_image[:, 0:1, :, :] + 0.587 * rgb_image[:, 1:2, :, :] + 0.114 * rgb_image[:, 2:3, :, :]
        
        # Sobel filters
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).to(rgb_image.device)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).to(rgb_image.device)
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        gradient = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # Normalize gradient
        grad_max = gradient.view(gradient.size(0), -1).max(dim=1, keepdim=True)[0]
        grad_max = grad_max.view(gradient.size(0), 1, 1, 1)
        gradient = gradient / (grad_max + 1e-8)
        
        return gradient
    
    def _cnn_chunk_forward(self, xb):
        """[MỚI] Tách riêng phần CNN của 1 chunk thành hàm riêng để dùng với
        torch.utils.checkpoint (xem forward() bên dưới) -- không đổi phép tính,
        chỉ tách hàm để checkpoint() có thể gọi lại (recompute) lúc backward."""
        f1 = self.stage1(xb)
        f1_gap = self.gap(f1).view(xb.size(0), -1)
        f2 = self.stage2(f1)
        f2_gap = self.gap(f2).view(xb.size(0), -1)
        f3 = self.stage3(f2)
        f3_gap = self.gap(f3).view(xb.size(0), -1)
        z_b = torch.cat([f1_gap, f2_gap, f3_gap], dim=1)
        return self.cross_scale_fusion(z_b)

    def forward(self, x, positions=None, section_name=None, local_indices=None):
        """
        [SỬA lỗi #1, sửa lại]: CNN (stage1-3 + fusion) chạy theo CHUNK nhỏ (self.cnn_chunk).
        LƯU Ý QUAN TRỌNG: chunk hoá đơn thuần (bản trước) KHÔNG giảm bộ nhớ đỉnh lúc
        backward -- autograd vẫn giữ lại đồ thị tính đạo hàm (bao gồm feature map
        224x224x64 kênh trước maxpool, ~12.8MB/patch) của MỌI chunk đã chạy cho đến khi
        backward() xong, vì z_spot_chunks giữ tham chiếu tới toàn bộ chuỗi tính toán đó.
        Với section ~2000 patch, cnn_chunk=32 -> ~62 chunk tích luỹ dần -> OOM giữa chừng
        (đúng như traceback: crash ở stage2, không phải chunk đầu). Fix: bọc
        _cnn_chunk_forward bằng torch.utils.checkpoint -- CHỈ giữ lại input/output của
        mỗi chunk, activation trung gian được TÍNH LẠI (không lưu) lúc backward, đổi lấy
        ~30-50% compute thêm để giảm mạnh bộ nhớ đỉnh. CHỈ áp dụng lúc train (cần
        backward); lúc eval/predict (torch.no_grad()) không cần checkpoint, chạy thẳng
        để không tốn compute recompute vô ích.
        Spatial SGC (Eq.2) vẫn áp dụng trên ĐỦ N embedding sau khi ghép các chunk lại
        -- nhờ vậy A_norm_full không còn bị cắt mất láng giềng thật như thiết kế batching cũ.
        """
        B = x.size(0)
        chunk = getattr(self, "cnn_chunk", 64)

        z_spot_chunks = []
        for start in range(0, B, chunk):
            xb = x[start:start + chunk]
            if self.training:
                # use_reentrant=False: API mới hơn, ổn định hơn với BatchNorm/nhiều input
                z_b = cp.checkpoint(self._cnn_chunk_forward, xb, use_reentrant=False)
            else:
                z_b = self._cnn_chunk_forward(xb)
            z_spot_chunks.append(z_b)
        z_spot = torch.cat(z_spot_chunks, dim=0)   # (N, 128) -- ĐỦ cả section, không bị cắt
    
        # Spatial SGC (Eq. 2) -- logic KHÔNG đổi, chỉ khác input z_spot giờ đủ N
        if section_name is not None and section_name in self.A_norm_cache:
            A_norm_full = self.A_norm_cache[section_name]
            A_norm_full = A_norm_full.to(x.device)
            if local_indices is not None:
                if torch.is_tensor(local_indices):
                    local_indices = local_indices.cpu().numpy()
                A_norm_batch = A_norm_full[local_indices][:, local_indices]
            else:
                A_norm_batch = A_norm_full
            z = z_spot
            for _ in range(2):
                z = A_norm_batch @ z
            z_hat = self.sgc_weight(z)
        else:
            z_hat = z_spot
    
        y_hat = self.pred_head(z_hat)
        return y_hat
    
    def training_step(self, batch, batch_idx):
        # Dataset tra ve: patch_3ch, positions, exp, section_name, local_indices
        patch_3ch, positions, exp, section_name, local_indices = batch
        y_hat = self(patch_3ch, positions, section_name, local_indices)
        loss = F.mse_loss(y_hat, exp)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', loss, on_epoch=True, sync_dist=True)
        self.log('train_pcc', mean_gene_pearson(y_hat, exp), on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        patch_3ch, positions, exp, section_name, local_indices = batch
        y_hat = self(patch_3ch, positions, section_name, local_indices)
        loss = F.mse_loss(y_hat, exp)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log('val_mse', loss, on_epoch=True, sync_dist=True)
        self.log('val_pcc', mean_gene_pearson(y_hat, exp), on_epoch=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        patch_3ch, positions, exp, centers, section_name, local_indices = batch
        y_hat = self(patch_3ch, positions, section_name, local_indices)
        loss = F.mse_loss(y_hat, exp)
        self.log('test_loss', loss)
        return y_hat, exp, centers
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        # Shared optimisation schedule used for the fair baseline comparison.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }
