import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, n_genes=785, k_neighbors=4, learning_rate=1e-4, max_epochs=100,
                 cnn_chunk=64, use_sgc=True, sgc_nonlinear=True, sgc_alpha=1.0, sgc_hops=2):
        super().__init__()
        self.save_hyperparameters()

        self.n_genes = n_genes
        self.k_neighbors = k_neighbors
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.cnn_chunk = cnn_chunk  # sẽ được set lại ngay dưới nếu bạn truyền vào
        self.use_sgc = use_sgc
        self.sgc_nonlinear = sgc_nonlinear
        # [TUNE] trong so dong gop cua SGC vao residual. CNN embedding da co san
        # cau truc khong gian (Moran's I pred~0.83 >> gt~0.23) nen SGC 2-hop
        # full (alpha=1) lam tron qua da -> PCC tut 0.259->0.189. Giam alpha
        # (vd 0.3) de SGC chi hieu chinh nhe, giu PCC cao nhung van giu kien
        # truc SGC cua paper.
        self.sgc_alpha = sgc_alpha
        # [TUNE] so lan lan truyen (hop) tren do thi KNN khong gian. SGC goc
        # thuan tuyen: A_norm^sgc_hops @ z_spot. Mac dinh 2-hop (paper). Tang
        # len de mo rong tam anh huong khong gian, giam xuong 1 de chi lay
        # truc tiep hang xom. Phai >= 1.
        self.sgc_hops = max(int(sgc_hops), 1)
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
        # [PHI TUYEN] goc la SGC thuan tuyen (Linear, khong activation). them
        # ReLU giua 2 hop de thanh GCN nhe -> co phi tuyen, hoc duoc bien/duc
        # dac trung khong gian thay vi chi lam tron. Tat bang sgc_nonlinear=False
        # de quay ve SGC goc.
        self.sgc_weight = nn.Linear(128, 128, bias=False)
        self.sgc_act = nn.ReLU(inplace=True)

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
    
    def forward(self, x, positions=None, section_name=None, local_indices=None):
        """
        [SỬA lỗi #1]: CNN (stage1-3 + fusion) chạy theo CHUNK nhỏ (self.cnn_chunk, mặc định
        64 patch/lần) -- feature map 224x224x64 kênh trước maxpool tốn ~12.8MB/patch, 1
        section ~2000 patch chạy 1 lần sẽ tốn ~25.7GB, không khả thi trên T4 16GB dù model
        ít tham số. Việc chunk KHÔNG đổi giá trị toán học (mỗi patch CNN độc lập với patch
        khác -- chỉ BatchNorm là tính thống kê theo từng chunk, nhưng đây CŨNG chính là cách
        BN đã hoạt động trước đây với batch=32 ngẫu nhiên, nên không phải hồi quy so với hiện
        tại). Spatial SGC (Eq.2) vẫn áp dụng trên ĐỦ N embedding sau khi ghép các chunk lại
        -- nhờ vậy A_norm_full không còn bị cắt mất láng giềng thật như thiết kế batching cũ.
        """
        # [CHUNK INPUT] x co the rat lon (full section = 2162 patches). Giu x tren
        # CPU, chi chuyen tung chunk len GPU de tranh OOM. z_spot tich luy tren
        # GPU (N x 128 nhe).
        dev = next(self.parameters()).device
        B = x.size(0)
        chunk = getattr(self, "cnn_chunk", 64)

        def _cnn_chunk_forward(xb):
            f1 = self.stage1(xb)
            f1_gap = self.gap(f1).view(xb.size(0), -1)
            f2 = self.stage2(f1)
            f2_gap = self.gap(f2).view(xb.size(0), -1)
            f3 = self.stage3(f2)
            f3_gap = self.gap(f3).view(xb.size(0), -1)
            z_b = torch.cat([f1_gap, f2_gap, f3_gap], dim=1)
            return self.cross_scale_fusion(z_b)

        # [CHECKPOINT] batch=697 (1 tile) -> toan bo activation graph cua 697
        # patch bi giu song cung luc (den khi cat z_spot) ~11.6GB -> OOM. Dung
        # gradient checkpoint tung chunk: chi giu output fusion (re nhe), activation
        # duoc tinh lai o backward. Peak chi con 1 chunk ~1.1GB. Van co gradient
        # ve CNN binh thuong.
        z_spot_chunks = []
        for start in range(0, B, chunk):
            xb = x[start:start + chunk].to(dev)   # chunk len GPU
            z_b = torch.utils.checkpoint.checkpoint(
                _cnn_chunk_forward, xb, use_reentrant=False)
            z_spot_chunks.append(z_b)
        z_spot = torch.cat(z_spot_chunks, dim=0)   # (N, 128) -- ĐỦ cả section, không bị cắt
    
        # Spatial SGC (Eq. 2)
        # [VÁ BUG] code cũ: A_norm_full[local_indices][:, local_indices] với
        # local_indices là 1 batch NHỎ NGẪU NHIÊN (32 spot) -> chỉ lấy giao điểm
        # 32x32 giữa mấy spot ngẫu nhiên -> KHÔNG thấy láng giềng thật -> SGC vô
        # hiệu (PCC~0). SỬA: SGC chỉ chạy khi feed NGUYÊN section (z_spot.shape[0]
        # == N), lúc đó dùng A_norm_full NGUYÊN (N x N) lan đúng 2-hop trên đồ
        # thị KNN không gian thật. Với batch nhỏ (< N) -> skip SGC (trả CNN
        # embedding) để không crash (vì A_norm_full @ z_spot sai kích thước).
        # Yêu cầu: 1 forward feed đủ N spot (BATCH_SIZE = N cho BRAIN-ST).
        if (self.use_sgc and section_name is not None
                and section_name in self.A_norm_cache
                and z_spot.shape[0] == self.A_norm_cache[section_name].shape[0]):
            # [CHUNK INPUT] x giu tren CPU (patches chunk sau len GPU), nen
            # x.device = cpu. z_spot dang o GPU (dev) -> A_norm phai len dev,
            # khong phai x.device (se bi CPU -> mat khop device voi z_spot).
            A_norm_full = self.A_norm_cache[section_name].to(dev)
            z = z_spot
            if self.sgc_nonlinear:
                # [PHI TUYEN] GCN nhe sgc_hops-hop: lan truyen -> Linear -> ReLU
                # giua cac hop (tru hop cuoi). Co phi tuyen nen hoc duoc bien/duc
                # dac trung khong gian thay vi chi lam tron. Residual giu z_spot.
                for i in range(self.sgc_hops):
                    z = A_norm_full @ z                   # hop i+1 (propagation)
                    if i < self.sgc_hops - 1:
                        z = self.sgc_act(self.sgc_weight(z))   # Linear + ReLU giua hop
                z_hat = z_spot + self.sgc_alpha * self.sgc_act(z)   # residual * alpha
            else:
                # SGC goc (thuan tuyen): A_norm^sgc_hops @ z_spot roi Linear, khong act
                for _ in range(self.sgc_hops):
                    z = A_norm_full @ z
                z_hat = z_spot + self.sgc_alpha * self.sgc_weight(z)
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

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # [CHUNK INPUT] Giu patch tren CPU (batch co the la full section = 2162
        # patches, qua lon de up whole leen GPU). Chi move exp (nhe) len device;
        # forward se tu chunk patch tu CPU len GPU tung phan.
        if isinstance(batch, (list, tuple)):
            patch, pos, exp, sec, lidx = batch
            return (patch, pos.to(device) if pos is not None else pos,
                    exp.to(device), sec, lidx.to(device) if lidx is not None else lidx)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
    
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
