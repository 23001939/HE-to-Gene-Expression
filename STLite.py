import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA

# =============================================================================
# 0. CẤU HÌNH (CONFIGURATION)
# =============================================================================
@dataclass
class STLiteConfig:
    micro_patch_size: int = 224         # Kích thước ảnh cắt quanh mỗi spot
    in_channels: int = 3
    token_dim: int = 64                 # Chiều dữ liệu của Trạng thái tế bào (Latent)
    encoder_hidden: int = 32
    
    num_prototypes: int = 16            # Số lượng Nguyên mẫu mô (Module B)
    
    gcn_k_neighbors: int = 6            # Số hàng xóm trong đồ thị k-NN (Module C)
    gcn_num_layers: int = 2
    gcn_distance_sigma: float = 500.0   # phù hợp với pixel coordinates (~khoảng cách giữa các spot)
    
    num_pathways: int = 32              # Số lượng Pathway (Trưởng phòng)
    total_genes: int = 785              # Tổng số gen mục tiêu cần dự đoán
    
    pathway_loss_weight: float = 0.3    # Trọng số cho L_pathway

# =============================================================================
# 1. MODULE A: MÃ HÓA HÌNH THÁI (Cell-aware Token Encoder)
# =============================================================================
class CellAwareTokenEncoder(nn.Module):
    def __init__(self, cfg: STLiteConfig):
        super().__init__()
        # Depthwise -> Pointwise -> Pooling -> Linear
        self.depthwise = nn.Conv2d(
            cfg.in_channels, cfg.in_channels, kernel_size=3, 
            padding=1, groups=cfg.in_channels, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(cfg.in_channels)
        self.pointwise = nn.Conv2d(cfg.in_channels, cfg.encoder_hidden, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(cfg.encoder_hidden)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(cfg.encoder_hidden, cfg.token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, N_spots, Channels, H, W)
        B, N, C, h, w = x.shape
        x_reshaped = x.reshape(B * N, C, h, w)
        
        out = self.act(self.bn_dw(self.depthwise(x_reshaped)))
        out = self.act(self.bn_pw(self.pointwise(out)))
        out = self.pool(out).flatten(1)
        z = self.proj(out)
        
        return z.reshape(B, N, -1) # Output: (B, N, token_dim)

# =============================================================================
# 2. MODULE B: HỌC NGUYÊN MẪU MÔ (Tissue Prototype Learning)
# =============================================================================
class TissuePrototypeLearning(nn.Module):
    def __init__(self, cfg: STLiteConfig):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(cfg.num_prototypes, cfg.token_dim) * 0.02)
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor):
        # Tính độ tương đồng và gán mềm (soft assignment)
        temp = self.log_temperature.exp().clamp(min=1e-3, max=100.0)
        logits = torch.einsum("bnd,kd->bnk", z, self.prototypes) / temp
        assignment = torch.softmax(logits, dim=-1)
        
        # Tạo vector đặc trưng mới
        h = torch.einsum("bnk,kd->bnd", assignment, self.prototypes)
        return h

# =============================================================================
# 3. MODULE C: GIAO TIẾP KHÔNG GIAN (Spatial GCN Mixer)
# =============================================================================
class SpatialGraphConvLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin_self = nn.Linear(dim, dim)
        self.lin_neigh = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor, neighbor_idx: torch.Tensor, neighbor_weight: torch.Tensor):
        B, N, D = h.shape
        k = neighbor_idx.shape[-1]
        
        # Gom đặc trưng từ các hàng xóm
        batch_idx = torch.arange(B, device=h.device).view(B, 1, 1).expand(B, N, k)
        neighbor_feats = h[batch_idx, neighbor_idx]
        
        # Tính trung bình có trọng số theo khoảng cách vật lý
        agg = (neighbor_feats * neighbor_weight.unsqueeze(-1)).sum(dim=2)
        
        out = self.lin_self(h) + self.lin_neigh(agg)
        return self.act(self.norm(out))

class SpatialGCNMixer(nn.Module):
    def __init__(self, cfg: STLiteConfig):
        super().__init__()
        self.k = cfg.gcn_k_neighbors
        self.sigma = cfg.gcn_distance_sigma
        self.layers = nn.ModuleList([
            SpatialGraphConvLayer(cfg.token_dim) for _ in range(cfg.gcn_num_layers)
        ])

    def forward(self, h: torch.Tensor, coords: torch.Tensor):
        B, N, _ = coords.shape
        # Tính ma trận khoảng cách và tìm k hàng xóm
        dist = torch.cdist(coords, coords, p=2)
        dist.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        
        neg_dist, neighbor_idx = torch.topk(-dist, k=self.k, dim=-1)
        weight_logits = -((-neg_dist) ** 2) / (self.sigma ** 2)
        neighbor_weight = torch.softmax(weight_logits, dim=-1)
        
        out = h
        for layer in self.layers:
            out = layer(out, neighbor_idx, neighbor_weight) + out # Residual connection
            
        return out

# =============================================================================
# 4. MODULE D: GIẢI MÃ GEN PHÂN CẤP (Hierarchical Gene Decoder)
# =============================================================================
class HierarchicalGeneDecoder(nn.Module):
    def __init__(self, cfg: STLiteConfig):
        super().__init__()
        # Bước 1: Nén từ 64 chiều (Đặc trưng tế bào) xuống 32 chiều (Pathway)
        self.to_pathway = nn.Sequential(
            nn.Linear(cfg.token_dim, cfg.num_pathways),
            nn.GELU(),
        )
        # Bước 2: Bung từ 32 chiều (Pathway) ra 785 chiều (Tổng số gen)
        self.pathway_to_genes = nn.Linear(cfg.num_pathways, cfg.total_genes)

    def forward(self, y: torch.Tensor):
        pathway_repr = self.to_pathway(y)               # (B, N, 32)
        gene_expr = self.pathway_to_genes(pathway_repr) # (B, N, 785)
        return gene_expr, pathway_repr

# =============================================================================
# 5. LẮP RÁP TOÀN BỘ HỆ THỐNG VÀ TÍNH LOSS
# =============================================================================
class STLite(nn.Module):
    def __init__(self, cfg: STLiteConfig):
        super().__init__()
        self.cfg = cfg
        self.module_a = CellAwareTokenEncoder(cfg)
        self.module_b = TissuePrototypeLearning(cfg)
        self.module_c = SpatialGCNMixer(cfg)
        self.module_d = HierarchicalGeneDecoder(cfg)

    def forward(self, images: torch.Tensor, coords: torch.Tensor):
        # Chạy dữ liệu qua dây chuyền 4 trạm
        z = self.module_a(images)
        h = self.module_b(z)
        y = self.module_c(h, coords)
        gene_expr, pathway_repr = self.module_d(y)

        return {
            "gene_expression": gene_expr,
            "pathway_representation": pathway_repr
        }

    def compute_loss(self, outputs: dict, gene_target: torch.Tensor, pathway_target: torch.Tensor):
        # L_gene: So sánh 785 gen dự đoán với 785 gen thật
        l_gene = F.mse_loss(outputs["gene_expression"], gene_target)
        
        # L_pathway: Ép biểu diễn ẩn (32 chiều) phải khớp với Pathway chuẩn
        l_pathway = F.mse_loss(outputs["pathway_representation"], pathway_target)
        
        # Tổng hợp Loss
        total_loss = l_gene + self.cfg.pathway_loss_weight * l_pathway
        
        return total_loss, l_gene, l_pathway
