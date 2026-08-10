"""
models/THItoGene_model.py -- Kiến trúc THItoGene gốc (ODConv2d + EfficientCapsNet + ViT + GAT),
GIỮ NGUYÊN forward() y hệt vis_model.py gốc (THItoGene-main), chỉ đổi phần training/loss/
optimizer để khớp đúng convention hiện tại của pipeline (giống STModel/HisToGene):
    - training_step/validation_step: MSE loss (giống STModel/HisToGene, khớp F.mse_loss
      trên pred.view_as(exp) -- đã có sẵn y hệt trong bản gốc, không đổi gì ở đây).
    - configure_optimizers: đổi từ Adam(lr) thuần (bản gốc) sang AdamW(weight_decay=1e-4)
      + CosineAnnealingLR(T_max=max_epochs, eta_min=min_lr) -- ĐÚNG theo xác nhận của bạn,
      khớp STModel/HisToGene/LightHGGEP để so sánh công bằng về training budget.

Kiến trúc nội bộ (ODConv2d, EfficientCapsNet, ViT, MultiHeadGAT) lấy nguyên vẹn từ
ODConv.py / efficient_capsnet.py / transformer.py / GATLayer.py (copy y hệt, không sửa 1
dòng nào -- đã đối chiếu bằng diff) -- chỉ IMPORT để dùng, không viết lại logic bên trong.
"""
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from training_metrics import mean_gene_pearson
from ODConv import ODConv2d
from efficient_capsnet import EfficientCapsNet
from transformer import ViT
from GATLayer import MultiHeadGAT


class THItoGene(pl.LightningModule):
    """forward() giống HỆT vis_model.py gốc (THItoGene-main) -- không đổi bất kỳ phép
    tính nào trong kiến trúc. Chỉ thêm learning_rate/max_epochs/weight_decay/min_lr theo
    đúng chữ ký __init__ của STModel/HisToGene để load_from_checkpoint + configure_optimizers
    dùng chung 1 khuôn mẫu xuyên suốt pipeline."""

    def __init__(self, patch_size=112, n_layers=4, n_genes=785, dim=1024,
                 learning_rate=1e-4, dropout=0.2, n_pos=64, heads=(16, 8), caps=20,
                 route_dim=64, max_epochs=100, weight_decay=1e-4, min_lr=1e-6):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.min_lr = min_lr

        patch_dim = 3 * patch_size * patch_size  # giữ để tương thích chữ ký gốc (không dùng trực tiếp)
        self.route_dim = route_dim
        self.caps = caps

        self.relu = nn.ReLU()

        # ODConv2d: in_planes=3 (RGB thuần, KHÔNG có Sobel 4-kênh như LightHGGEP -- đúng
        # bản gốc THItoGene, giữ nguyên kiến trúc, không lai với input formulation của
        # LightHGGEP).
        self.odconv2d = ODConv2d(in_planes=3, out_planes=16, kernel_size=4, stride=4)

        caps_out = (caps + 2) * route_dim

        self.caps_layer = EfficientCapsNet(rout_capsules=caps, route_dim=route_dim)

        self.x_embed = nn.Embedding(n_pos, route_dim)
        self.y_embed = nn.Embedding(n_pos, route_dim)

        self.vit = ViT(dim=caps_out, depth=n_layers, heads=heads[0], mlp_dim=2 * dim,
                       dropout=dropout, emb_dropout=dropout)

        self.gat = MultiHeadGAT(in_features=caps_out, nhid=1024, out_features=512,
                                heads=heads[1], dropout=dropout, alpha=0.01)

        self.gene_head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, n_genes)
        )

    def forward(self, patches, centers, adj):
        # === Y HỆT vis_model.py gốc, không đổi 1 dòng tính toán nào ===
        B, N, C, H, W = patches.shape
        patches = patches.reshape(B * N, C, H, W)
        patches = self.odconv2d(patches)
        patches = self.relu(patches)

        patches = self.caps_layer(patches)
        patches = patches.reshape(-1, self.caps, self.route_dim)

        centers_x = self.x_embed(centers[:, :, 0]).permute(1, 0, 2)
        centers_y = self.y_embed(centers[:, :, 1]).permute(1, 0, 2)

        x = torch.concat((patches, centers_x, centers_y), dim=1)
        x = x.reshape(1, x.shape[0], -1)

        x = self.vit(x)
        x = x.reshape(x.shape[1], -1)

        x = self.gat(x, adj)
        x = self.gene_head(x)
        return x

    def training_step(self, batch, batch_idx):
        patch, center, exp, adj = batch
        # [MỚI] adj đến từ DataLoader(batch_size=1) nên có thêm chiều batch (1,N,N) --
        # forward()/self.gat(x, adj) cần đúng (N,N) 2D (x đã reshape về (N,D) bên trong
        # forward, KHÔNG giữ chiều batch cho x). patch/center vẫn giữ nguyên chiều batch
        # (1,N,...) vì forward() cần B,N,C,H,W / centers[:, :, 0] đúng 3 chiều.
        adj = adj.squeeze(0)
        pred = self(patch, center, adj)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mse', loss, on_epoch=True, sync_dist=True)
        self.log('train_pcc', mean_gene_pearson(pred.view_as(exp), exp), on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp, adj = batch
        adj = adj.squeeze(0)
        pred = self(patch, center, adj)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('valid_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_mse', loss, on_epoch=True, sync_dist=True)
        self.log('val_pcc', mean_gene_pearson(pred.view_as(exp), exp), on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        patch, center, exp, adj = batch
        adj = adj.squeeze(0)
        pred = self(patch, center, adj)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('test_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        # [ĐÃ ĐỔI theo xác nhận của bạn] Bản gốc: torch.optim.Adam(self.parameters(), lr=lr)
        # thuần, không weight_decay, không scheduler. Đổi sang khớp đúng STModel/HisToGene/
        # LightHGGEP để so sánh công bằng về training budget (cùng optimizer/scheduler family).
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.min_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
