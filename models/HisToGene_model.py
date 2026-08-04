import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from training_metrics import mean_gene_pearson

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        # self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x): #change
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        # attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.,
                 gradient_checkpointing=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gradient_checkpointing = gradient_checkpointing
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            def layer_forward(hidden, attn=attn, ff=ff):
                hidden = attn(hidden) + hidden
                return ff(hidden) + hidden
            # Recompute each transformer block during backward rather than
            # retaining all activations.  This is essential for slide-level
            # attention on the largest HER2ST sections on a T4.
            if self.training and self.gradient_checkpointing and x.requires_grad:
                x = checkpoint(layer_forward, x, use_reentrant=False)
            else:
                x = layer_forward(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.,
                 gradient_checkpointing=True):
        super().__init__()
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout,
                                       gradient_checkpointing=gradient_checkpointing)
        self.to_latent = nn.Identity()

    def forward(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)
        return x
    
# adaptation of histogene
class HisToGene(pl.LightningModule):
    def __init__(self, patch_size=112, n_layers=4, n_genes=1000, dim=1024,
                 learning_rate=1e-4, dropout=0.1, n_pos=64,
                 max_epochs=100, weight_decay=1e-4, min_lr=1e-6,
                 gradient_checkpointing=True):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        patch_dim = 3*patch_size*patch_size
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.x_embed = nn.Embedding(n_pos,dim)
        self.y_embed = nn.Embedding(n_pos,dim)
        self.vit = ViT(dim=dim, depth=n_layers, heads=16, mlp_dim=2*dim,
                       dropout=dropout, emb_dropout=dropout,
                       gradient_checkpointing=gradient_checkpointing)

        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes)
        )

    def forward(self, patches, centers):
        patches = self.patch_embedding(patches)
        centers_x = self.x_embed(centers[:,:,0])
        centers_y = self.y_embed(centers[:,:,1])
        x = patches + centers_x + centers_y
        h = self.vit(x)
        x = self.gene_head(h)
        return x

    def training_step(self, batch, batch_idx):        
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('train_loss', loss)
        self.log('train_mse', loss, on_epoch=True, sync_dist=True)
        self.log('train_pcc', mean_gene_pearson(pred, exp), on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('valid_loss', loss, on_epoch=True, sync_dist=True)
        self.log('val_mse', loss, on_epoch=True, sync_dist=True)
        self.log('val_pcc', mean_gene_pearson(pred, exp), on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('test_loss', loss)

    def configure_optimizers(self):
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
