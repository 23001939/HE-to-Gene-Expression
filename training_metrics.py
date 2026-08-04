"""Small metrics used only for epoch progress logging."""

import torch


def mean_gene_pearson(prediction: torch.Tensor, target: torch.Tensor,
                      eps: float = 1e-8) -> torch.Tensor:
    """Mean Pearson correlation across genes; preceding axes are spots."""
    prediction = prediction.detach().reshape(-1, prediction.shape[-1])
    target = target.detach().reshape(-1, target.shape[-1])
    prediction = prediction - prediction.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    denom = prediction.square().sum(dim=0).sqrt() * target.square().sum(dim=0).sqrt()
    corr = (prediction * target).sum(dim=0) / denom.clamp_min(eps)
    return torch.nanmean(corr)
