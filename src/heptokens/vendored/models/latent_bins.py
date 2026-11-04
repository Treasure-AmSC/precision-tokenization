"""Latent binning: an autoencoder bottleneck whose tokens are per-dimension bins."""

import torch
import torch.nn as nn


class LatentBins(nn.Module):
    """Uniform bins on [-bin_range, bin_range] for each latent dimension, one token per dimension.

    mu = bin_range * tanh(h / bin_range) keeps the latent inside the bins.
    Training: z = mu + noise_std_bins * bin width * N(0, 1).
    Evaluation: z = the centre of mu's bin, which is what the tokens encode.

    Args:
        dim: Latent dimensions (tokens per element).
        num_bins: Bins per dimension (values per token).
        bin_range: Half-width of the binned interval.
        noise_std_bins: Training noise std, in bin widths.
    """

    def __init__(self, dim: int, num_bins: int, bin_range: float, noise_std_bins: float):
        super().__init__()
        assert dim > 0 and num_bins >= 2 and bin_range > 0 and noise_std_bins >= 0
        self.dim = dim
        self.num_bins = num_bins
        self.bin_range = float(bin_range)
        self.bin_width = 2 * self.bin_range / num_bins
        self.noise_std = noise_std_bins * self.bin_width

    def bins(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h [..., dim] -> mu, bin indices [..., dim]."""
        mu = self.bin_range * torch.tanh(h / self.bin_range)
        indices = torch.floor((mu.detach() + self.bin_range) / self.bin_width).long()
        return mu, indices.clamp(0, self.num_bins - 1)

    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Bin indices [..., dim] -> bin centres [..., dim]."""
        return -self.bin_range + (indices + 0.5) * self.bin_width

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h [B, T, dim] -> z [B, T, dim], indices [dim, B, T], zero loss [dim] (as ResidualVQ)."""
        assert h.shape[-1] == self.dim, f"{h.shape[-1]} latent dimensions, expected {self.dim}"
        mu, indices = self.bins(h)
        if self.training:
            z = mu + self.noise_std * torch.randn_like(mu)
        else:
            z = self.get_output_from_indices(indices).to(mu.dtype)
        return z, indices.permute(2, 0, 1), h.new_zeros(self.dim)
