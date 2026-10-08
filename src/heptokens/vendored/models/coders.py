# Simple encoders and decoders for use in autoencoder models.
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class BaseEncoder(nn.Module, ABC):
    """Abstract interface for VQ-VAE encoders.

    All encoders must accept a batch dict and return latent embeddings.
    The batch dict must contain at minimum the primary input tensor and,
    for variable-length data, a boolean validity mask.

    Contract:
        forward(batch: dict) -> Tensor of shape [batch, n_tokens, codebook_dim]
            where invalid (masked) positions are zeroed out.
    """

    @abstractmethod
    def forward(self, batch: dict) -> torch.Tensor:
        """Encode a batch to latent embeddings."""
        ...


class BaseDecoder(nn.Module, ABC):
    """Abstract interface for VQ-VAE decoders.

    All decoders must accept quantized embeddings + the original batch and
    return reconstructed data, as well as expose a compute_loss method.

    Contract:
        forward(z_q, batch) -> reconstructed Tensor
        compute_loss(z_q, batch) -> scalar loss Tensor
    """

    @abstractmethod
    def forward(self, z_q: torch.Tensor, batch: dict) -> torch.Tensor:
        """Decode quantized embeddings to data space."""
        ...

    @abstractmethod
    def compute_loss(self, z_q: torch.Tensor, batch: dict) -> torch.Tensor:
        """Compute reconstruction loss between decoded output and batch targets."""
        ...


class CoderModel(nn.Module):
    """
    Base class for Encoder and Decoder models.
    Default is a simple MLP with LayerNorm and ReLU activations.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int] = [128, 256, 512]):
        super(CoderModel, self).__init__()
        # Build model
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,  # noqa: ARG002 -- unused; per-patch MLP needs no positional info
    ) -> torch.Tensor:
        """Forward pass through the model."""
        if mask is not None:
            # Select valid inputs
            x = x[mask]
        # Encode valid inputs
        encoding = self.model(x)
        if mask is not None:
            # Create full output tensor with zeros for invalid positions
            full_encoding = torch.zeros(
                (*mask.shape, encoding.shape[1]), device=encoding.device, dtype=encoding.dtype
            )
            full_encoding[mask] = encoding
            return full_encoding
        else:
            return encoding


class Coder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model: nn.Module = CoderModel,
        input_key: str = "csts",
        mask_key: str = "mask",
        positions_key: Optional[str] = None,
        data_sample=None,
    ):
        super(Coder, self).__init__()
        # Build coder (encoder/decoder). data_sample is accepted for API
        # compatibility with callers that pass it (e.g. LitVqVae), but is
        # unused by the base Encoder/Decoder/CoderModel path.
        self.coder = model(input_dim=input_dim, output_dim=output_dim)
        self.input_key = input_key
        self.mask_key = mask_key
        # Optional key for a per-token spatial coordinates tensor (e.g.
        # [eta, cos_phi, sin_phi]), forwarded to the inner model as
        # `positions=...` for architectures that support positional encoding
        # (e.g. Transformer). None by default -- no behavior change unless a
        # config explicitly sets this.
        self.positions_key = positions_key


class Encoder(Coder, BaseEncoder):
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode input data to latent embeddings.

        Args:
            batch: Dictionary containing the primary input tensor (key given by
                ``self.input_key``, default ``'csts'``) of shape
                [batch_size, n_tokens, d_vector] and an optional boolean validity
                mask (key given by ``self.mask_key``, default ``'mask'``) of
                shape [batch_size, n_tokens].

        Returns:
            Latent embeddings of shape [batch_size, n_tokens, codebook_dim]
            with invalid (masked) positions zeroed out.
        """
        mask = batch.get(self.mask_key)
        if self.positions_key:
            z_e = self.coder(batch[self.input_key], mask, positions=batch.get(self.positions_key))
        else:
            z_e = self.coder(batch[self.input_key], mask)
        return z_e


class Decoder(Coder, BaseDecoder):
    def forward(self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode quantized embeddings.

        Args:
            z_q: Quantized embeddings [batch_size, n_tokens, codebook_dim]
            batch: Original batch dict; uses the mask at ``self.mask_key``
                to process only valid positions.

        Returns:
            Reconstructed tensor of shape [batch_size, n_tokens, d_vector]
            with invalid positions zeroed out.
        """
        mask = batch.get(self.mask_key)
        if self.positions_key:
            return self.coder(z_q, mask, positions=batch.get(self.positions_key))
        return self.coder(z_q, mask)

    def compute_loss(self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute reconstruction loss given quantized embeddings and original batch.
        Default is L1 loss on valid (unmasked) positions.

        Args:
            z_q: Quantized embeddings [batch_size, n_tokens, codebook_dim]
            batch: Original batch dict with the primary input at ``self.input_key``
                and an optional mask at ``self.mask_key``.

        Returns:
            Reconstruction loss tensor (scalar).
        """
        targets = batch[self.input_key]
        mask = batch.get(self.mask_key)

        reconstructed = self.forward(z_q, batch)

        if mask is not None:
            recon_loss = nn.functional.l1_loss(reconstructed[mask], targets[mask])
        else:
            recon_loss = nn.functional.l1_loss(reconstructed, targets)
        return recon_loss


class CellDecoder(Decoder):
    """Decoder for calorimeter cell patch data with pixel-level sparsity handling.

    Expects the input key to hold a tensor of shape ``[B, T, 2*n_pixels]`` where
    the first half is the energy (energy^p, p=``energy_power``) and the second
    half is a binary pixel-level occupancy indicator — both computed in
    ``CellDataset.__getitem__``.

    Loss:
      - Binary cross-entropy on the indicator for all pixels in non-empty patches
      - MSE (or L1, ``energy_loss_type="l1"``) on energy for pixels where the
        indicator is 1 (occupied pixels only),
        optionally weighted per pixel by ``w_i = E_i**energy_loss_alpha`` where
        ``E_i`` is the linear truth energy recovered from the compressed target
        (``E = y**(1/energy_power)``). ``energy_loss_alpha=0`` reproduces the
        original unweighted MSE exactly.
      - Optionally (``energy_sum_loss_weight>0``) a per-patch linear-space
        energy-sum term ``((sum_i ohat_i*Ehat_i) - (sum_i o_i*E_i))**2``.

    After each ``compute_loss`` call, ``self.last_metrics`` holds the scalar
    breakdown (weighted/unweighted energy MSE, occupancy BCE, energy-sum term)
    plus the configured weights, so the Lightning module can log them.

    Args:
        energy_loss_alpha: Exponent alpha for the per-pixel energy-MSE weight
            ``w_i = E_i**alpha`` (E_i = linear truth energy). 0 disables
            weighting (bitwise-identical to the original loss).
        energy_sum_loss_weight: Weight lambda for the optional per-patch
            linear-space energy-sum term. 0 disables it.
        energy_power: Preprocessing energy compression exponent p (targets are
            stored as ``E**p``); the inverse ``1/p`` recovers linear energy.
            Must match ``preprocess_hep4m_cells.py``'s ``ENERGY_POWER``.
        energy_loss_type: Per-pixel energy loss, ``"mse"`` (default, original
            loss) or ``"l1"``. The alpha weighting applies to either.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model: nn.Module = CoderModel,
        input_key: str = "csts",
        mask_key: str = "mask",
        positions_key: Optional[str] = None,
        data_sample=None,
        energy_loss_alpha: float = 0.0,
        energy_sum_loss_weight: float = 0.0,
        energy_power: float = 0.3,
        energy_loss_type: str = "mse",
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            model=model,
            input_key=input_key,
            mask_key=mask_key,
            positions_key=positions_key,
            data_sample=data_sample,
        )
        self.energy_loss_alpha = float(energy_loss_alpha)
        self.energy_sum_loss_weight = float(energy_sum_loss_weight)
        self.energy_power = float(energy_power)
        if energy_loss_type not in ("mse", "l1"):
            raise ValueError(f"energy_loss_type must be 'mse' or 'l1', got {energy_loss_type!r}")
        self.energy_loss_type = energy_loss_type
        self.last_metrics: Dict[str, float] = {}

    def compute_loss(self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        targets = batch[self.input_key]  # [B, T, 2*n_pixels]
        mask = batch.get(self.mask_key)  # [B, T] bool or None
        recon = self.forward(z_q, batch)  # [B, T, 2*n_pixels]

        n = targets.shape[-1] // 2
        energy_recon = recon[..., :n]
        ind_logits = recon[..., n:]
        energy_true = targets[..., :n]
        ind_true = targets[..., n:]  # binary float {0., 1.}

        if mask is not None:
            loss_ind = nn.functional.binary_cross_entropy_with_logits(
                ind_logits[mask], ind_true[mask]
            )
            pixel_mask = mask.unsqueeze(-1) & (ind_true > 0.5)
        else:
            loss_ind = nn.functional.binary_cross_entropy_with_logits(ind_logits, ind_true)
            pixel_mask = ind_true > 0.5

        if pixel_mask.any():
            energy_recon_sel = energy_recon[pixel_mask]
            energy_true_sel = energy_true[pixel_mask]
            # Unweighted loss — for "mse" identical to the original loss (also
            # the alpha=0 return value, so alpha=0 stays bitwise-identical).
            if self.energy_loss_type == "l1":
                loss_energy_unweighted = nn.functional.l1_loss(energy_recon_sel, energy_true_sel)
            else:
                loss_energy_unweighted = nn.functional.mse_loss(energy_recon_sel, energy_true_sel)
            if self.energy_loss_alpha != 0.0:
                # Targets are raw E**p — CellDataset applies no standardization
                # (unlike tracks/topos), so E = y**(1/p) exactly and y>=0.
                # Compute in float32: pow(1/p) can overflow fp16 under AMP.
                energy_true_f = energy_true_sel.float()
                energy_recon_f = energy_recon_sel.float()
                energy_true_lin = energy_true_f.clamp(min=0.0).pow(1.0 / self.energy_power)
                weight = energy_true_lin.pow(self.energy_loss_alpha)
                diff = energy_recon_f - energy_true_f
                per_pixel = diff.abs() if self.energy_loss_type == "l1" else diff**2
                # Normalize by sum(w) (not pixel count) so the loss scale is
                # comparable across alpha.
                weight_sum = weight.sum().clamp_min(torch.finfo(weight.dtype).eps)
                loss_energy = (weight * per_pixel).sum() / weight_sum
            else:
                loss_energy = loss_energy_unweighted
        else:
            loss_energy = recon.sum() * 0.0  # differentiable zero
            loss_energy_unweighted = loss_energy

        loss = loss_energy + loss_ind

        if self.energy_sum_loss_weight != 0.0:
            loss_energy_sum = self._energy_sum_loss(
                energy_recon, ind_logits, energy_true, ind_true, mask
            )
            loss = loss + self.energy_sum_loss_weight * loss_energy_sum
        else:
            loss_energy_sum = recon.sum() * 0.0

        self.last_metrics = {
            f"energy_{self.energy_loss_type}": loss_energy_unweighted.detach(),
            f"energy_{self.energy_loss_type}_weighted": loss_energy.detach(),
            "occupancy_bce": loss_ind.detach(),
            "energy_sum_mse": loss_energy_sum.detach(),
            "energy_loss_alpha": self.energy_loss_alpha,
            "energy_sum_loss_weight": self.energy_sum_loss_weight,
        }
        return loss

    def _energy_sum_loss(
        self,
        energy_recon: torch.Tensor,
        ind_logits: torch.Tensor,
        energy_true: torch.Tensor,
        ind_true: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Per-patch *relative* linear-space energy-sum MSE, averaged over
        patches with real energy.

        ``sum_reco = sum_i sigmoid(ind_logits_i) * Ehat_i`` and
        ``sum_true = sum_i o_i * E_i`` per patch, both in linear energy space
        (``E = y**(1/energy_power)``). The residual is taken relative to
        ``sum_true`` (``(sum_reco - sum_true)/sum_true``) so it matches the
        ratio-based eval metric and shares the y-space terms' scale (an
        absolute GeV^2 term would be dominated by the hottest patches and sit
        ~1e4x higher); returns the mean squared relative error over patches
        with ``sum_true > 0`` (empty patches have no meaningful ratio).

        Computed in float32: pow(1/p) ~ pow(3.33) and the squared sums can
        overflow fp16 under AMP.
        """
        inv_p = 1.0 / self.energy_power
        energy_recon_lin = energy_recon.float().clamp(min=0.0).pow(inv_p)
        energy_true_lin = energy_true.float().clamp(min=0.0).pow(inv_p)
        sum_reco = (torch.sigmoid(ind_logits.float()) * energy_recon_lin).sum(dim=-1)  # [B, T]
        sum_true = (ind_true.float() * energy_true_lin).sum(dim=-1)  # [B, T]
        valid = sum_true > 0
        if mask is not None:
            valid = valid & mask
        if not valid.any():
            return energy_recon.sum().float() * 0.0
        rel = (sum_reco[valid] - sum_true[valid]) / sum_true[valid].clamp_min(
            torch.finfo(sum_true.dtype).eps
        )
        return (rel**2).mean()


class DiffusionDecoder(BaseDecoder):
    """Decoder using diffusion model for reconstruction loss."""

    def __init__(
        self,
        input_dim: int,  # codebook_dim
        output_dim: int,  # d_vector (data dimension)
        model: nn.Module = CoderModel,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        input_key: str = "csts",
        mask_key: str = "mask",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.output_dim = output_dim
        self.latent_dim = input_dim

        # Model takes: [noisy_data (output_dim) + latent (latent_dim) + time_embed] -> output_dim
        self.time_embed_dim = 32
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # Input is: noisy_csts + z_q + time_embedding
        coder_input_dim = output_dim + input_dim + self.time_embed_dim
        self.coder = model(input_dim=coder_input_dim, output_dim=output_dim)

        # Initialize beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, num_timesteps)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        # Precompute alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.input_key = input_key
        self.mask_key = mask_key

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine annealing schedule."""
        steps = torch.arange(timesteps + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def forward(
        self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor], t: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for inference - performs DDPM sampling.

        Args:
            z_q: Quantized embeddings [batch, n_csts, codebook_dim]
            batch: Batch dict (uses 'mask' if present)
            t: Ignored (full denoising always performed)

        Returns:
            Generated samples [batch, n_csts, d_vector]
        """
        mask = batch.get(self.mask_key)
        device = z_q.device
        batch_size, n_csts = z_q.shape[0], z_q.shape[1]

        # Start from pure noise
        x = torch.randn(batch_size, n_csts, self.output_dim, device=device)

        # Denoise step by step from T-1 to 0
        for t_idx in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)

            # Time embedding
            t_normalized = (t_batch.float() / self.num_timesteps).view(-1, 1)
            t_embed = self.time_embed(t_normalized)
            t_embed = t_embed.unsqueeze(1).expand(-1, n_csts, -1)

            # Predict noise
            model_input = torch.cat([x, z_q, t_embed], dim=-1)
            predicted_noise = self.coder(model_input, mask)

            # Compute denoising step
            alpha_cumprod = self.alphas_cumprod[t_idx]
            beta = self.betas[t_idx]

            # DDPM sampling formula
            if t_idx > 0:
                noise = torch.randn_like(x)
                alpha_cumprod_prev = self.alphas_cumprod[t_idx - 1]
            else:
                noise = torch.zeros_like(x)
                alpha_cumprod_prev = torch.tensor(1.0, device=device)

            # Posterior mean
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(
                alpha_cumprod
            )

            # Posterior variance
            posterior_variance = beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)

            # Update x
            x = (
                torch.sqrt(alpha_cumprod_prev) * pred_x0
                + torch.sqrt(1 - alpha_cumprod_prev - posterior_variance) * predicted_noise
                + torch.sqrt(posterior_variance) * noise
            )

            # Mask invalid positions
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        return x

    def compute_loss(self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute diffusion loss (noise prediction loss).

        Args:
            z_q: Quantized embeddings [batch, n_csts, codebook_dim]
            batch: Original batch dict with 'csts' [batch, n_csts, d_vector] and 'mask'

        Returns:
            Reconstruction loss tensor
        """
        csts = batch[self.input_key]  # [batch, n_tokens, d_vector]
        mask = batch.get(self.mask_key)  # [batch, n_tokens] or None

        # Sample random timestep per batch element
        t = torch.randint(0, self.num_timesteps, (csts.shape[0],), device=csts.device)

        # Sample noise
        noise = torch.randn_like(csts)  # [batch, n_csts, d_vector]

        # Add noise to clean data
        sqrt_alphas = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        noisy_csts = sqrt_alphas * csts + sqrt_one_minus_alphas * noise

        # Time embedding
        t_normalized = (t.float() / self.num_timesteps).view(-1, 1)  # [batch, 1]
        t_embed = self.time_embed(t_normalized)  # [batch, time_embed_dim]
        t_embed = t_embed.unsqueeze(1).expand(
            -1, csts.shape[1], -1
        )  # [batch, n_csts, time_embed_dim]

        # Concatenate inputs: [noisy_data, latent, time]
        model_input = torch.cat([noisy_csts, z_q, t_embed], dim=-1)  # [batch, n_csts, input_dim]

        # Predict noise
        predicted_noise = self.coder(model_input, mask)  # [batch, n_csts, d_vector]

        # L2 loss on predicted noise (only on valid positions if mask provided)
        if mask is not None:
            valid_noise = noise[mask]
            valid_predicted = predicted_noise[mask]
        else:
            valid_noise = noise.reshape(-1, noise.shape[-1])
            valid_predicted = predicted_noise.reshape(-1, predicted_noise.shape[-1])
        recon_loss = nn.functional.mse_loss(valid_predicted, valid_noise)

        return recon_loss
