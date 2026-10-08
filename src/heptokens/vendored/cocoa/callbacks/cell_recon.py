"""Reconstruction monitor for HEP4M calorimeter cell patch VQ-VAE."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from heptokens.vendored.callbacks.recon import BaseReconstructionMonitor

log = logging.getLogger(__name__)

# ── Patch geometry (mirrors preprocess_hep4m_cells.py / CellImageHelper) ─────
_POW2_SCALE = [3, 3, 3, 3, 3, 2]
_PATCH_SIZES = [2**s for s in _POW2_SCALE]  # [8, 8, 8, 8, 8, 4] (downsample-invariant)
_NATIVE_BINS_TO_KEEP = [64, 64, 32, 16, 16, 8]  # ROI bins per layer at downsample=1
_N_LAYERS = 6


def _layer_geometry(downsample: int) -> tuple[list[int], list[int], list[int]]:
    """Per-downsample (bins_to_keep, patches_per_layer, layer_starts).

    Mirrors CellImageHelper.downsample: dividing bins_to_keep by `downsample`
    (not just grans) keeps the physical ROI window fixed while shrinking the
    patch/pixel count per layer — see CellImageHelper's docstring. Must match
    whatever downsample the data was preprocessed with, or patch counts won't
    line up with the actual batch tensors.
    """
    bins_to_keep = [b // downsample for b in _NATIVE_BINS_TO_KEEP]
    patches_per_layer = [(b // p) ** 2 for b, p in zip(bins_to_keep, _PATCH_SIZES)]
    layer_starts = [sum(patches_per_layer[:li]) for li in range(_N_LAYERS)]
    return bins_to_keep, patches_per_layer, layer_starts


def _patches_to_image(patches: torch.Tensor, layer_idx: int, bins_to_keep: list[int]) -> np.ndarray:
    """Reconstruct a [H, W] 2D energy image from the flat patch tensor for one layer.

    Args:
        patches: float Tensor [patches_per_layer[layer_idx], 64] — pixel energies,
            zero-padded to 64 for layer 5 (only first 16 pixels are real).
        layer_idx: 0–5 calorimeter layer index.
        bins_to_keep: this instance's (downsample-aware) bins_to_keep list.

    Returns:
        NumPy array [H, W] where H = W = bins_to_keep[layer_idx].
    """
    ps = _PATCH_SIZES[layer_idx]
    n1d = bins_to_keep[layer_idx] // ps  # patches per side
    real_pixels = ps * ps
    p = patches[:, :real_pixels]  # trim zero-padding (layer 5: 16→16)
    img = p.reshape(n1d, n1d, ps, ps)
    img = img.permute(0, 2, 1, 3)  # [nH, ps, nW, ps]
    return img.reshape(n1d * ps, n1d * ps).cpu().numpy()


def _add_patch_grid(ax, bins: int, patch_size: int) -> None:
    """Overlay white patch-boundary grid lines on an imshow axis."""
    boundaries = np.arange(patch_size, bins, patch_size) - 0.5
    for b in boundaries:
        ax.axvline(b, color="white", linewidth=0.5, alpha=0.7)
        ax.axhline(b, color="white", linewidth=0.5, alpha=0.7)


class CellReconstructionMonitor(BaseReconstructionMonitor):
    """Reconstruction monitor for calorimeter cell image patches.

    Extends BaseReconstructionMonitor with three epoch-level wandb figures:

    * ``val/calo_images``       — N events × 6 layers, original | reconstructed
    * ``val/layer_energy_hist`` — per-layer energy histograms (orig vs recon)
    * ``val/patch_occupancy``   — per-layer fraction of non-zero patches (orig vs recon)

    Scalars ``val/scaled_mae`` and ``val/scaled_mse`` are inherited.

    Args:
        n_events: Number of events to show in the image grid.
        log_every_n_epochs: Run visualisations every N validation epochs.
        max_batches: Validation batches to accumulate per epoch.
        downsample: Must match the pixel downsample factor the data was
            preprocessed with (CellImageHelper.downsample / --downsample in
            preprocess_hep4m_cells.py), so patch counts/layer boundaries line
            up with the actual batch tensors. Default 1 (native cell size).
    """

    def __init__(
        self,
        n_events: int = 4,
        log_every_n_epochs: int = 1,
        max_batches: int = 4,
        downsample: int = 1,
    ):
        super().__init__(
            input_key="energy",
            mask_key="mask",
            log_every_n_epochs=log_every_n_epochs,
            max_batches=max_batches,
        )
        self.n_events = n_events
        self.downsample = downsample
        self.bins_to_keep, self.patches_per_layer, self.layer_starts = _layer_geometry(downsample)
        self._originals: list[torch.Tensor] = []  # [n_patches, 2*n_pixels] per event
        self._recons: list[torch.Tensor] = []

    # ── batch-level hook ─────────────────────────────────────────────────────

    def compute_and_log_metrics(self, trainer, pl_module, batch, batch_idx):
        with torch.no_grad():
            z_q = pl_module.encode(batch)[0]
            recon = pl_module.decode(z_q, batch)  # [B, T, 2*n_pixels]

        targets = batch[self.input_key]  # [B, T, 2*n_pixels]
        patch_mask = batch.get(self.mask_key)  # [B, T] or None
        n = targets.shape[-1] // 2

        energy_true = targets[..., :n]
        ind_true = targets[..., n:]
        energy_recon = recon[..., :n]
        ind_logits = recon[..., n:]

        # Pixel mask: patch-level AND ground-truth occupied
        pixel_mask = (
            (patch_mask.unsqueeze(-1) & (ind_true > 0.5))
            if patch_mask is not None
            else (ind_true > 0.5)
        )

        if pixel_mask.any():
            pl_module.log(
                "val/energy_mae",
                torch.mean(torch.abs(energy_recon[pixel_mask] - energy_true[pixel_mask])),
                prog_bar=False,
            )
            pl_module.log(
                "val/energy_mse",
                torch.mean((energy_recon[pixel_mask] - energy_true[pixel_mask]) ** 2),
                prog_bar=False,
            )

        if patch_mask is not None:
            ind_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                ind_logits[patch_mask], ind_true[patch_mask]
            )
        else:
            ind_bce = torch.nn.functional.binary_cross_entropy_with_logits(ind_logits, ind_true)
        pl_module.log("val/indicator_bce", ind_bce, prog_bar=False)

        # Accumulate for epoch-level image figures
        still_needed = self.n_events - len(self._originals)
        if still_needed <= 0:
            return
        recon_cpu = recon.cpu()
        targets_cpu = targets.cpu()
        take = min(still_needed, targets_cpu.shape[0])
        self._originals.extend(targets_cpu[:take].unbind(0))
        self._recons.extend(recon_cpu[:take].unbind(0))

    # ── epoch-level hook ─────────────────────────────────────────────────────

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._originals:
            return
        if trainer.logger is None or not hasattr(trainer.logger, "experiment"):
            self._originals.clear()
            self._recons.clear()
            return

        orig_list = self._originals[: self.n_events]
        recon_list = self._recons[: self.n_events]
        self._originals.clear()
        self._recons.clear()

        # ── Figure 1: 2D calorimeter images ──────────────────────────────────
        n_shown = len(orig_list)
        fig1, axes1 = plt.subplots(
            n_shown * 2,
            _N_LAYERS,
            figsize=(2.5 * _N_LAYERS, 2.5 * n_shown * 2),
            squeeze=False,
        )
        fig1.suptitle("Calorimeter images: original (top) | reconstructed (bottom)", fontsize=10)

        # Split energy from indicator (first half is energy, second half is indicator logits)
        n_pix = orig_list[0].shape[-1] // 2
        # Original: ground-truth energy (zeros where no cell)
        orig_list = [t[..., :n_pix] for t in orig_list]
        # Reconstructed: zero out pixels where predicted indicator < 0.5
        recon_list = [
            t[..., :n_pix] * (torch.sigmoid(t[..., n_pix:]) > 0.5).float() for t in recon_list
        ]

        for ev_idx, (orig_e, recon_e) in enumerate(zip(orig_list, recon_list)):
            for li in range(_N_LAYERS):
                start = self.layer_starts[li]
                end = start + self.patches_per_layer[li]

                orig_img = _patches_to_image(orig_e[start:end], li, self.bins_to_keep)
                recon_img = _patches_to_image(recon_e[start:end], li, self.bins_to_keep)

                vmax = max(orig_img.max(), 1e-6)
                for row_offset, img in enumerate([orig_img, recon_img]):
                    ax = axes1[ev_idx * 2 + row_offset, li]
                    ax.imshow(
                        img.T,
                        origin="lower",
                        cmap="viridis",
                        aspect="auto",
                        interpolation="nearest",
                        vmin=0,
                        vmax=vmax,
                    )
                    _add_patch_grid(ax, self.bins_to_keep[li], _PATCH_SIZES[li])
                    ax.tick_params(labelsize=5)
                    if ev_idx == 0 and row_offset == 0:
                        ax.set_title(f"L{li}", fontsize=8)
                    if li == 0:
                        label = "orig" if row_offset == 0 else "recon"
                        ax.set_ylabel(f"ev{ev_idx} {label}", fontsize=6)

        fig1.tight_layout()
        trainer.logger.experiment.log(
            {"val/calo_images": wandb.Image(fig1), "epoch": trainer.current_epoch}
        )
        plt.close(fig1)

        # ── Figure 2: per-layer energy histograms ─────────────────────────────
        all_orig_e = torch.stack(orig_list)  # [N, n_patches, 64]
        all_recon_e = torch.stack(recon_list)

        fig2, axes2 = plt.subplots(2, 3, figsize=(12, 7))
        fig2.suptitle("Per-layer energy pixel distributions (non-zero pixels)", fontsize=10)
        axes2 = axes2.flatten()

        for li in range(_N_LAYERS):
            start = self.layer_starts[li]
            end = start + self.patches_per_layer[li]
            ps2 = _PATCH_SIZES[li] ** 2

            orig_px = all_orig_e[:, start:end, :ps2].numpy().ravel()
            recon_px = all_recon_e[:, start:end, :ps2].numpy().ravel()

            orig_nz = orig_px[orig_px > 0]
            recon_nz = recon_px[recon_px > 0]

            ax = axes2[li]
            if len(orig_nz) > 0:
                bins = np.linspace(
                    0, max(orig_nz.max(), recon_nz.max() if len(recon_nz) else 1e-6), 40
                )
                ax.hist(orig_nz, bins=bins, histtype="step", label="orig", density=True)
                if len(recon_nz) > 0:
                    ax.hist(recon_nz, bins=bins, histtype="step", label="recon", density=True)
            ax.set_title(f"Layer {li}", fontsize=8)
            ax.set_xlabel("energy^0.3", fontsize=7)
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=6)

        fig2.tight_layout()
        trainer.logger.experiment.log(
            {"val/layer_energy_hist": wandb.Image(fig2), "epoch": trainer.current_epoch}
        )
        plt.close(fig2)

        # ── Figure 3: patch occupancy per layer ───────────────────────────────
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        x = np.arange(_N_LAYERS)
        width = 0.35

        orig_occ = []
        recon_occ = []
        for li in range(_N_LAYERS):
            start = self.layer_starts[li]
            end = start + self.patches_per_layer[li]
            ps2 = _PATCH_SIZES[li] ** 2
            orig_occ.append(float((all_orig_e[:, start:end, :ps2].sum(-1) > 0).float().mean()))
            recon_occ.append(float((all_recon_e[:, start:end, :ps2].sum(-1) > 0).float().mean()))

        ax3.bar(x - width / 2, orig_occ, width, label="orig")
        ax3.bar(x + width / 2, recon_occ, width, label="recon")
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"L{i}" for i in range(_N_LAYERS)])
        ax3.set_ylabel("Mean patch occupancy")
        ax3.set_title("Fraction of non-zero patches per layer")
        ax3.legend()
        ax3.set_ylim(0, 1)
        fig3.tight_layout()
        trainer.logger.experiment.log(
            {"val/patch_occupancy": wandb.Image(fig3), "epoch": trainer.current_epoch}
        )
        plt.close(fig3)
