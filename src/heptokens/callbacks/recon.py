# A callback to monitor reconstruction quality in original (unscaled) space.
import logging
from pathlib import Path

import numpy as np
import torch as T
from lightning.pytorch.callbacks import Callback
from sklearn.base import BaseEstimator

from heptokens.data.collation import inverse_preprocess_batch
from heptokens.utils.torch_utils import dict_to_device

log = logging.getLogger(__name__)


class ReconstructionMonitor(Callback):
    """Monitor reconstruction quality in original (unscaled) space.

    Args:
        cst_fn: Fitted transformer for constituents (e.g., QuantileTransformer)
        jet_fn: Fitted transformer for jets (e.g., QuantileTransformer)
        log_every_n_epochs: How often to compute unscaled metrics (default: 1)
        pos_tokenizer: PositionTokenizer for positions-out data (batch carries a separate
            ``positions`` tensor). Truth jets use the exact positions, reco jets the binned
            ones. Required iff the batch has ``positions``.
    """

    def __init__(
        self,
        cst_fn: BaseEstimator,
        jet_fn: BaseEstimator | None = None,
        log_every_n_epochs: int = 1,
        max_batches: int = 1,
        pt_idx: int = 0,
        deta_idx: int = 1,
        dphi_idx: int = 2,
        energy_idx: int | None = None,
        jet_pt_idx: int = 0,
        jet_mass_idx: int = 1,
        jet_eta_idx: int = 2,
        jet_phi_idx: int = 3,
        compute_jet_metrics: bool = True,
        iqr_med_pt_min: float = 5_000.0,
        iqr_med_pt_max: float = 140_000.0,
        iqr_med_n_bins: int = 20,
        iqr_med_min_entries_per_bin: int = 10,
        pos_tokenizer=None,
    ):
        super().__init__()
        self.cst_fn = cst_fn
        self.jet_fn = jet_fn
        self.pos_tokenizer = pos_tokenizer
        self.log_every_n_epochs = log_every_n_epochs
        self.max_batches = max_batches
        self.pt_idx = pt_idx
        self.deta_idx = deta_idx
        self.dphi_idx = dphi_idx
        self.energy_idx = energy_idx
        self.jet_pt_idx = jet_pt_idx
        self.jet_mass_idx = jet_mass_idx
        self.jet_eta_idx = jet_eta_idx
        self.jet_phi_idx = jet_phi_idx
        self.compute_jet_metrics = compute_jet_metrics
        self.iqr_med_pt_min = iqr_med_pt_min
        self.iqr_med_pt_max = iqr_med_pt_max
        self.iqr_med_n_bins = iqr_med_n_bins
        self.iqr_med_min_entries_per_bin = iqr_med_min_entries_per_bin
        self.save_metrics = False  # Set True externally to save metrics to disk

        # Store constituent features across batches for epoch-level plots
        self.cst_originals = []
        self.cst_recons = []

        if compute_jet_metrics:
            # Store jet metrics across batches for epoch-level summary
            self.jet_residuals = []

    @staticmethod
    def wrap_phi(phi):
        """Wrap angle(s) to [-pi, pi)."""
        return (phi + np.pi) % (2 * np.pi) - np.pi

    # Add helper method to compute jets from constituents
    def _compute_jet_from_constituents(self, csts, mask, jets=None):
        """Compute jet 4-vector from constituent pts, etas, phis.

        If jets is provided, constituent eta/phi are treated as relative (deta, dphi)
        and offset by jet coordinates. If jets is None, constituent eta/phi are absolute.

        Args:
            csts: [batch, n_constituents, features] where features are [pt, eta/deta, phi/dphi, ...]
            mask: [batch_size, n_constituents] boolean mask of valid constituents
            jets: optional [batch, jet_features] for relative coordinate offset

        Returns:
            dict with jet pt, mass, eta, phi
        """
        csts_detas = csts[:, :, self.deta_idx].cpu().numpy()
        csts_dphis = csts[:, :, self.dphi_idx].cpu().numpy()

        if jets is not None:
            jet_etas = jets[:, self.jet_eta_idx].cpu().numpy()
            jet_phis = jets[:, self.jet_phi_idx].cpu().numpy()
            csts_phis = self._delta_phi(csts_dphis + jet_phis[:, None], 0.0)
            csts_etas = csts_detas + jet_etas[:, None]
        else:
            csts_etas = csts_detas
            csts_phis = csts_dphis

        if self.energy_idx is not None:
            # Massless calo clusters: pT = E / cosh(eta)
            csts_energies = csts[:, :, self.energy_idx].cpu().numpy()
            csts_pts = csts_energies / np.cosh(csts_etas)
        else:
            csts_pts = csts[:, :, self.pt_idx].cpu().numpy()

        pxs = csts_pts * np.cos(csts_phis)
        pys = csts_pts * np.sin(csts_phis)
        pzs = csts_pts * np.sinh(csts_etas)
        energies = csts_pts * np.cosh(csts_etas)

        mask_np = mask.cpu().numpy()

        # Sum for valid constituents only
        reco_jet_pxs = np.sum(np.where(mask_np, pxs, 0), axis=-1)
        reco_jet_pys = np.sum(np.where(mask_np, pys, 0), axis=-1)
        reco_jet_pzs = np.sum(np.where(mask_np, pzs, 0), axis=-1)
        reco_jet_pts = np.sqrt(reco_jet_pxs**2 + reco_jet_pys**2)
        reco_jet_pts = np.where(reco_jet_pts == 0, 1e-10, reco_jet_pts)
        reco_jet_etas = np.arcsinh(reco_jet_pzs / reco_jet_pts)
        reco_jet_phis = np.arctan2(reco_jet_pys, reco_jet_pxs)

        reco_jet_energies = np.sum(np.where(mask_np, energies, 0), axis=-1)
        reco_jet_m2s = reco_jet_energies**2 - (reco_jet_pxs**2 + reco_jet_pys**2 + reco_jet_pzs**2)
        reco_jet_masses = np.sqrt(np.clip(reco_jet_m2s, 0.0, None))

        return {
            "pt": reco_jet_pts,
            "mass": reco_jet_masses,
            "eta": reco_jet_etas,
            "phi": reco_jet_phis,
        }

    def _delta_phi(self, phi1, phi2):
        """Compute delta phi wrapped to [-pi, pi]."""
        dphi = phi1 - phi2
        return np.arctan2(np.sin(dphi), np.cos(dphi))

    def _directions_from_positions(self, positions, jets=None):
        """(eta, cos phi, sin phi) per constituent from a separate ``positions`` tensor.

        positions: [..., 3] absolute (eta, cos phi, sin phi), or [..., 2] (deta, dphi)
        relative to the jet axis taken from ``jets``.
        """
        pos = positions.cpu().numpy()
        if pos.shape[-1] == 3:
            return pos[..., 0], pos[..., 1], pos[..., 2]
        if pos.shape[-1] != 2:
            raise ValueError(f"positions must have 2 or 3 columns, got {pos.shape[-1]}")
        if jets is None:
            raise ValueError("(deta, dphi) positions need a 'jets' tensor for the jet axis")
        etas = pos[..., 0] + jets[:, self.jet_eta_idx].cpu().numpy()[:, None]
        phis = self._delta_phi(pos[..., 1] + jets[:, self.jet_phi_idx].cpu().numpy()[:, None], 0.0)
        return etas, np.cos(phis), np.sin(phis)

    def _compute_jet_from_positions(self, csts, etas, cosphis, sinphis, mask):
        """Jet 4-vector with directions from positions; same formula as heptokens-cocoa's
        TopoReconstructionMonitor._compute_jet_from_clusters."""
        if self.energy_idx is not None:
            pts = csts[:, :, self.energy_idx].cpu().numpy() / np.cosh(etas)
        else:
            pts = csts[:, :, self.pt_idx].cpu().numpy()

        pxs = pts * cosphis
        pys = pts * sinphis
        pzs = pts * np.sinh(etas)
        energies = pts * np.cosh(etas)

        mask_np = mask.cpu().numpy()
        jet_px = np.sum(np.where(mask_np, pxs, 0), axis=-1)
        jet_py = np.sum(np.where(mask_np, pys, 0), axis=-1)
        jet_pz = np.sum(np.where(mask_np, pzs, 0), axis=-1)
        jet_pt = np.sqrt(jet_px**2 + jet_py**2)
        jet_pt = np.where(jet_pt == 0, 1e-10, jet_pt)
        jet_eta = np.arcsinh(jet_pz / jet_pt)
        jet_phi = np.arctan2(jet_py, jet_px)

        jet_e = np.sum(np.where(mask_np, energies, 0), axis=-1)
        jet_m2 = jet_e**2 - (jet_px**2 + jet_py**2 + jet_pz**2)
        jet_mass = np.sqrt(np.clip(jet_m2, 0.0, None))

        return {"pt": jet_pt, "mass": jet_mass, "eta": jet_eta, "phi": jet_phi}

    def _on_batch_end(self, trainer, pl_module, batch, batch_idx, stage="val"):
        """Shared logic for computing metrics in original space."""
        if batch_idx >= self.max_batches:
            return

        has_positions = "positions" in batch
        if self.pos_tokenizer is not None and not has_positions:
            raise ValueError("pos_tokenizer is set but the batch has no 'positions' tensor")
        if has_positions and self.pos_tokenizer is None:
            raise ValueError(
                "batch has a 'positions' tensor (positions-out data) but no pos_tokenizer is set"
            )

        # Get reconstructions
        with T.no_grad():
            encoding = pl_module.encode(batch)[0]
            recon_scaled = pl_module.decode(encoding, batch)

        # Create reconstruction dict in scaled space
        recon_dict_scaled = {
            "csts": recon_scaled,
            "mask": batch["mask"],
        }
        if "jets" in batch:
            recon_dict_scaled["jets"] = batch["jets"].clone()

        # Inverse transform
        original_unscaled = inverse_preprocess_batch(
            dict_to_device(batch, "cpu"), self.cst_fn, self.jet_fn
        )
        recon_unscaled = inverse_preprocess_batch(
            dict_to_device(recon_dict_scaled, "cpu"), self.cst_fn, self.jet_fn
        )

        # Constituent-level metrics
        mask = batch["mask"].cpu()
        original_csts = original_unscaled["csts"][mask]
        recon_csts = recon_unscaled["csts"][mask]

        unscaled_mae = T.mean(T.abs(original_csts - recon_csts))
        unscaled_mse = T.mean((original_csts - recon_csts) ** 2)

        pl_module.log(f"{stage}/unscaled_mae", unscaled_mae, prog_bar=False)
        pl_module.log(f"{stage}/unscaled_mse", unscaled_mse, prog_bar=False)

        # Store constituent features for epoch-level scatter plots
        self.cst_originals.append(original_csts.cpu().numpy())
        self.cst_recons.append(recon_csts.cpu().numpy())

        # Jet-level metrics (pt-based only, since coordinates are relative)
        if self.compute_jet_metrics:
            # Compute jets from original and reconstructed constituents
            """
            jet_truth = {
                "pt": original_unscaled["jets"][:, self.jet_pt_idx].cpu().numpy(),
                "mass": original_unscaled["jets"][:, self.jet_mass_idx].cpu().numpy(),
                "eta": original_unscaled["jets"][:, self.jet_eta_idx].cpu().numpy(),
                "phi": self._delta_phi(
                    original_unscaled["jets"][:, self.jet_phi_idx].cpu().numpy(), 0.0
                ),
            }
            """
            jets_for_offset = original_unscaled.get("jets")
            if has_positions:
                pos_truth = batch["positions"].cpu()
                pos_reco = self.pos_tokenizer.decode(self.pos_tokenizer(pos_truth))
                dir_truth = self._directions_from_positions(pos_truth, jets_for_offset)
                dir_reco = self._directions_from_positions(pos_reco, jets_for_offset)
                jet_truth = self._compute_jet_from_positions(
                    original_unscaled["csts"], *dir_truth, mask
                )
                jet_reco = self._compute_jet_from_positions(recon_unscaled["csts"], *dir_reco, mask)
            else:
                jet_truth = self._compute_jet_from_constituents(
                    original_unscaled["csts"], mask, jets_for_offset
                )
                jet_reco = self._compute_jet_from_constituents(
                    recon_unscaled["csts"], mask, jets_for_offset
                )

            # Compute pt residuals and radial distance
            pt_residuals = jet_truth["pt"] - jet_reco["pt"]

            pt_ratio = np.divide(
                jet_reco["pt"],
                jet_truth["pt"],
                out=np.full_like(jet_reco["pt"], np.nan),
                where=jet_truth["pt"] > 0,
            )

            mass_residuals = jet_truth["mass"] - jet_reco["mass"]
            eta_residuals = jet_truth["eta"] - jet_reco["eta"]
            phi_residuals = self._delta_phi(jet_truth["phi"], jet_reco["phi"])

            # Relative residuals: (truth - reco) / truth
            # Use a floor on |truth| to avoid division by zero
            _floor = 1e-3
            pt_rel_residuals = np.divide(
                pt_residuals, jet_truth["pt"],
                out=np.full_like(pt_residuals, np.nan),
                where=np.abs(jet_truth["pt"]) > _floor,
            )
            mass_rel_residuals = np.divide(
                mass_residuals, jet_truth["mass"],
                out=np.full_like(mass_residuals, np.nan),
                where=np.abs(jet_truth["mass"]) > _floor,
            )
            eta_rel_residuals = np.divide(
                eta_residuals, jet_truth["eta"],
                out=np.full_like(eta_residuals, np.nan),
                where=np.abs(jet_truth["eta"]) > _floor,
            )
            phi_rel_residuals = np.divide(
                phi_residuals, jet_truth["phi"],
                out=np.full_like(phi_residuals, np.nan),
                where=np.abs(jet_truth["phi"]) > _floor,
            )

            # Compute mean radial distance in (deta, dphi) space
            mask_np = mask.cpu().numpy()
            if has_positions:
                eta_t, cos_t, sin_t = (a[mask_np] for a in dir_truth)
                eta_r, cos_r, sin_r = (a[mask_np] for a in dir_reco)
                dphi = np.arctan2(sin_t * cos_r - cos_t * sin_r, cos_t * cos_r + sin_t * sin_r)
                radial_distance = np.sqrt((eta_t - eta_r) ** 2 + dphi**2)
            else:
                original_valid = original_unscaled["csts"][mask_np]
                recon_valid = recon_unscaled["csts"][mask_np]
                original_deta = original_valid[:, self.deta_idx].cpu().numpy()
                original_dphi = original_valid[:, self.dphi_idx].cpu().numpy()
                recon_deta = recon_valid[:, self.deta_idx].cpu().numpy()
                recon_dphi = recon_valid[:, self.dphi_idx].cpu().numpy()

                radial_distance = np.sqrt(
                    (original_deta - recon_deta) ** 2 + (original_dphi - recon_dphi) ** 2
                )

            residuals = {
                "pt": pt_residuals,
                "mass": mass_residuals,
                "eta": eta_residuals,
                "phi": phi_residuals,
                "pt_rel": pt_rel_residuals,
                "mass_rel": mass_rel_residuals,
                "eta_rel": eta_rel_residuals,
                "phi_rel": phi_rel_residuals,
                "radial_dist": radial_distance,
                "truth_pt": jet_truth["pt"],
                "pt_ratio": pt_ratio,
                "reco_pt": jet_reco["pt"],
            }

            # Store for epoch-level aggregation
            self.jet_residuals.append(residuals)

            # Log immediate metrics
            pl_module.log(f"{stage}/jet_pt_bias", float(np.mean(pt_residuals)), prog_bar=False)
            pl_module.log(f"{stage}/jet_pt_resolution", float(np.std(pt_residuals)), prog_bar=False)
            pl_module.log(
                f"{stage}/constituent_radial_dist", float(np.mean(radial_distance)), prog_bar=False
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Compute metrics in original space during validation."""
        if trainer.sanity_checking:
            return
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        self._on_batch_end(trainer, pl_module, batch, batch_idx, stage="val")

    def _on_epoch_end(self, trainer, pl_module, stage="val"):
        """Shared logic for creating summary plots at end of epoch."""
        # Feature scatter plots from accumulated batches
        all_original = None
        all_recon = None
        if self.cst_originals:
            all_original = np.concatenate(self.cst_originals, axis=0)
            all_recon = np.concatenate(self.cst_recons, axis=0)
            self.cst_originals.clear()
            self.cst_recons.clear()

        all_residuals = None
        if self.compute_jet_metrics and self.jet_residuals:
            all_residuals = {
                key: np.concatenate([batch[key] for batch in self.jet_residuals])
                for key in self.jet_residuals[0].keys()
            }
            self.jet_residuals.clear()

        # Save raw metrics to disk when enabled
        if self.save_metrics:
            output_dir = Path(trainer.default_root_dir) / "test_metrics"
            output_dir.mkdir(parents=True, exist_ok=True)
            save_dict = {}
            if all_original is not None:
                save_dict["cst_original"] = all_original
                save_dict["cst_recon"] = all_recon
            if all_residuals is not None:
                save_dict.update(all_residuals)
            if save_dict:
                np.savez(output_dir / "test_metrics.npz", **save_dict)
                log.info(f"Saved test metrics to {output_dir / 'test_metrics.npz'}")

        # Plot constituent scatter plots
        if all_original is not None and trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            import matplotlib.pyplot as plt
            import wandb

            n_features = all_original.shape[1]
            n_cols = min(4, n_features)
            n_rows = int(np.ceil(n_features / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
            axes = np.atleast_1d(axes).flatten()

            for feature_idx in range(n_features):
                ax = axes[feature_idx]
                orig = all_original[:, feature_idx]
                reco = all_recon[:, feature_idx]
                ax.scatter(orig, reco, alpha=0.3, s=1)
                ax.set_xlabel("Original")
                ax.set_ylabel("Reconstructed")
                ax.set_title(f"Feature {feature_idx}")
                ax.plot(
                    [orig.min(), orig.max()],
                    [orig.min(), orig.max()],
                    "r--",
                )

            for i in range(n_features, len(axes)):
                axes[i].axis("off")

            fig.tight_layout()
            trainer.logger.experiment.log(
                {f"{stage}/recon_features": wandb.Image(fig), "epoch": trainer.current_epoch}
            )
            plt.close(fig)

        if all_residuals is None:
            return

        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            import matplotlib.pyplot as plt
            import mplhep as hep
            import wandb

            # Use HEP style
            plt.style.use(hep.style.CMS)

            # Create residual plots
            plot_order = ["pt", "mass", "eta", "phi", "pt_rel", "mass_rel", "eta_rel", "phi_rel", "radial_dist", "truth_pt", "reco_pt"]
            labels = {
                "pt": "Jet pt residual (truth - reco)",
                "mass": "Jet mass residual (truth - reco)",
                "eta": "Jet eta residual (truth - reco)",
                "phi": "Jet phi residual (truth - reco)",
                "pt_rel": "Jet pt relative residual (truth - reco) / truth",
                "mass_rel": "Jet mass relative residual (truth - reco) / truth",
                "eta_rel": "Jet eta relative residual (truth - reco) / truth",
                "phi_rel": "Jet phi relative residual (truth - reco) / truth",
                "radial_dist": "Constituent radial distance",
                "truth_pt": "Truth jet pt [MeV]",
                "reco_pt": "Reconstructed jet pt [MeV]",
            }
            vars_to_plot = [key for key in plot_order if key in all_residuals]
            n_vars = len(vars_to_plot)
            n_cols = min(3, n_vars)
            n_rows = int(np.ceil(n_vars / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows))
            axes = np.atleast_1d(axes).flatten()

            for i, var in enumerate(vars_to_plot):
                ax = axes[i]
                label = labels.get(var, var)
                residuals = all_residuals[var]

                # Plot histogram (filter non-finite values)
                residuals = all_residuals[var]
                finite = residuals[np.isfinite(residuals)]

                if finite.size == 0:
                    ax.text(
                        0.5,
                        0.5,
                        f"No finite values in {var}",
                        ha="center",
                        va="center",
                        fontsize=14,
                        transform=ax.transAxes,
                        color="red",
                        weight="bold",
                    )
                else:
                    n_nan = residuals.size - finite.size
                    ax.hist(finite, bins=50, histtype="step", linewidth=2)

                    # Add statistics text
                    mean_val = np.mean(finite)
                    std_val = np.std(finite)
                    median_val = np.median(finite)
                    stats = f"Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nStd: {std_val:.3f}"
                    if n_nan > 0:
                        stats += f"\nNaN: {n_nan}"
                    ax.text(
                        0.05,
                        0.95,
                        stats,
                        transform=ax.transAxes,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )
                ax.set_xlabel(label, fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
                ax.grid(True, alpha=0.3)

            # Hide unused axes
            for i in range(n_vars, len(axes)):
                axes[i].axis("off")

            fig.tight_layout()
            trainer.logger.experiment.log(
                {f"{stage}/jet_residuals": wandb.Image(fig), "epoch": trainer.current_epoch}
            )
            plt.close(fig)

            truth_pt = all_residuals.get("truth_pt")
            pt_ratio = all_residuals.get("pt_ratio")
            if truth_pt is None or pt_ratio is None:
                return
            truth_pt = np.asarray(truth_pt).reshape(-1)
            pt_ratio = np.asarray(pt_ratio).reshape(-1)
            if truth_pt.size != pt_ratio.size:
                min_size = min(truth_pt.size, pt_ratio.size)
                log.warning(
                    "Mismatched truth_pt (%d) and pt_ratio (%d) lengths; truncating to %d.",
                    truth_pt.size,
                    pt_ratio.size,
                    min_size,
                )
                truth_pt = truth_pt[:min_size]
                pt_ratio = pt_ratio[:min_size]

            finite_mask = np.isfinite(truth_pt) & np.isfinite(pt_ratio) & (pt_ratio > 0)
            truth_pt = truth_pt[finite_mask]
            pt_ratio = pt_ratio[finite_mask]
            if truth_pt.size == 0:
                return

            pt_min = self.iqr_med_pt_min
            pt_max = self.iqr_med_pt_max
            n_bins = self.iqr_med_n_bins
            min_entries_per_bin = self.iqr_med_min_entries_per_bin

            pt_bin_edges = np.linspace(pt_min, pt_max, n_bins + 1)
            pt_bin_centers = 0.5 * (pt_bin_edges[:-1] + pt_bin_edges[1:])
            iqr_over_median = np.full(n_bins, np.nan)

            for i in range(n_bins):
                in_bin = truth_pt >= pt_bin_edges[i]
                if i == n_bins - 1:
                    in_bin &= truth_pt <= pt_bin_edges[i + 1]
                else:
                    in_bin &= truth_pt < pt_bin_edges[i + 1]

                ratios_in_bin = pt_ratio[in_bin]
                if ratios_in_bin.size < min_entries_per_bin:
                    continue

                median = np.median(ratios_in_bin)
                if not np.isfinite(median) or median <= 0:
                    continue

                q25, q75 = np.percentile(ratios_in_bin, [25, 75])
                iqr_over_median[i] = (q75 - q25) / median

            fig, ax = plt.subplots(figsize=(6, 6))
            valid = np.isfinite(iqr_over_median)
            if np.any(valid):
                ax.plot(pt_bin_centers[valid], iqr_over_median[valid], marker="o", linewidth=0)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No bins with sufficient entries",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )

            ax.set_xlim(pt_min, pt_max)
            ax.set_ylim(0.0, np.nanmax(iqr_over_median) * 2)
            ax.set_xlabel("Truth jet $p_T$ [MeV]", fontsize=12)
            ax.set_ylabel("IQR(reco/truth) / median(reco/truth)", fontsize=12)
            fig.tight_layout()
            trainer.logger.experiment.log(
                {
                    f"{stage}/jet_pt_response_iqr_over_median_vs_truth_pt": wandb.Image(fig),
                    "epoch": trainer.current_epoch,
                }
            )
            plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Create summary plots at end of validation epoch."""
        self._on_epoch_end(trainer, pl_module, stage="val")


