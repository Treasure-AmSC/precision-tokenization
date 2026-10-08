"""Generic VQ-VAE reconstruction monitoring callback."""

import logging

import torch as T
from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)


class BaseReconstructionMonitor(Callback):
    """Base class for VQ-VAE reconstruction monitoring callbacks.

    Can be used directly for a quick modality-agnostic sanity check: it logs
    ``val/scaled_mae`` and ``val/scaled_mse`` in the *preprocessed* (scaled)
    feature space using the first ``input_key`` tensor found in the batch.

    For meaningful physical-space metrics, subclass and override
    ``compute_and_log_metrics`` with modality-specific logic.

    Example::

        class MyCaloMonitor(BaseReconstructionMonitor):
            def __init__(self, inverse_fn, **kwargs):
                super().__init__(**kwargs)
                self.inverse_fn = inverse_fn

            def compute_and_log_metrics(self, trainer, pl_module, batch, batch_idx):
                with T.no_grad():
                    z_q = pl_module.encode(batch)[0]
                    recon = pl_module.decode(z_q, batch)
                original = self.inverse_fn(batch["calo"])
                reconstructed = self.inverse_fn(recon)
                pl_module.log("val/calo_mae", T.mean(T.abs(original - reconstructed)))

    Args:
        input_key: Key in the batch dict for the primary input tensor.
        mask_key: Key in the batch dict for the boolean validity mask.
            Set to ``None`` if the modality has no variable-length padding.
        log_every_n_epochs: Run metrics every N validation epochs.
        max_batches: Number of validation batches to process per epoch.
    """

    def __init__(
        self,
        input_key: str = "csts",
        mask_key: str | None = "mask",
        log_every_n_epochs: int = 1,
        max_batches: int = 1,
    ):
        super().__init__()
        self.input_key = input_key
        self.mask_key = mask_key
        self.log_every_n_epochs = log_every_n_epochs
        self.max_batches = max_batches

    def compute_and_log_metrics(self, trainer, pl_module, batch: dict, batch_idx: int) -> None:
        """Compute and log reconstruction metrics.

        Default implementation logs MAE and MSE in the *scaled* feature space.
        Override in subclasses to add physical-unit metrics.

        Args:
            trainer: Lightning Trainer
            pl_module: The VQ-VAE Lightning module (exposes .encode / .decode)
            batch: Current validation batch dict
            batch_idx: Batch index within the validation epoch
        """
        with T.no_grad():
            z_q = pl_module.encode(batch)[0]
            recon = pl_module.decode(z_q, batch)

        targets = batch[self.input_key]
        mask = batch.get(self.mask_key) if self.mask_key else None

        if mask is not None:
            diff = recon[mask] - targets[mask]
        else:
            diff = recon - targets

        pl_module.log("val/scaled_mae", T.mean(T.abs(diff)), prog_bar=False)
        pl_module.log("val/scaled_mse", T.mean(diff**2), prog_bar=False)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.sanity_checking:
            return
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        if batch_idx >= self.max_batches:
            return
        self.compute_and_log_metrics(trainer, pl_module, batch, batch_idx)
