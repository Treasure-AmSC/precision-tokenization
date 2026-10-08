"""Vector-quantization layers with dead-code revival.

``vector_quantize_pytorch`` has no protection against codebook
collapse: once a code's EMA cluster size decays to ~0 it is never revived.
These subclasses periodically reinitialize dead codes to a real encoder
output vector from the current batch. With ``threshold_ema_dead_code <= 0``
revival is disabled and behavior is identical to the base classes.
"""

import torch

from vector_quantize_pytorch import VectorQuantize


class RevivingVectorQuantize(VectorQuantize):
    """VectorQuantize with dead-code revival.

    Args:
        threshold_ema_dead_code: codebook entries whose EMA ``cluster_size``
            falls below this threshold are reinitialized (during training
            only) to a random encoder output vector from the current batch.
            Set to 0 to disable (identical behavior to the base class).
    """

    def __init__(self, *args, threshold_ema_dead_code: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold_ema_dead_code = threshold_ema_dead_code

    def forward(self, input: torch.Tensor):
        quantize, embed_ind, loss = super().forward(input)

        if self.training and self.threshold_ema_dead_code > 0:
            with torch.no_grad():
                dead_mask = self.cluster_size < self.threshold_ema_dead_code
                n_dead = int(dead_mask.sum().item())
                if n_dead > 0:
                    flatten = input.reshape(-1, self.dim)
                    n_samples = flatten.shape[0]
                    if n_samples > 0:
                        rand_idx = torch.randint(0, n_samples, (n_dead,), device=flatten.device)
                        replacement = flatten[rand_idx].t()  # [dim, n_dead]
                        self.embed.data[:, dead_mask] = replacement
                        self.embed_avg.data[:, dead_mask] = replacement
                        # Reset cluster_size just above threshold so revived
                        # codes aren't immediately re-flagged as dead next step.
                        self.cluster_size.data[dead_mask] = self.threshold_ema_dead_code

        return quantize, embed_ind, loss


class RevivingResidualVQ(torch.nn.Module):
    """Same as vector_quantize_pytorch.ResidualVQ, but stacks
    RevivingVectorQuantize layers instead of plain VectorQuantize."""

    def __init__(self, *, num_quantizers: int, **kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [RevivingVectorQuantize(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x: torch.Tensor):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        for layer in self.layers:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        all_losses, all_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, all_indices, all_losses
