"""Checks for the latent-binning bottleneck (run: python tests/test_latent_bins.py, or pytest)."""

from functools import partial

import torch

from heptokens.vendored.models.coders import CellDecoder, Encoder
from heptokens.vendored.models.latent_bins import LatentBins
from heptokens.vendored.models.transformer import Transformer
from heptokens.vendored.models.vq_layers import RevivingResidualVQ
from heptokens.vendored.models.vq_vae import LitVqVae


def test_eval_latent_is_the_bin_centre():
    bins = LatentBins(dim=3, num_bins=4, bin_range=1.0, noise_std_bins=0.29).eval()
    h = torch.tensor([[[0.0, 50.0, -50.0]]])  # mu = 0, ~1, ~-1
    z, indices, loss = bins(h)
    assert indices[:, 0, 0].tolist() == [2, 3, 0]
    assert torch.allclose(z, torch.tensor([[[0.25, 0.75, -0.75]]]))
    assert indices.shape == (3, 1, 1) and loss.shape == (3,) and not loss.any()

    h = 3 * torch.randn(8, 5, 3)
    z, indices, _ = bins(h)
    mu = torch.tanh(h)
    assert (z - mu).abs().max() <= 0.25 + 1e-6
    assert torch.equal(bins.get_output_from_indices(indices.permute(1, 2, 0)), z)


def test_training_noise_is_noise_std_bins_bin_widths():
    bins = LatentBins(dim=4, num_bins=8, bin_range=2.0, noise_std_bins=0.29).train()
    z, indices, _ = bins(torch.zeros(1000, 50, 4))
    assert abs(z.std().item() / (0.29 * 0.5) - 1) < 0.02
    assert (indices == 4).all()  # tokens are mu's bins, noise-free


def make_model(**quantizer) -> LitVqVae:
    coder = partial(Transformer, d_model=16, n_heads=2, num_layers=1, dim_feedforward=32, pos_dim=3)
    keys = dict(input_key="energy", mask_key="mask", positions_key="positions")
    torch.manual_seed(0)
    return LitVqVae(
        encoder=partial(Encoder, model=coder, **keys),
        decoder=partial(CellDecoder, model=coder, **keys),
        codebook_dim=4,
        input_key="energy",
        data_sample=make_batch(),
        **quantizer,
    )


def make_batch() -> dict:
    torch.manual_seed(1)
    mask = torch.rand(2, 6) > 0.3
    energy = torch.rand(2, 6, 128) * mask.unsqueeze(-1)
    energy[..., 64:] = (energy[..., :64] > 0.5).float()
    return {"energy": energy, "mask": mask, "positions": torch.randn(2, 6, 3)}


def test_tokens_decode_as_the_eval_latent():
    model = make_model(
        codebook_size=None,
        num_quantizers=None,
        latent_bins=dict(num_bins=4096, bin_range=4.0, noise_std_bins=0.29),
    )
    assert isinstance(model.vector_quantization, LatentBins) and model.codebook_size == 4096
    batch = make_batch()
    assert torch.isfinite(model.training_step(batch, 0))

    model.eval()
    z_q, indices, _ = model.encode(batch)
    assert torch.equal(model.encode_indices(batch), indices)
    assert (indices[~batch["mask"]] == -1).all() and (indices[batch["mask"]] >= 0).all()
    z_tokens = model.vector_quantization.get_output_from_indices(indices.clamp(min=0))
    valid = batch["mask"]
    assert torch.equal(model.decode(z_tokens, batch)[valid], model.decode(z_q, batch)[valid])


def test_bins_reject_rvq_settings():
    try:
        make_model(
            codebook_size=1024,
            num_quantizers=None,
            latent_bins=dict(num_bins=16, bin_range=4.0, noise_std_bins=0.29),
        )
    except AssertionError:
        return
    raise AssertionError("latent_bins accepted a codebook_size")


def test_default_is_the_residual_vq():
    model = make_model(codebook_size=32, num_quantizers=2)
    assert isinstance(model.vector_quantization, RevivingResidualVQ)
    model.eval()
    batch = make_batch()
    assert torch.equal(model.encode_indices(batch), model.encode(batch)[1])


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: ok")
