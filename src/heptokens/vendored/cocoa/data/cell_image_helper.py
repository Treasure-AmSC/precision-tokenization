import awkward as ak
import numpy as np
import torch


class CellImageHelper:
    def __init__(self, pow2_scale, downsample: int = 1):
        """
        Args:
            pow2_scale: per-layer patch size exponents (patch_size = 2**pow2_scale[i]).
            downsample: coarsen the pixel grid by this integer factor relative to
                the native per-layer calorimeter cell pitch (grans below). The
                default native grans exactly match the detector's real cell
                granularity (see scripts/make_tokenization_schematic.py's
                docstring for the empirical verification: every occupied pixel
                contains exactly one cell at downsample=1, i.e. no summing
                occurs).

                Setting downsample > 1 divides BOTH grans and bins_to_keep_dict
                by this factor, so the ROI's *physical* eta-phi window stays
                fixed (bins_to_keep * (6/grans) is invariant to downsample —
                same window as downsample=1) while each pixel becomes
                downsample-times wider, so multiple cells genuinely land in,
                and get summed (via np.histogram2d(weights=energy)) into, the
                same pixel. This also means the *number* of pixels/patches per
                layer shrinks by downsample**2 (a real coarsening of a fixed
                physical region, not a bigger field of view — an earlier
                version of this class kept bins_to_keep_dict fixed instead,
                which just enlarged the ROI window around the jet rather than
                genuinely downsampling it, adding empty margin instead of
                summing more cells within the same window).

                Must evenly divide every entry of the native grans list
                [256, 256, 128, 64, 64, 32] AND of bins_to_keep_dict's values
                [64, 64, 32, 16, 16, 8]. In practice only downsample=1 (native)
                and downsample=2 are usable with this pow2_scale/bins_to_keep
                geometry: downsample=4 and 8 divide evenly but shrink at least
                one layer's bins_to_keep below its patch_size, leaving that
                layer with zero patches (verified: raises ValueError below).
        """
        native_grans = [256, 256, 128, 64, 64, 32]
        native_bins_to_keep = [64, 64, 32, 16, 16, 8]
        if any(g % downsample != 0 for g in native_grans):
            raise ValueError(
                f"downsample={downsample} must evenly divide every native gran in "
                f"{native_grans} (valid values: 1, 2, 4, 8, 16, 32)"
            )
        if any(b % downsample != 0 for b in native_bins_to_keep):
            raise ValueError(
                f"downsample={downsample} must evenly divide every native "
                f"bins_to_keep value in {native_bins_to_keep} too (so the physical "
                "ROI window stays fixed) — valid values are 1, 2, 4, 8"
            )
        self.downsample = downsample
        self.grans = [g // downsample for g in native_grans]
        self.bins_to_keep_dict = {
            i: b // downsample for i, b in enumerate(native_bins_to_keep)
        }
        if any(g < self.bins_to_keep_dict[i] for i, g in enumerate(self.grans)):
            raise ValueError(
                f"downsample={downsample} makes grans={self.grans} smaller than "
                f"bins_to_keep_dict={self.bins_to_keep_dict} for at least one layer"
            )
        self.n_layers = len(self.grans)

        # it's same in both eta and phi directions
        self.num_patches_per_layer_1d = []
        for i in range(self.n_layers):
            patch_size = 2 ** pow2_scale[i]
            n_patches_1d = self.bins_to_keep_dict[i] // patch_size
            if n_patches_1d < 1:
                raise ValueError(
                    f"downsample={downsample} shrinks layer {i}'s bins_to_keep to "
                    f"{self.bins_to_keep_dict[i]}, smaller than its patch_size "
                    f"({patch_size}) — that layer would contribute zero patches. "
                    "Try a smaller downsample."
                )
            self.num_patches_per_layer_1d.append(n_patches_1d)

        self.eta_bin_edges_dict = {}
        self.phi_bin_edges_dict = {}
        for i, g in enumerate(self.grans):
            self.eta_bin_edges_dict[i] = np.linspace(-3, 3, g + 1)
            self.phi_bin_edges_dict[i] = np.linspace(-2 * np.pi, 2 * np.pi, 2 * g + 1)

        self.eta_bin_mids_dict = {}
        self.phi_bin_mids_dict = {}
        for i in range(self.n_layers):
            self.eta_bin_mids_dict[i] = torch.from_numpy(
                0.5 * (self.eta_bin_edges_dict[i][:-1] + self.eta_bin_edges_dict[i][1:])
            )
            self.phi_bin_mids_dict[i] = torch.from_numpy(
                0.5 * (self.phi_bin_edges_dict[i][:-1] + self.phi_bin_edges_dict[i][1:])
            )

    def to_device(self, device):
        for i in range(self.n_layers):
            self.eta_bin_mids_dict[i] = self.eta_bin_mids_dict[i].to(device)
            self.phi_bin_mids_dict[i] = self.phi_bin_mids_dict[i].to(device)

    def get_left_right_edge_idx(self, refbin, i):
        left_bin = refbin - self.bins_to_keep_dict[i] // 2
        if left_bin < 0:
            left_bin = 0
            right_bin = self.bins_to_keep_dict[i]
            return left_bin, right_bin

        right_bin = refbin + self.bins_to_keep_dict[i] // 2
        if right_bin > self.grans[i]:
            right_bin = self.grans[i]
            left_bin = right_bin - self.bins_to_keep_dict[i]

        return left_bin, right_bin

    def set_to_image(self, ref, coordinates, var_dict):
        """
        Works on single event. No batches
        """

        ref_eta, ref_phi = ref
        eta_ak, phi_ak, layer_ak = coordinates

        ref_eta_bin = np.digitize(ref_eta, self.eta_bin_edges_dict[5]) - 1
        ref_phi_bin = np.digitize(ref_phi, self.phi_bin_edges_dict[5]) - 1

        left_eta_bin5, right_eta_bin5 = self.get_left_right_edge_idx(ref_eta_bin, 5)
        left_phi_bin5 = ref_phi_bin - self.bins_to_keep_dict[5] // 2
        right_phi_bin5 = ref_phi_bin + self.bins_to_keep_dict[5] // 2

        # for patch midpoint calculation
        left_eta, right_eta = (
            self.eta_bin_edges_dict[5][left_eta_bin5],
            self.eta_bin_edges_dict[5][right_eta_bin5],
        )
        left_phi, right_phi = (
            self.phi_bin_edges_dict[5][left_phi_bin5],
            self.phi_bin_edges_dict[5][right_phi_bin5],
        )

        eta_np = ak.to_numpy(eta_ak)
        phi_np = ak.to_numpy(phi_ak)
        layer_np = ak.to_numpy(layer_ak)

        im_ind_dict = {}
        etaphi_lbirbi_dict = {}
        patch_mids_flattened = []
        feat_dict = {f"feat0_{i}": [] for i in range(self.n_layers)}

        for li in range(self.n_layers):
            mask = layer_np == li
            eta_li = eta_np[mask]
            phi_li = phi_np[mask]

            # phi from [-π, π] to [-2π, 2π]
            delphi = phi_li - ref_phi
            phi_li[delphi > np.pi] = phi_li[delphi > np.pi] - 2 * np.pi
            phi_li[delphi < -np.pi] = phi_li[delphi < -np.pi] + 2 * np.pi

            eta_lbi = int(self.grans[li] / self.grans[5] * left_eta_bin5)
            eta_rbi = int(self.grans[li] / self.grans[5] * right_eta_bin5) + 1
            eta_bins = self.eta_bin_edges_dict[li][eta_lbi:eta_rbi]

            phi_lbi = int(self.grans[li] / self.grans[-1] * left_phi_bin5)
            phi_rbi = int(self.grans[li] / self.grans[-1] * right_phi_bin5) + 1
            phi_bins = self.phi_bin_edges_dict[li][phi_lbi:phi_rbi]

            # indicator
            im_ind = np.histogram2d(eta_li, phi_li, bins=[eta_bins, phi_bins], density=False)[0] > 0
            im_ind_dict[f"ind_{li}"] = torch.from_numpy(im_ind).float()

            # variables
            for vn, v in var_dict.items():
                im_v, _, _ = np.histogram2d(
                    eta_li,
                    phi_li,
                    bins=[eta_bins, phi_bins],
                    density=False,
                    weights=ak.to_numpy(v[mask]),
                )
                feat_dict[f"feat0_{li}"].append(torch.from_numpy(im_v).float())

            # storing the bin edges for later use
            etaphi_lbirbi_dict[li] = torch.tensor([[eta_lbi, eta_rbi], [phi_lbi, phi_rbi]])

            # patch midpoint calculation
            n_patches = self.num_patches_per_layer_1d[li]
            eta_patch_size = (right_eta - left_eta) / n_patches
            phi_patch_size = (right_phi - left_phi) / n_patches
            eta_patch_mids = torch.arange(left_eta, right_eta, eta_patch_size) + eta_patch_size / 2
            phi_patch_mids = torch.arange(left_phi, right_phi, phi_patch_size) + phi_patch_size / 2

            eta_grid, phi_grid = torch.meshgrid(eta_patch_mids, phi_patch_mids, indexing="ij")
            patch_mids_flattened.append(
                torch.stack(
                    [
                        eta_grid.reshape(-1),
                        torch.cos(phi_grid).reshape(-1),
                        torch.sin(phi_grid).reshape(-1),
                    ],
                    dim=-1,
                )
            )

        patch_mids_flattened = torch.cat(patch_mids_flattened, dim=0)

        # stack the continuous features
        for li in range(self.n_layers):
            feat_dict[f"feat0_{li}"] = torch.stack(feat_dict[f"feat0_{li}"], axis=-1)

        return feat_dict, im_ind_dict, etaphi_lbirbi_dict, patch_mids_flattened

    def image_to_set(self, feat_dict, ind_logit_dict, etaphi_lbirbi_dict, ind_logit_th=0.0):
        """
        Works on single event
        """
        feat_tensor = []
        for i in range(self.n_layers):
            ind_mask = ind_logit_dict[f"ind_{i}"] > ind_logit_th

            # set of features
            feat_tensor.append(feat_dict[f"feat0_{i}"][ind_mask])

            # eta, phi, layer
            eta_lbi, eta_rbi = etaphi_lbirbi_dict[i][0]
            phi_lbi, phi_rbi = etaphi_lbirbi_dict[i][1]

            eta_idxs, phi_idxs = torch.where(ind_mask)
            eta = self.eta_bin_mids_dict[i][eta_idxs + eta_lbi]
            phi = self.phi_bin_mids_dict[i][phi_idxs + phi_lbi]
            phi = (phi + np.pi) % (2 * np.pi) - np.pi  # back to [-π, π]
            layer = torch.full((len(eta),), i, dtype=torch.int64, device=eta.device)

            feat_tensor[-1] = torch.cat(
                [
                    feat_tensor[-1],
                    eta.unsqueeze(1),
                    phi.unsqueeze(1),
                    layer.unsqueeze(1),
                    ind_logit_dict[f"ind_{i}"][ind_mask].unsqueeze(1),
                ],
                dim=1,
            )

        feat_tensor = torch.cat(feat_tensor, dim=0)
        return feat_tensor
