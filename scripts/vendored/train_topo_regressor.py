#!/usr/bin/env python
"""Train a small transformer regressor on exported per-cluster tokens (Step 1:
scripts/export_topo_tokens.py) to predict jet pT -- Step 2 of the downstream
jet-pT regression ladder.

Identical training recipe across every config (only --train_npz/--val_npz,
and hence the input dimensionality, change): a heptokens.models.transformer
.SetToVectorTransformer backbone (mean/CLS-pooled small transformer encoder,
single-output regression head) trained with Adam + warmup/cosine schedule,
MSE loss on standardized log(pT), early stopping on val loss, fixed seed.

Usage:
    python scripts/train_topo_regressor.py \\
        --train_npz results/topo_token_regression/tokens/cb4096_cd8_nq4/train.npz \\
        --val_npz   results/topo_token_regression/tokens/cb4096_cd8_nq4/val.npz \\
        --out_dir   results/topo_token_regression/regressors/cb4096_cd8_nq4

Capacity-check variant (2-4x larger model, otherwise identical):
    python scripts/train_topo_regressor.py ... --d_model 256 --num_layers 6
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from heptokens.vendored.models.transformer import SetToVectorTransformer
from heptokens.vendored.models.utils import warmup_cosine_scheduler

log = logging.getLogger(__name__)


def load_split(npz_path):
    d = np.load(npz_path)
    return d["z"], d["mask"], d["truth_pt"]


def make_loader(z, mask, y, batch_size, shuffle, seed):
    ds = TensorDataset(
        torch.from_numpy(z).float(),
        torch.from_numpy(mask).bool(),
        torch.from_numpy(y).float(),
    )
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen, drop_last=shuffle)


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    losses = []
    for z_b, mask_b, y_b in loader:
        z_b, mask_b, y_b = z_b.to(device), mask_b.to(device), y_b.to(device)
        with torch.set_grad_enabled(train):
            pred = model(z_b, mask_b).reshape(-1)
            loss = torch.nn.functional.mse_loss(pred, y_b)
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pooling", default="mean", choices=["mean", "sum", "max", "cls"])
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading train split: {args.train_npz}")
    z_tr, mask_tr, pt_tr = load_split(args.train_npz)
    log.info(f"Loading val split: {args.val_npz}")
    z_va, mask_va, pt_va = load_split(args.val_npz)

    # Drop events with truth_pt <= 0 (no valid clusters in the event -- a
    # handful always exist, e.g. 17/1e6 in this dataset). log(pT) on these
    # is -inf/NaN and would otherwise poison target_mean/target_std (computed
    # over the WHOLE array) and hence every event's standardized target,
    # causing immediate NaN loss from epoch 0 -- matches eval_topo_regressor
    # .py's existing `truth_pt > 0` filter, just missing here before.
    valid_tr = pt_tr > 0
    valid_va = pt_va > 0
    n_dropped_tr = int((~valid_tr).sum())
    n_dropped_va = int((~valid_va).sum())
    if n_dropped_tr or n_dropped_va:
        log.warning(f"Dropping {n_dropped_tr:,} train / {n_dropped_va:,} val events with truth_pt <= 0")
    z_tr, mask_tr, pt_tr = z_tr[valid_tr], mask_tr[valid_tr], pt_tr[valid_tr]
    z_va, mask_va, pt_va = z_va[valid_va], mask_va[valid_va], pt_va[valid_va]
    log.info(f"train: {z_tr.shape}, val: {z_va.shape}")

    input_dim = z_tr.shape[-1]

    # Standardize log(pT) using TRAIN statistics only; predictions are
    # un-standardized back to physical pT at eval time (Step 3).
    log_pt_tr = np.log(pt_tr.astype(np.float64))
    target_mean = float(log_pt_tr.mean())
    target_std = float(log_pt_tr.std())
    log.info(f"log(pT) train stats: mean={target_mean:.4f} std={target_std:.4f}")

    y_tr = ((log_pt_tr - target_mean) / target_std).astype(np.float32)
    y_va = ((np.log(pt_va.astype(np.float64)) - target_mean) / target_std).astype(np.float32)

    train_loader = make_loader(z_tr, mask_tr, y_tr, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(z_va, mask_va, y_va, args.batch_size, shuffle=False, seed=args.seed)

    model = SetToVectorTransformer(
        input_dim=input_dim,
        output_dim=1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="gelu",
        pooling=args.pooling,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model has {n_params:,} parameters (d_model={args.d_model}, num_layers={args.num_layers})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = warmup_cosine_scheduler(
        optimizer, warmup_epochs=args.warmup_epochs, min_lr=args.min_lr, max_epochs=args.max_epochs
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_best = 0
    history = []

    for epoch in range(args.max_epochs):
        train_loss = run_epoch(model, train_loader, args.device, optimizer)
        val_loss = run_epoch(model, val_loader, args.device, optimizer=None)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        log.info(f"epoch {epoch:3d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  lr={lr_now:.2e}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr_now})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                log.info(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, val_loss={best_val_loss:.5f})")
                break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    meta = {
        "input_dim": input_dim,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "pooling": args.pooling,
        "n_params": n_params,
        "target_mean": target_mean,
        "target_std": target_std,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "seed": args.seed,
        "train_npz": str(args.train_npz),
        "val_npz": str(args.val_npz),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Saved best checkpoint + meta.json to {out_dir}")


if __name__ == "__main__":
    main()
