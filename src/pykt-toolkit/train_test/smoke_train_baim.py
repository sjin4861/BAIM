#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from typing import Dict

import torch


@dataclass
class DummyBatch:
    qseqs: torch.Tensor
    cseqs: torch.Tensor
    rseqs: torch.Tensor
    tseqs: torch.Tensor
    shft_qseqs: torch.Tensor
    shft_cseqs: torch.Tensor
    shft_rseqs: torch.Tensor
    shft_tseqs: torch.Tensor
    masks: torch.Tensor
    smasks: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "qseqs": self.qseqs,
            "cseqs": self.cseqs,
            "rseqs": self.rseqs,
            "tseqs": self.tseqs,
            "shft_qseqs": self.shft_qseqs,
            "shft_cseqs": self.shft_cseqs,
            "shft_rseqs": self.shft_rseqs,
            "shft_tseqs": self.shft_tseqs,
            "masks": self.masks,
            "smasks": self.smasks,
        }


def make_dummy_batch(
    batch_size: int,
    seq_len: int,
    num_q: int,
    num_c: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    # q in [0, num_q)
    q = torch.randint(low=0, high=max(num_q, 2), size=(batch_size, seq_len), device=device)
    # c in [0, num_c) (concepts); if num_c==0, fill zeros
    if num_c > 0:
        c = torch.randint(low=0, high=num_c, size=(batch_size, seq_len), device=device)
    else:
        c = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

    r = torch.randint(low=0, high=2, size=(batch_size, seq_len), device=device)
    t = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # shift by one (teacher forcing style)
    qshft = q[:, 1:]
    cshft = c[:, 1:]
    rshft = r[:, 1:]
    tshft = t[:, 1:]

    qseqs = q[:, :-1]
    cseqs = c[:, :-1]
    rseqs = r[:, :-1]
    tseqs = t[:, :-1]

    # masks over the (seq_len-1) positions
    masks = torch.ones((batch_size, seq_len - 1), dtype=torch.bool, device=device)
    smasks = masks.clone()

    return DummyBatch(
        qseqs=qseqs,
        cseqs=cseqs,
        rseqs=rseqs,
        tseqs=tseqs,
        shft_qseqs=qshft,
        shft_cseqs=cshft,
        shft_rseqs=rshft,
        shft_tseqs=tshft,
        masks=masks,
        smasks=smasks,
    ).as_dict()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-train BAIM models with synthetic data.")
    p.add_argument(
        "--model",
        type=str,
        default="qdkt_baim",
        choices=["qdkt_baim", "akt_baim", "qikt_baim"],
        help="Model to smoke-train.",
    )
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"]) 
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure imports work when run from anywhere.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.abspath(os.path.join(this_dir, ".."))
    if pkg_root not in os.sys.path:
        os.sys.path.insert(0, pkg_root)

    from pykt.models.qdkt_baim import QDKTBAIM
    from pykt.models.akt_baim import AKTBAIM
    from pykt.models.qikt_baim import QIKTBAIM

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Small synthetic vocab sizes for smoke.
    num_q = 50
    num_c = 20  # used by qikt_baim

    if args.model == "qdkt_baim":
        model = QDKTBAIM(num_q=num_q, emb_size=64, dropout=0.1, device=device, seed=0)
    elif args.model == "akt_baim":
        model = AKTBAIM(num_q=num_q, emb_size=64, n_blocks=1, dropout=0.1, num_attn_heads=4, d_ff=128, device=device, seed=0)
    elif args.model == "qikt_baim":
        model = QIKTBAIM(num_q=num_q, num_c=num_c, emb_size=64, dropout=0.1, mlp_layer_num=1, other_config={}, device=device, seed=0)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    model.model.train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[SMOKE] model={args.model} device={device} batch_size={args.batch_size} seq_len={args.seq_len} steps={args.steps}")

    for step in range(1, args.steps + 1):
        batch = make_dummy_batch(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_q=num_q,
            num_c=num_c,
            device=device,
        )

        model.opt = opt
        y, loss = model.train_one_step(batch, process=True, weighted_loss=0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            loss_val = float(loss.detach().cpu())
            y_mean = float(y.detach().mean().cpu())
        print(f"  step={step} loss={loss_val:.6f} y_mean={y_mean:.6f} y_shape={tuple(y.shape)}")

    print("[DONE] Smoke training completed.")


if __name__ == "__main__":
    main()
