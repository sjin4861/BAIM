from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


def _add_repo_paths() -> None:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "src" / "pykt-toolkit"))


def main() -> None:
    _add_repo_paths()

    # Import after sys.path adjustments
    from pykt.models.akt_baim import AKTBAIM  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Minimal synthetic batch
    batch_size = 2
    seq_len = 20
    num_q = 100

    q = torch.randint(0, num_q, (batch_size, seq_len), device=device)
    c = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    r = torch.randint(0, 2, (batch_size, seq_len), device=device)

    # AKTBAIM expects data via QueBaseModel API in training normally, but we can call the net.
    model = AKTBAIM(
        num_q=num_q,
        emb_size=64,
        n_blocks=1,
        dropout=0.1,
        num_attn_heads=4,
        d_ff=128,
        device=str(device),
    ).to(device)

    net = model.model
    net.train()

    # Forward
    preds, _ = net(q, c, r)
    # preds is shifted internally (y[:, 1:]) in QueBaseModel.predict_one_step,
    # but here we call net directly so preds aligns with input sequence length.

    target = r.float()
    loss = torch.nn.functional.binary_cross_entropy(preds, target)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    print("[SMOKE] OK")
    print(" device:", device)
    print(" preds shape:", tuple(preds.shape))
    print(" loss:", float(loss.detach().cpu()))


if __name__ == "__main__":
    # Avoid tokenizers parallelism warning noise if HF is installed
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
