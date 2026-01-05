import torch
from torch import nn
from torch.nn import Module, Embedding, Linear, Dropout, MaxPool1d, Sequential, ReLU
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import os


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


device = get_device()


class transformer_FFN(Module):
    def __init__(self, emb_size, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = Sequential(
            Linear(self.emb_size, self.emb_size),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.emb_size, self.emb_size),
        )

    def forward(self, in_fea):
        return self.FFN(in_fea)


def ut_mask(seq_len):
    return (
        torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        .to(dtype=torch.bool)
        .to(device)
    )


def lt_mask(seq_len):
    return (
        torch.tril(torch.ones(seq_len, seq_len), diagonal=-1)
        .to(dtype=torch.bool)
        .to(device)
    )


def pos_encode(seq_len):
    return torch.arange(seq_len).unsqueeze(0).to(device)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def resolve_path(path):
    if not path:
        return None
    if os.path.exists(path):
        return path
    current_check = path
    for _ in range(5):
        current_check = os.path.join("..", current_check)
        if os.path.exists(current_check):
            print(f"[PolyaQDKT] Path resolved: {path} -> {current_check}")
            return current_check
    return path
