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
    if torch.backends.mps.is_available():  # Check for Apple Silicon GPU support
        return torch.device("mps")
    elif torch.cuda.is_available():  # Check for CUDA GPU support
        return torch.device("cuda")
    else:  # Fallback to CPU if neither MPS nor CUDA is available
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
                # Dropout(self.dropout),
            )
    def forward(self, in_fea):
        return self.FFN(in_fea)

def ut_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool).to(device)

def lt_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.tril(torch.ones(seq_len,seq_len),diagonal=-1).to(dtype=torch.bool).to(device)

def pos_encode(seq_len):
    """ position Encoding
    """
    return torch.arange(seq_len).unsqueeze(0).to(device)

def get_clones(module, N):
    """ Cloning nn modules
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# --- 경로 안전하게 처리하는 헬퍼 함수 ---
def resolve_path(path):
    if not path: return None
    
    # 1. 현재 실행 위치 기준(상대 경로)으로 먼저 시도
    if os.path.exists(path):
        return path
    
    # 2. 실행 위치가 pykt-toolkit 내부일 경우를 대비해 상위 경로 탐색
    # 예: ../../../embedding/... 처럼 상위로 올라가며 찾기
    # (현재 파일 위치 기준이 아니라, 실행 CWD 기준 상위 폴더 탐색)
    current_check = path
    for _ in range(5): # 최대 5단계 상위까지 확인
        current_check = os.path.join("..", current_check)
        if os.path.exists(current_check):
            print(f"[PolyaQDKT] Path resolved: {path} -> {current_check}")
            return current_check
    
    # 3. 그래도 없으면 에러 (하지만 경로는 출력해줌)
    return path 