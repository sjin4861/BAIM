import torch
import numpy as np
import os

from .qdkt_baim import QDKTBAIM
from .akt_baim import AKTBAIM
from .simplekt_baim import SimpleKTBAIM
from .qikt_baim import QIKTBAIM
from .sparsekt_baim import SparseKTBAIM


def get_device():
    if torch.backends.mps.is_available():  # Check for Apple Silicon GPU support
        return torch.device("mps")
    elif torch.cuda.is_available():  # Check for CUDA GPU support
        return torch.device("cuda")
    else:  # Fallback to CPU if neither MPS nor CUDA is available
        return torch.device("cpu")


device = get_device()
# device = torch.device("cpu")


def init_model(model_name, model_config, data_config, emb_type):
    if model_name == "qdkt_baim":
        model = QDKTBAIM(
            num_q=data_config["num_q"],
            **model_config,
            emb_type=emb_type,
            emb_path=data_config["emb_path"],
            device=device,
        ).to(device)
    elif model_name == "akt_baim":
        model = AKTBAIM(
            num_q=data_config["num_q"],
            emb_size=model_config.get("emb_size", 256),
            n_blocks=model_config.get("n_blocks", 1),
            dropout=model_config.get("dropout", 0.1),
            d_ff=model_config.get("d_ff", 256),
            kq_same=model_config.get("kq_same", 1),
            final_fc_dim=model_config.get("final_fc_dim", 512),
            num_attn_heads=model_config.get("num_attn_heads", 8),
            separate_qa=model_config.get("separate_qa", False),
            l2=model_config.get("l2", 1e-5),
            # Polya config parameters
            emb_type=emb_type,
            emb_path=data_config["emb_path"],
            flag_load_emb=model_config.get("flag_load_emb", False),
            flag_emb_freezed=model_config.get("flag_emb_freezed", False),
            pretrain_dim=model_config.get("pretrain_dim", 768),
            device=device,
        ).to(device)
    elif model_name == "simplekt_baim":
        model = SimpleKTBAIM(
            n_question=data_config["num_q"],
            n_pid=0,
            # SimpleKT specific params
            d_model=model_config.get("d_model", 256),
            n_blocks=model_config.get("n_blocks", 1),
            dropout=model_config.get("dropout", 0.1),
            d_ff=model_config.get("d_ff", 256),
            num_attn_heads=model_config.get("num_attn_heads", 8),
            kq_same=model_config.get("kq_same", 1),
            final_fc_dim=model_config.get("final_fc_dim", 512),
            final_fc_dim2=model_config.get("final_fc_dim2", 256),
            seq_len=model_config.get("seq_len", 200),
            separate_qa=model_config.get("separate_qa", False),
            l2=model_config.get("l2", 1e-5),
            # Polya params
            emb_type=emb_type,
            emb_path=data_config["emb_path"],
            flag_load_emb=model_config.get("flag_load_emb", False),
            flag_emb_freezed=model_config.get("flag_emb_freezed", False),
            pretrain_dim=model_config.get("pretrain_dim", 768),
            device=device,
        ).to(device)
    elif model_name == "qikt_baim":
        model = QIKTBAIM(
            num_q=data_config["num_q"],
            num_c=data_config["num_c"],
            # QIKT specific params
            emb_size=model_config.get("emb_size", 256),
            dropout=model_config.get("dropout", 0.1),
            mlp_layer_num=model_config.get("mlp_layer_num", 1),
            other_config=model_config.get("other_config", {}),
            # Polya params
            emb_type=emb_type,
            emb_path=data_config["emb_path"],
            flag_load_emb=model_config.get("flag_load_emb", False),
            flag_emb_freezed=model_config.get("flag_emb_freezed", False),
            pretrain_dim=model_config.get("pretrain_dim", 768),
            device=device,
        ).to(device)
    elif model_name == "sparsekt_baim":
        model = SparseKTBAIM(
            n_question=data_config["num_q"],
            n_pid=0,
            # SparseKT specific params``
            d_model=model_config.get("d_model", 256),
            n_blocks=model_config.get("n_blocks", 2),
            dropout=model_config.get("dropout", 0.1),
            d_ff=model_config.get("d_ff", 256),
            num_attn_heads=model_config.get("num_attn_heads", 4),
            kq_same=model_config.get("kq_same", 1),
            final_fc_dim=model_config.get("final_fc_dim", 256),
            final_fc_dim2=model_config.get("final_fc_dim2", 256),
            seq_len=model_config.get("seq_len", 200),
            separate_qa=model_config.get("separate_qa", False),
            l2=model_config.get("l2", 1e-5),
            # Sparse Attention Params
            sparse_ratio=model_config.get("sparse_ratio", 0.8),
            k_index=model_config.get("k_index", 5),
            stride=model_config.get("stride", 1),
            # Polya params
            emb_type=emb_type,
            emb_path=data_config["emb_path"],
            flag_load_emb=model_config.get("flag_load_emb", False),
            flag_emb_freezed=model_config.get("flag_emb_freezed", False),
            pretrain_dim=model_config.get("pretrain_dim", 768),
            device=device,
        ).to(device)
    else:
        print("The wrong model name was used...")
        return None
    return model


def load_model(model_name, model_config, data_config, emb_type, ckpt_path):
    model = init_model(model_name, model_config, data_config, emb_type)

    ckpt_file = os.path.join(ckpt_path, emb_type + "_model.ckpt")

    # Try loading with default encoding first, fallback to latin1 if UTF-8 fails
    try:
        net = torch.load(ckpt_file)
    except UnicodeDecodeError:
        print(f"Warning: UTF-8 decode error, loading with latin1 encoding...")
        net = torch.load(ckpt_file, encoding="latin1")

    model.load_state_dict(net)
    return model
