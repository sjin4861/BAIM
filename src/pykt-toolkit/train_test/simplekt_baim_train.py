import argparse
from wandb_train import main

"""
SimpleKT-BAIM Training Script (State-Aware MoE + SimpleKT)

Matched Defaults with original SimpleKT:
- d_model: 256
- n_blocks: 2
- num_attn_heads: 4
- learning_rate: 1e-3
- batch_size: 256
- final_fc_dim: 256

Key Features:
1. State-Aware Routing: Parallel GRU tracks student state to guide MoE router.
2. Smart Proxy: Router and State Tracker share the same learnable summary.
3. Early Fusion: Polya-fused question embedding is fed into SimpleKT Transformer.
4. Load Balancing: Ensures experts are used evenly.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 1. Basic Configs
    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--model_name", type=str, default="simplekt_baim")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model_simplekt_baim")

    # 2. Training Configs (Aligned with SimpleKT Defaults)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument(
        "--learning_rate", type=float, default=1e-3
    )  # [변경] 1e-4 -> 1e-3
    parser.add_argument("--batch_size", type=int, default=256)  # [변경] 128 -> 256
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--weighted_loss", type=int, default=0)
    parser.add_argument("--train_subset_rate", type=float, default=1.0)

    # 3. Embedding Configs (BAIM-specific)
    parser.add_argument(
        "--emb_path", type=str, default="", help="Path to BAIM embedding file (.pt)"
    )
    parser.add_argument(
        "--flag_load_emb", action="store_true", help="Load pretrained embeddings"
    )
    parser.add_argument(
        "--flag_emb_freezed", action="store_true", help="Freeze pretrained embeddings"
    )
    parser.add_argument(
        "--pretrain_dim",
        type=int,
        default=768,
        help="Dimension of pretrained embeddings",
    )

    # 4. SimpleKT Specific Configs
    # Transformer structure
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument(
        "--n_blocks", type=int, default=2, help="Number of transformer blocks"
    )  # [변경] 1 -> 2
    parser.add_argument(
        "--num_attn_heads", type=int, default=4, help="Number of attention heads"
    )  # [변경] 8 -> 4
    parser.add_argument("--d_ff", type=int, default=256)

    # FC Layer dims
    parser.add_argument("--final_fc_dim", type=int, default=256)  # [변경] 512 -> 256
    parser.add_argument("--final_fc_dim2", type=int, default=256)

    # SimpleKT Misc (Compatibility args)
    parser.add_argument("--num_layers", type=int, default=2)  # [변경] 1 -> 2
    parser.add_argument("--nheads", type=int, default=4)  # [변경] 8 -> 4
    parser.add_argument("--loss1", type=float, default=0.5)
    parser.add_argument("--loss2", type=float, default=0.5)
    parser.add_argument("--loss3", type=float, default=0.5)
    parser.add_argument("--start", type=int, default=50)

    # Polya needs these, kept defaults or added if missing in simplekt_que but required by logic
    parser.add_argument("--kq_same", type=int, default=1)
    parser.add_argument("--separate_qa", type=int, default=0)  # 0: False, 1: True
    parser.add_argument("--l2", type=float, default=1e-5)

    # 5. WandB Configs
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="",
        help="if not empty string, it will overwrite the default wandb project name",
    )

    args = parser.parse_args()

    params = vars(args)
    main(params)
