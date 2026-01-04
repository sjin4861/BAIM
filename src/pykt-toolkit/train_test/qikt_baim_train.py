import argparse
from wandb_train import main

"""
QIKT-BAIM Training Script (State-Aware MoE + QIKT)

Key Features:
1. State-Aware Routing: Parallel GRU tracks student state to guide MoE router.
2. Smart Proxy Content: QIKT's Question Embedding is replaced by BAIM-fused embedding.
3. Load Balancing: Ensures experts are used evenly.
4. Multi-Objective Loss: QIKT's original loss structure (Question/Concept All/Next) is preserved.

Example Usage:
python qikt_baim_train.py \
  --dataset_name xes3g5m \
  --model_name qikt_baim \
  --emb_type qid \
  --emb_path /path/to/baim_tensor.pt \
  --flag_load_emb \
  --emb_size 256 \
  --batch_size 128 \
  --use_wandb 1
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 1. Basic Configs
    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--model_name", type=str, default="qikt_baim")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model_qikt_baim")

    # 2. Training Configs (National Rule + QIKT Defaults)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)  # [변경] 3407 -> 42
    parser.add_argument("--dropout", type=float, default=0.1)  # [변경] 0.4 -> 0.1
    parser.add_argument("--learning_rate", type=float, default=1e-3)  # QIKT uses 1e-3
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--weighted_loss", type=int, default=0)
    parser.add_argument("--train_subset_rate", type=float, default=1.0)

    # 3. Embedding Configs (BAIM)
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

    # 4. QIKT Specific Configs
    parser.add_argument("--emb_size", type=int, default=256)  # [변경] 300 -> 256
    parser.add_argument("--mlp_layer_num", type=int, default=1)  # [변경] 2 -> 1

    # QIKT Loss Lambdas (Original Defaults)
    parser.add_argument("--loss_q_all_lambda", type=float, default=0)
    parser.add_argument("--loss_c_all_lambda", type=float, default=0)
    parser.add_argument("--loss_q_next_lambda", type=float, default=0)
    parser.add_argument("--loss_c_next_lambda", type=float, default=0)

    # QIKT Output Lambdas (Original Defaults)
    parser.add_argument("--output_q_all_lambda", type=float, default=1)
    parser.add_argument("--output_c_all_lambda", type=float, default=1)
    parser.add_argument("--output_q_next_lambda", type=float, default=0)
    parser.add_argument("--output_c_next_lambda", type=float, default=1)

    parser.add_argument("--output_mode", type=str, default="an")  # an or an_irt

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

    # [QIKT Logic] Move specific configs to 'other_config'
    remove_keys = ["output_mode"] + [x for x in params.keys() if "lambda" in x]
    other_config = {}
    for k in remove_keys:
        other_config[k] = params[k]
        del params[k]
    params["other_config"] = other_config

    main(params)
