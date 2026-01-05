import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--model_name", type=str, default="qdkt_baim")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument(
        "--emb_path",
        type=str,
        default="",
        help="Path to 4-stage BAIM embeddings (num_q, 4, 768)",
    )
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--train_subset_rate", type=float, default=1.0)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--emb_size", type=int, default=256, help="Model embedding dimension"
    )
    parser.add_argument(
        "--pretrain_dim",
        type=int,
        default=768,
        help="Pretrained embedding dimension (e.g., from 4-stage BAIM)",
    )
    parser.add_argument("--weighted_loss", type=int, default=0)

    parser.add_argument(
        "--flag_load_emb",
        action="store_true",
        help="Load pre-trained 4-stage BAIM embeddings",
    )
    parser.add_argument(
        "--flag_emb_freezed",
        action="store_true",
        help="Freeze loaded embeddings during training",
    )

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="",
        help="if not empty string, it will overwrite the default wandb project name",
    )
    parser.add_argument("--add_uuid", type=int, default=1)

    args = parser.parse_args()

    params = vars(args)
    main(params)
