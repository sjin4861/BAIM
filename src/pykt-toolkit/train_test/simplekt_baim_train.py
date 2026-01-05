import argparse
from wandb_train import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--model_name", type=str, default="simplekt_baim")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model_simplekt_baim")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--weighted_loss", type=int, default=0)
    parser.add_argument("--train_subset_rate", type=float, default=1.0)

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

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument(
        "--n_blocks", type=int, default=2, help="Number of transformer blocks"
    )
    parser.add_argument(
        "--num_attn_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--d_ff", type=int, default=256)

    parser.add_argument("--final_fc_dim", type=int, default=256)
    parser.add_argument("--final_fc_dim2", type=int, default=256)

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--loss1", type=float, default=0.5)
    parser.add_argument("--loss2", type=float, default=0.5)
    parser.add_argument("--loss3", type=float, default=0.5)
    parser.add_argument("--start", type=int, default=50)
    parser.add_argument("--kq_same", type=int, default=1)
    parser.add_argument("--separate_qa", type=int, default=0)
    parser.add_argument("--l2", type=float, default=1e-5)

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
