import argparse
from wandb_train import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--model_name", type=str, default="qikt_baim")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model_qikt_baim")

    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
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

    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--mlp_layer_num", type=int, default=1)

    parser.add_argument("--loss_q_all_lambda", type=float, default=0)
    parser.add_argument("--loss_c_all_lambda", type=float, default=0)
    parser.add_argument("--loss_q_next_lambda", type=float, default=0)
    parser.add_argument("--loss_c_next_lambda", type=float, default=0)

    parser.add_argument("--output_q_all_lambda", type=float, default=1)
    parser.add_argument("--output_c_all_lambda", type=float, default=1)
    parser.add_argument("--output_q_next_lambda", type=float, default=0)
    parser.add_argument("--output_c_next_lambda", type=float, default=1)

    parser.add_argument("--output_mode", type=str, default="an")

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

    remove_keys = ["output_mode"] + [x for x in params.keys() if "lambda" in x]
    other_config = {}
    for k in remove_keys:
        other_config[k] = params[k]
        del params[k]
    params["other_config"] = other_config
    main(params)
