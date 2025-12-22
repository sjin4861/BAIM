import argparse
from wandb_train import main

"""
Polya-AKT V5 Training Script (State-Aware MoE + Parallel AKT)

Matched Defaults with original AKT:
- dropout: 0.2
- d_ff: 512
- n_blocks: 4
- learning_rate: 1e-4

Key Improvements over V4:
1. State-Aware Routing: Uses Parallel GRU to track student state.
2. Smart Proxy Sharing: Router and State Tracker share the same learnable summary.
3. Load Balancing: Adds auxiliary loss to prevent expert collapse.
4. Dual Projection: Separates correct/incorrect embedding spaces like qDKT.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 1. Basic Configs
    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--model_name", type=str, default="polya_akt_v5") # V5 Target
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model_polya_akt_v5")
    
    # 2. Training Configs
    parser.add_argument("--seed", type=int, default=42) # AKT 원본은 3407이지만, V5 실험 통일성을 위해 42 유지
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2) # [변경] 0.1 -> 0.2
    
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256) # AKT는 d_model 사용
    parser.add_argument("--learning_rate", type=float, default=1e-4) 
    parser.add_argument("--batch_size", type=int, default=128) # 원본 32지만, V5 실험 위해 128 시도
    parser.add_argument("--num_epochs", type=int, default=200)

    # 3. Embedding Configs (Polya-specific)
    parser.add_argument("--emb_path", type=str, 
                        default="",
                        help="Path to Polya embedding file (.pt)")
    parser.add_argument('--flag_load_emb', action='store_true', 
                        help="Load pretrained embeddings")
    parser.add_argument('--flag_emb_freezed', action='store_true',
                        help="Freeze pretrained embeddings")
    parser.add_argument("--pretrain_dim", type=int, default=768,
                        help="Dimension of pretrained embeddings")

    # 4. AKT-specific Configs
    parser.add_argument("--n_blocks", type=int, default=4, # [변경] 2 -> 4
                        help="Number of AKT encoder blocks")
    parser.add_argument("--num_attn_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=512, # [변경] 256 -> 512
                        help="Feed-forward dimension")
    parser.add_argument("--final_fc_dim", type=int, default=512,
                        help="Final FC layer dimension")
    parser.add_argument("--kq_same", type=int, default=1,
                        help="Whether key and query projections are the same")
    parser.add_argument("--l2", type=float, default=1e-5,
                        help="L2 regularization")
    parser.add_argument("--separate_qa", action='store_true', default=False) # AKTQue에 있는 옵션

    # 5. WandB Configs
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument("--wandb_project_name", type=str, default="", 
                        help="if not empty string, it will overwrite the default wandb project name")
    
    # 6. Additional Configs
    parser.add_argument("--weighted_loss", type=int, default=0)
    parser.add_argument("--train_subset_rate", type=float, default=1.0)
    
    args = parser.parse_args()

    params = vars(args)
    main(params)