import argparse
from wandb_train import main

"""
Polya-SparseKT V5 Training Script (State-Aware MoE + SparseKT)

Matched Defaults with National Rules & SparseKT:
- learning_rate: 1e-4  (National Rule)
- batch_size: 128      (National Rule)
- d_model: 256         (National Rule)
- dropout: 0.1         (National Rule)
- n_blocks: 2          (SparseKT Default)
- num_attn_heads: 4    (SparseKT Default)

Key Features:
1. State-Aware Routing: Parallel GRU tracks student state.
2. Smart Proxy: Router and State Tracker share summary.
3. Sparse Attention: Efficient long-sequence modeling with Top-K.
4. Load Balancing: Prevents expert collapse.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 1. Basic Configs
    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--model_name", type=str, default="polya_sparsekt_v5")
    # 주의: Sparse Attention을 쓰려면 'qid' 대신 'qid_sparseattn'을 써야 함
    parser.add_argument("--emb_type", type=str, default="qid_sparseattn") 
    parser.add_argument("--save_dir", type=str, default="saved_model_polya_sparsekt_v5")
    
    # 2. Training Configs (National Rule Defaults)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-4) # [변경] 1e-4
    parser.add_argument("--batch_size", type=int, default=128)       # [변경] 128
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--weighted_loss", type=int, default=0)
    parser.add_argument("--train_subset_rate", type=float, default=1.0)

    # 3. Embedding Configs (Polya)
    parser.add_argument("--emb_path", type=str, 
                        default="",
                        help="Path to Polya embedding file (.pt)")
    parser.add_argument('--flag_load_emb', action='store_true', 
                        help="Load pretrained embeddings")
    parser.add_argument('--flag_emb_freezed', action='store_true',
                        help="Freeze pretrained embeddings")
    parser.add_argument("--pretrain_dim", type=int, default=768,
                        help="Dimension of pretrained embeddings")

    # 4. SparseKT Specific Configs
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    
    # Sparse Params
    parser.add_argument("--sparse_ratio", type=float, default=0.8)
    parser.add_argument("--k_index", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    
    # FC & Misc
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

    # 5. WandB Configs
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument("--wandb_project_name", type=str, default="", 
                        help="if not empty string, it will overwrite the default wandb project name")
    
    args = parser.parse_args()

    params = vars(args)
    main(params)