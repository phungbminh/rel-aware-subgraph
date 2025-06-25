import torch
import argparse
import numpy as np
import random
import os
from model.RASG import RASGModel
from extraction.datasets import SubgraphDataset
from trainer.trainer import run_training

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser(description="RASG: Relation-aware Subgraph Extraction for Link Prediction")
    parser.add_argument("--db-path-train", type=str, required=True)
    parser.add_argument("--db-path-valid", type=str, default=None)
    parser.add_argument("--db-path-test", type=str, default=None)
    parser.add_argument("--db-name-pos", type=str, default="positive")
    parser.add_argument("--db-name-neg", type=str, default="negative")
    parser.add_argument("--raw-data-paths", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--relation-emb-dim", type=int, default=32)
    parser.add_argument("--node-label-emb-dim", type=int, default=16)
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--att-dim", type=int, default=64)
    parser.add_argument("--out-dim", type=int, default=1)
    parser.add_argument("--max-dist", type=int, default=10)
    parser.add_argument("--num-rels", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)

    # Tự động detect số relations nếu chưa chỉ định
    num_rels = 51


    # === Init Dataset cho từng split ===
    def make_dataset(db_path):
        return SubgraphDataset(
            db_path=db_path,
            db_name_pos=args.db_name_pos,
            db_name_neg=args.db_name_neg,
            raw_data_paths=args.raw_data_paths,
            relation_emb_dim=args.relation_emb_dim,
            node_label_emb_dim=args.node_label_emb_dim,
            max_dist=args.max_dist,
        )
    train_dataset = make_dataset(args.db_path_train)
    valid_dataset = make_dataset(args.db_path_valid) if args.db_path_valid else train_dataset
    test_dataset  = make_dataset(args.db_path_test)  if args.db_path_test else train_dataset

    # === Model ===
    model = RASGModel(
        num_rels=num_rels,
        node_label_max_dist=args.max_dist,
        node_label_emb_dim=args.node_label_emb_dim,
        rel_emb_dim=args.relation_emb_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        num_layers=args.num_layers,
        att_dim=args.att_dim,
        out_dim=args.out_dim,
    )

    # === Training & Eval ===
    mrr, hits = run_training(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        lr=args.lr,
        margin=args.margin,
        patience=args.patience,
    )
    print(f"\nFinal Test MRR: {mrr:.4f} | Hits@1/3/10: {hits}")

if __name__ == "__main__":
    main()
