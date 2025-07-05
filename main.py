import torch
import argparse
import numpy as np
import random
import os
import pickle
import json
import time
from model.model import RASG
from extraction.datasets import SubGraphDataset
from trainer.trainer import run_training
from utils import CSRGraph

def set_seed(seed):
    """Cài đặt seed toàn bộ cho reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_mapping(mapping_dir):
    """Tải các ánh xạ từ thư mục"""
    mapping_files = {
        'entity2id': 'entity2id.pkl',
        'id2entity': 'id2entity.pkl',
        'relation2id': 'relation2id.pkl',
        'id2relation': 'id2relation.pkl',
        'metadata': 'metadata.json'
    }
    mappings = {}
    for name, file in mapping_files.items():
        path = os.path.join(mapping_dir, file)
        if os.path.exists(path):
            with open(path, 'rb') if file.endswith('.pkl') else open(path, 'r') as f:
                mappings[name] = pickle.load(f) if file.endswith('.pkl') else json.load(f)
        else:
            raise FileNotFoundError(f"Cannot find mapping file: {path}")
    return mappings

def load_global_graph(graph_path):
    """Tải đồ thị toàn cục từ file pickle"""
    with open(graph_path, 'rb') as f:
        return pickle.load(f)

def get_args():
    """Định nghĩa argument parser cho pipeline RASG"""
    parser = argparse.ArgumentParser(
        description="RASG: Relation-Aware Subgraph Learning for Knowledge Graphs"
    )

    # ========== Path Arguments ==========
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory containing all datasets")
    parser.add_argument("--mapping-dir", type=str, default="mappings",
                        help="Directory containing entity/relation mappings")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for logs and models")

    # ========== Dataset Arguments ==========
    parser.add_argument("--train-db", type=str, default="train.lmdb",
                        help="LMDB path for training data")
    parser.add_argument("--valid-db", type=str, default="valid.lmdb",
                        help="LMDB path for validation data")
    parser.add_argument("--test-db", type=str, default="test.lmdb",
                        help="LMDB path for test data")
    parser.add_argument("--global-graph", type=str, default="mappings/global_graph.pkl",
                        help="Path to global CSR graph")
    parser.add_argument("--num-negatives", type=int, default=5,
                        help="Number of negative samples per positive (for evaluation)")
    parser.add_argument("--num-train-negatives", type=int, default=1,
                        help="Number of negative samples per positive during training")
    parser.add_argument("--use-full-dataset", action="store_true",
                        help="Use full dataset (auto-adjust memory settings)")

    # ========== Training Arguments ==========
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Margin for ranking loss")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="Evaluate every N epochs (999 = skip validation)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loader workers")

    # ========== Model Arguments ==========
    parser.add_argument("--node-emb-dim", type=int, default=32,
                        help="Node label embedding dimension")
    parser.add_argument("--rel-emb-dim", type=int, default=64,
                        help="Relation embedding dimension")
    parser.add_argument("--gnn-hidden", type=int, default=256,
                        help="GNN hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument("--att-dim", type=int, default=128,
                        help="Attention pooling dimension")
    parser.add_argument("--att-heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--max-dist", type=int, default=10,
                        help="Maximum distance for node labeling")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout probability")

    # ========== System Arguments ==========
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for training (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save-model", action="store_true",
                        help="Save best model checkpoint")
    parser.add_argument("--is-debug", action="store_true",
                        help="Enable debug mode for verbose output")

    return parser.parse_args()

def setup_directories(args):
    """Tạo các thư mục đầu ra nếu cần"""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

def log_args(args, logger=None):
    """Ghi lại các tham số"""
    msg = "\n===== Experiment Configuration ====="
    for arg, value in vars(args).items():
        msg += f"\n{arg:>20}: {value}"
    msg += "\n" + "=" * 40
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

def auto_adjust_for_full_dataset(args):
    """Auto-adjust parameters for full dataset training"""
    if args.use_full_dataset:
        # Detect GPU setup
        import torch
        num_gpus = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if num_gpus > 0 else 0
        
        print(f"[INFO] Detected {num_gpus} GPUs, {gpu_memory:.1f}GB each")
        
        if num_gpus >= 2 and gpu_memory >= 15:
            # Multi-GPU setup (2x T4 16GB)
            print("[AUTO] Multi-GPU detected, using aggressive settings")
            if args.batch_size < 16:
                print(f"[AUTO] Increasing batch-size from {args.batch_size} to 16 for multi-GPU")
                args.batch_size = 16
            
            if args.num_train_negatives < 2:
                print(f"[AUTO] Increasing num-train-negatives from {args.num_train_negatives} to 2 for multi-GPU")
                args.num_train_negatives = 2
                
            # LMDB requires num_workers=0 to prevent bus errors
            if args.num_workers > 0:
                print(f"[AUTO] Setting num-workers to 0 (was {args.num_workers}) - LMDB compatibility")
                args.num_workers = 0
                
        else:
            # Single GPU setup (P100 16GB)
            print("[AUTO] Single GPU detected, using conservative settings")
            if args.batch_size > 8:
                print(f"[AUTO] Reducing batch-size from {args.batch_size} to 8 for single GPU")
                args.batch_size = 8
            
            if args.num_train_negatives > 1:
                print(f"[AUTO] Reducing num-train-negatives from {args.num_train_negatives} to 1 for single GPU")
                args.num_train_negatives = 1
                
            # LMDB requires num_workers=0 to prevent bus errors
            if args.num_workers > 0:
                print(f"[AUTO] Setting num-workers to 0 (was {args.num_workers}) - LMDB compatibility")
                args.num_workers = 0
        
        # Increase patience for full dataset
        if args.patience < 15:
            print(f"[AUTO] Increasing patience from {args.patience} to 15 for full dataset")
            args.patience = 15
            
        print("[AUTO] Applied full dataset optimizations")
    
    return args

def main():
    # ===== Khởi tạo =====
    args = get_args()
    args = auto_adjust_for_full_dataset(args)  # Auto-adjust for full dataset
    setup_directories(args)
    set_seed(args.seed)

    # Tạo logger nếu có utils.logger, nếu không thì dùng print
    try:
        from utils.logger import setup_logger
        logger = setup_logger(os.path.join(args.output_dir, "logs", "experiment.log"))
    except Exception:
        logger = None
    log_args(args, logger)

    # ===== Tải dữ liệu =====
    if logger: logger.info("Loading mappings and global graph...")
    start_time = time.time()

    # Tải ánh xạ entity/relation/metadata
    mappings = load_mapping(os.path.join(args.data_root, args.mapping_dir))
    num_entities = mappings['metadata']['num_entities']
    num_relations = mappings['metadata']['num_relations']

    # Tải đồ thị toàn cục
    global_graph = load_global_graph(os.path.join(args.data_root, args.global_graph))
    if logger: logger.info(f"Loaded mappings: {num_entities} entities, {num_relations} relations")
    if logger: logger.info(f"Global graph loaded in {time.time() - start_time:.2f}s")

    # ===== Tạo datasets =====
    if logger: logger.info("Creating datasets...")
    dataset_config = {
        'mapping_dir': os.path.join(args.data_root, args.mapping_dir),
        'global_graph': global_graph,
        'num_negatives': args.num_negatives,
        'cache_size': 20000,
        'is_debug': args.is_debug,
    }
    train_dataset = SubGraphDataset(
        db_path=os.path.join(args.data_root, args.train_db),
        split='train',
        **dataset_config
    )
    valid_dataset = SubGraphDataset(
        db_path=os.path.join(args.data_root, args.valid_db),
        split='valid',
        **dataset_config
    )
    test_dataset = SubGraphDataset(
        db_path=os.path.join(args.data_root, args.test_db),
        split='test',
        **dataset_config
    )
    if logger: logger.info(f"Dataset sizes: Train={len(train_dataset)}, Valid={len(valid_dataset)}, Test={len(test_dataset)}")

    # ===== Khởi tạo mô hình =====
    if logger: logger.info("Initializing model...")
    model = RASG(
        num_rels=num_relations,
        max_dist=args.max_dist,
        node_emb_dim=args.node_emb_dim,
        rel_emb_dim=args.rel_emb_dim,
        gnn_hidden=args.gnn_hidden,
        num_layers=args.num_layers,
        att_dim=args.att_dim,
        att_heads=args.att_heads,
        dropout=args.dropout
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if logger: logger.info(f"Model initialized with {num_params / 1e6:.2f}M parameters")

    # ===== Huấn luyện =====
    # Add LMDB safety check before training
    if any('lmdb' in str(path).lower() for path in [args.train_db, args.valid_db, args.test_db]):
        print("[SAFETY] LMDB detected - enforcing num_workers=0 to prevent bus errors")
        args.num_workers = 0
    
    if logger: logger.info("Starting training...")
    checkpoint_path = os.path.join(args.output_dir, "checkpoints", "best_model.pt") if args.save_model else "best_model.pt"

    results = run_training(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        lr=args.lr,
        margin=args.margin,
        eval_every=args.eval_every,
        patience=args.patience,
        num_workers=args.num_workers,
        checkpoint_path=checkpoint_path,
        num_train_negatives=args.num_train_negatives,
        is_debug=args.is_debug
    )

    # ===== Báo cáo kết quả =====
    if logger: logger.info("\n===== Final Results =====")
    if logger: logger.info(f"Best Validation MRR: {results['best_val_mrr']:.4f}")
    if logger: logger.info(f"Test MRR: {results['test_mrr']:.4f}")
    if logger: logger.info(f"Test Hits@1: {results['test_hits'][0]:.4f}")
    if logger: logger.info(f"Test Hits@3: {results['test_hits'][1]:.4f}")
    if logger: logger.info(f"Test Hits@10: {results['test_hits'][2]:.4f}")

    # Lưu kết quả json
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if logger: logger.info(f"Results saved to {args.output_dir}")
    else: print(f"Results saved to {args.output_dir}/results.json")

if __name__ == "__main__":
    main()
