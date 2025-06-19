import os
import lmdb
import pickle
from extraction import generate_subgraph_datasets, SubgraphDataset
from utils import initialize_experiment, deserialize
import dgl
import torch
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
from torch.utils.data import DataLoader
from extraction.datasets import generate_subgraph_datasets, SubgraphDataset
from model import RASGModel
from torch.cuda.amp import GradScaler, autocast
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class Params:
    pass

def collate_fn(samples):
    # Unpack samples from SubgraphDataset
    pos_g_list, pos_g_labels, pos_r_labels, neg_g_list_list, neg_g_labels_list, neg_r_labels_list = zip(*samples)
    # Batch positive graphs
    pos_bg = dgl.batch(pos_g_list)
    # Flatten and batch negative graphs
    neg_graphs = [g for neg_list in neg_g_list_list for g in neg_list]
    neg_bg = dgl.batch(neg_graphs)
    # Collect relation labels (if needed)
    pos_r = torch.tensor(pos_r_labels, dtype=torch.long)
    neg_r = torch.tensor([r for neg in neg_r_labels_list for r in neg], dtype=torch.long)
    return pos_bg, pos_r, neg_bg, neg_r


def main():
    parser = argparse.ArgumentParser(description="Train GraIL++ on OGB-BioKG")
    # Paths
    parser.add_argument('--main-dir', type=str, required=True,
                        help='Root directory for OGB data')
    parser.add_argument('--db-path', type=str, required=True,
                        help='LMDB path for storing subgraphs')
    # Training hyperparams
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--rel-emb-dim', type=int, default=16)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--num-bases', type=int, default=None,
                        help='Number of bases for CompGCNConv')
    # Subgraph extraction params
    parser.add_argument('--max-links', type=int, default=10)
    parser.add_argument('--num-neg', type=int, default=1)
    parser.add_argument('--constrained-neg-prob', type=float, default=0.5)
    parser.add_argument('--hop', type=int, default=2)
    parser.add_argument('--add-transpose-rels', action='store_true',
                        help='Add reverse relations to the graph')
    # KGE options
    parser.add_argument('--use-kge', action='store_true',
                        help='Use precomputed KGE embeddings as node features')
    parser.add_argument('--kge-dataset', type=str, default='')
    parser.add_argument('--kge-model', type=str, default='')
    # Control regeneration
    parser.add_argument('--force-generate', action='store_true',
                        help='Force regeneration of subgraphs')
    parser.add_argument('--use-mixed-precision', action='store_true',
                        help='Enable mixed-precision training with torch.cuda.amp')

    parser.add_argument('--cache-dir', type=str, default='./cache',
                        help='Directory to load/save cached DGL graphs')
    parser.add_argument('--use-cache', action='store_true',
                        help='If set, sẽ load cache thay vì rebuild subgraphs')
    args = parser.parse_args()

    # 1) Generate or load subgraphs
    params = argparse.Namespace(
        main_dir=args.main_dir,
        db_path=args.db_path,
        max_links=args.max_links,
        num_neg_samples_per_link=args.num_neg,
        constrained_neg_prob=args.constrained_neg_prob,
        hop=args.hop,
        enclosing_sub_graph=True,
        max_nodes_per_hop=None,
        experiment_name="test"
    )

    # initialize_experiment(params, __file__)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(params.db_path), exist_ok=True)
    if not os.path.isdir(params.db_path):
        print("Generating subgraph datasets...")
        generate_subgraph_datasets(params)
        print("Subgraph generation completed.")

    # Mở environment (readonly) với số DB lớn nhất đủ chứa các splits
    env = lmdb.open(params.db_path, readonly=True, lock=False, max_dbs=10)

    # 2) Mở child‐DB
    db = env.open_db(b"train_pos")  # Đổi "train_pos" thành đúng tên bạn đã khai báo

    # 3) Bắt transaction và lấy cursor
    with env.begin(db=db) as txn:
        cursor = txn.cursor()
        keys = [key.decode("ascii") for key, _ in cursor]

    # 4) In ra
    print(f"Found {len(keys)} keys in 'train_pos':")
    for k in keys:
        print(k)

    # 1) Default DB entries (metadata)
    with env.begin() as txn_default:
        stat = txn_default.stat()  # stat của default database
        print(f"• default DB entries   : {stat['entries']}")

    # 2) Named DBs bạn muốn kiểm tra
    db_names = [
        'train_pos', 'train_neg',
        'valid_pos', 'valid_neg',
        'test_pos', 'test_neg',
    ]

    # 3) Với mỗi named-DB, mở transaction trên DB đó rồi gọi txn.stat()
    for name in db_names:
        try:
            dbi = env.open_db(name.encode(), create=False)
            with env.begin(db=dbi) as txn:
                stat = txn.stat()  # stat của chính named-DB
            print(f"• {name:11s} entries : {stat['entries']}")
        except lmdb.NotFoundError:
            print(f"• {name:11s} not found")

    raw_paths = {
        'mapping': './data/ogbl_biokg/mapping',
        'relations': './data/ogbl_biokg/raw/relations',
    }

    train_ds = SubgraphDataset(
        params.db_path,
        'train_pos', 'train_neg',
        raw_data_paths=raw_paths,
        included_relations=None,
        add_traspose_rels=True,
        num_neg_samples_per_link=1,
        use_kge_embeddings=None,
        dataset=None,
        kge_model=None
    )
    valid_ds = SubgraphDataset(
        params.db_path,
        'valid_pos', 'valid_neg',
        raw_data_paths=raw_paths,
        included_relations=None,
        add_traspose_rels=True,
        num_neg_samples_per_link=1,
        use_kge_embeddings=None,
        dataset=None,
        kge_model=None
    )
    print("=== Dataset sanity check ===")
    print(f"  n_feat_dim    = {train_ds.n_feat_dim!r}")
    print(f"  aug_num_rels  = {train_ds.aug_num_rels!r}")
    print(f"  num_neg/link  = {train_ds.num_neg_samples_per_link!r}")
    print(f"  num_graphs    = {len(train_ds)!r}")

    # Lấy sample đầu tiên
    pos_sub, g_label_pos, r_label_pos, neg_subs, g_labels_neg, r_labels_neg = train_ds[0]

    print("\nSample[0] returned:")
    print(f"  pos_sub type      : {type(pos_sub)}")
    print(f"  neg_subs len      : {len(neg_subs)}")

    # Các field trong pos_sub
    print("\npos_sub.ndata keys:", list(pos_sub.ndata.keys()))
    print("  feat   shape     :", tuple(pos_sub.ndata['feat'].shape))
    print("  query_rel shape  :", tuple(pos_sub.ndata['query_rel'].shape))
    print("\npos_sub.edata keys:", list(pos_sub.edata.keys()))
    print("  type   shape     :", tuple(pos_sub.edata['type'].shape))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)

    # 3) Initialize model, optimizer, loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_feat_dim = train_ds.n_feat_dim
    model = RASGModel(
        in_feat_dim=in_feat_dim,
        rel_emb_dim=args.rel_emb_dim,
        hidden_dim=args.hidden_dim,
        num_rels=train_ds.aug_num_rels,
        num_bases=args.num_bases,
        num_layers=args.num_layers
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MarginRankingLoss(margin=1.0)

    # 4) Training & validation loop
    scaler = torch.amp.GradScaler(device='cuda') if args.use_mixed_precision else None

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        total_loss = 0.0
        for pos_bg, pos_r, neg_bg, neg_r in train_loader:
            pos_bg, neg_bg = pos_bg.to(device), neg_bg.to(device)
            optimizer.zero_grad()
            if args.use_mixed_precision:
                #with autocast():
                with torch.amp.autocast('cuda'):
                    pos_scores = model(pos_bg)
                    neg_scores = model(neg_bg)
                    print(f"pos_scores{pos_scores[:3]} neg_scores{neg_scores[:3]}")
                    target = torch.ones_like(pos_scores, device=device)
                    loss = loss_fn(pos_scores, neg_scores, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                pos_scores = model(pos_bg)
                neg_scores = model(neg_bg)
                print(f"pos_scores{pos_scores[:3]} neg_scores{neg_scores[:3]}")
                target = torch.ones_like(pos_scores, device=device)
                loss = loss_fn(pos_scores, neg_scores, target)
                loss.backward()
                optimizer.step()
            # pos_scores = model(pos_bg)
            # neg_scores = model(neg_bg)
            # target = torch.ones_like(pos_scores, device=device)
            # loss = loss_fn(pos_scores, neg_scores, target)
            # loss.backward()
            # optimizer.step()
            total_loss += loss.item() * pos_scores.size(0)
        avg_train_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pos_bg, pos_r, neg_bg, neg_r in valid_loader:
                pos_bg, neg_bg = pos_bg.to(device), neg_bg.to(device)
                # pos_scores = model(pos_bg)
                # neg_scores = model(neg_bg)
                # target = torch.ones_like(pos_scores, device=device)
                # loss = loss_fn(pos_scores, neg_scores, target)
                # val_loss += loss.item() * pos_scores.size(0)
                if args.use_mixed_precision:
                    with torch.amp.autocast('cuda'):
                        pos_scores = model(pos_bg)
                        neg_scores = model(neg_bg)
                        print(f"pos_scores{pos_scores[:3]} neg_scores{neg_scores[:3]}")
                        assert torch.isfinite(pos_scores).all()
                        assert torch.isfinite(neg_scores).all()
                        target = torch.ones_like(pos_scores, device=device)
                        loss = loss_fn(pos_scores, neg_scores, target)
                else:
                    pos_scores = model(pos_bg)
                    neg_scores = model(neg_bg)
                    print(f"pos_scores{pos_scores[:3]} neg_scores{neg_scores[:3]}")
                    assert torch.isfinite(pos_scores).all()
                    assert torch.isfinite(neg_scores).all()
                    target = torch.ones_like(pos_scores, device=device)
                    loss = loss_fn(pos_scores, neg_scores, target)
                val_loss += loss.item() * pos_scores.size(0)

        avg_val_loss = val_loss / len(valid_ds)
        print(f"Epoch {epoch}/{args.epochs} - Val Loss: {avg_val_loss:.4f}")

if __name__ == "__main__":
    main()
