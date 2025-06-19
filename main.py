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
from trainer import Trainer

warnings.filterwarnings("ignore", category=UserWarning)


class Params:
    pass


def collate_fn(batch):
    # batch: list of tuples như trên
    pos_graphs = [item[0] for item in batch]
    pos_labels = torch.tensor([item[1] for item in batch])  # g_label_pos
    # Lấy negative đầu tiên trong mỗi sample (có thể random.choice để đa dạng)
    neg_graphs = [item[3][0] for item in batch]  # subgraphs_neg[0]
    neg_labels = torch.tensor([item[4][0] for item in batch])  # g_labels_neg[0]
    return dgl.batch(pos_graphs), pos_labels, dgl.batch(neg_graphs), neg_labels

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
    #
    # # Lấy sample đầu tiên
    # pos_sub, g_label_pos, r_label_pos, neg_subs, g_labels_neg, r_labels_neg = train_ds[0]
    #
    # print("\nSample[0] returned:")
    # print(f"  pos_sub type      : {type(pos_sub)}")
    # print(f"  neg_subs len      : {len(neg_subs)}")
    #
    # # Các field trong pos_sub
    # print("\npos_sub.ndata keys:", list(pos_sub.ndata.keys()))
    # print("  feat   shape     :", tuple(pos_sub.ndata['feat'].shape))
    # print("  query_rel shape  :", tuple(pos_sub.ndata['query_rel'].shape))
    # print("\npos_sub.edata keys:", list(pos_sub.edata.keys()))
    # print("  type   shape     :", tuple(pos_sub.edata['type'].shape))

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
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        valid_dataset=valid_ds,
        batch_size=8,
        lr=0.001,
        margin=1.0,
        optimizer_name='Adam',
        device=device,  # hoặc 'cpu'
        collate_fn=collate_fn,  # collate của bạn
        exp_dir='./exp_grail',
        save_every=5
    )

    # --- 4. Train ---
    trainer.train(num_epochs=20)

if __name__ == "__main__":
    main()
