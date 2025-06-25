import argparse
import os
import lmdb
import pickle
import numpy as np
import torch
from tqdm import tqdm
from ogb.linkproppred import LinkPropPredDataset
from scipy.sparse import csr_matrix

from utils.graph_utils import ssp_multigraph_to_pyg
from utils.cugraph_utils import (
    build_cugraph, cugraph_shortest_dist, build_cugraph_simple, filter_triples_by_degree
)
from extraction import (
    extract_relation_aware_subgraph,         # CPU
    extract_relation_aware_subgraph_cugraph  # GPU
)
import multiprocessing as mp
from typing import Any, Tuple, List, Optional

# ======== Worker-global variables for multiprocessing ========
global_edge_index = None
global_edge_type = None
global_num_nodes = None
global_k = None
global_tau = None
global_num_neg_samples_per_link = None
global_backend = None
global_graph = None

def _init_worker(
    edge_index, edge_type, num_nodes, k, tau, num_neg_samples_per_link, backend, graph
):
    """Initializer for multiprocessing workers."""
    global global_edge_index, global_edge_type, global_num_nodes
    global global_k, global_tau, global_num_neg_samples_per_link
    global global_backend, global_graph
    global_edge_index = edge_index
    global_edge_type = edge_type
    global_num_nodes = num_nodes
    global_k = k
    global_tau = tau
    global_num_neg_samples_per_link = num_neg_samples_per_link
    global_backend = backend
    global_graph = graph

def extract_for_one_worker(triple: Tuple[int, int, int]) -> Optional[Tuple[dict, List[dict]]]:
    """Extracts subgraph for a single triple (positive + negatives)."""
    h, t, r = triple
    try:
        if global_backend == 'cugraph':
            G_simple, G_full = global_graph
            filtered_nodes, _, _, _ = extract_relation_aware_subgraph_cugraph(
                G_simple, G_full, int(h), int(t), int(r), global_k, global_tau
            )
        else:
            filtered_nodes, _, _, _ = extract_relation_aware_subgraph(
                global_edge_index, global_edge_type, int(h), int(t), int(r),
                global_num_nodes, global_k, global_tau
            )
        subgraph_data = {'h': int(h), 't': int(t), 'r_label': int(r), 'g_label': 1, 'nodes': filtered_nodes.tolist()}

        # --- Negative sampling ---
        neg_samples = []
        for _ in range(global_num_neg_samples_per_link):
            # Try 10 times for a different tail
            for _ in range(10):
                t_neg = np.random.randint(0, global_num_nodes)
                if t_neg != t:
                    break
            else:
                t_neg = (int(t) + 1) % global_num_nodes
            if global_backend == 'cugraph':
                filtered_nodes_neg, _, _, _ = extract_relation_aware_subgraph_cugraph(
                    G_simple, G_full, int(h), int(t_neg), int(r), global_k, global_tau
                )
            else:
                filtered_nodes_neg, _, _, _ = extract_relation_aware_subgraph(
                    global_edge_index, global_edge_type, int(h), int(t_neg), int(r),
                    global_num_nodes, global_k, global_tau
                )
            neg_data = {
                'h': int(h), 't': int(t_neg), 'r_label': int(r),
                'g_label': 0, 'nodes': filtered_nodes_neg.tolist()
            }
            neg_samples.append(neg_data)
        return subgraph_data, neg_samples
    except Exception as e:
        print(f"[WARNING] Skip triple ({h},{t},{r}) with error: {e}")
        return None

def build_graph_backend(edge_index: torch.Tensor, edge_type: torch.Tensor, backend: str):
    """Khởi tạo backend (PyG hoặc cuGraph) tùy chọn."""
    if backend == "cugraph":
        print("[INFO] Using cuGraph (GPU) for subgraph extraction!")
        G_full = build_cugraph(edge_index, edge_type)
        G_simple = build_cugraph_simple(edge_index)
        return (G_simple, G_full), "cugraph"
    else:
        print("[INFO] Using PyG (CPU) for subgraph extraction.")
        return (edge_index, edge_type), "pyg"

def build_split_subgraph_parallel(
    split_name: str, triples: np.ndarray, edge_index: torch.Tensor, edge_type: torch.Tensor,
    num_nodes: int, db_path: str, num_neg_samples_per_link: int = 1, k: int = 2, tau: int = 2,
    map_size: int = int(2e10), num_workers: int = 4, max_links: Optional[int] = None,
    backend: str = "pyg", graph: Any = None
):
    """Sinh subgraph cho từng triple (song song hoặc tuần tự tùy backend), lưu ra LMDB."""
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    env = lmdb.open(db_path, map_size=map_size, max_dbs=4, lock=False)
    db_pos = env.open_db(b'positive')
    db_neg = env.open_db(b'negative')

    if max_links is not None:
        triples = triples[:max_links]
        print(f"Processing only {max_links} triples for split={split_name}")

    # Set global cho worker (trường hợp multiprocessing)
    global global_backend, global_graph, global_edge_index, global_edge_type, global_num_nodes, global_k, global_tau, global_num_neg_samples_per_link
    global_backend = backend
    global_graph = graph
    global_edge_index = edge_index
    global_edge_type = edge_type
    global_num_nodes = num_nodes
    global_k = k
    global_tau = tau
    global_num_neg_samples_per_link = num_neg_samples_per_link

    if backend == "cugraph":
        print(f"[INFO] Using cuGraph (GPU) – run sequentially (no multiprocessing pool)")
        with env.begin(write=True) as txn:
            for idx, triple in tqdm(
                enumerate(triples), total=len(triples), desc=f"Build {split_name} subgraphs"
            ):
                result = extract_for_one_worker(triple)
                if result is None:
                    continue
                subgraph_data, neg_samples = result
                txn.put(str(idx).encode(), pickle.dumps(subgraph_data), db=db_pos)
                txn.put(str(idx).encode(), pickle.dumps(neg_samples), db=db_neg)
    else:
        print(f"[INFO] Using PyG (CPU) – run with multiprocessing Pool ({num_workers} workers)")
        ctx = mp.get_context("spawn")
        with env.begin(write=True) as txn:
            with ctx.Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(edge_index, edge_type, num_nodes, k, tau, num_neg_samples_per_link, backend, graph)
            ) as pool:
                for idx, result in tqdm(
                    enumerate(pool.imap(extract_for_one_worker, triples)),
                    total=len(triples), desc=f"Build {split_name} subgraphs"
                ):
                    if result is None:
                        continue
                    subgraph_data, neg_samples = result
                    txn.put(str(idx).encode(), pickle.dumps(subgraph_data), db=db_pos)
                    txn.put(str(idx).encode(), pickle.dumps(neg_samples), db=db_neg)
    print(f"Saved {len(triples)} positive/negative subgraphs to {db_path}")

def build_adj_list(triples_all: np.ndarray, n_entities: int, n_relations: int) -> List[csr_matrix]:
    """Build adjacency list for each relation."""
    adj_list = []
    for rel in range(n_relations):
        mask = (triples_all[:, 1] == rel)
        heads, tails = triples_all[mask, 0], triples_all[mask, 2]
        data = np.ones(len(heads), dtype=np.int8)
        adj = csr_matrix((data, (heads, tails)), shape=(n_entities, n_entities))
        adj_list.append(adj)
    return adj_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ogb-root", type=str, required=True, help="OGB data dir, contains ogbl-biokg/")
    parser.add_argument("--split", type=str, nargs='+', default=['train','valid','test'], help="Splits to build")
    parser.add_argument("--db-root", type=str, required=True, help="Output dir for LMDBs (will create lmdb_train, lmdb_valid, ...)")
    parser.add_argument("--num-neg-samples-per-link", type=int, default=1)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--tau", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-links", type=int, default=None, help="Max number of triples per split (for fast test)")
    parser.add_argument("--backend", type=str, default="pyg", choices=["pyg", "cugraph"],
                        help="Subgraph extraction backend: 'pyg' (CPU) or 'cugraph' (GPU)")
    args = parser.parse_args()

    # 1. Load OGB-BioKG và chia split
    dataset = LinkPropPredDataset(name='ogbl-biokg', root=args.ogb_root)
    split_edge = dataset.get_edge_split()
    print(f"Splits found: {list(split_edge.keys())}")

    # 2. Build full KG graph
    triples_all = np.concatenate([
        np.stack([split_edge['train']['head'], split_edge['train']['relation'], split_edge['train']['tail']], axis=1),
        np.stack([split_edge['valid']['head'], split_edge['valid']['relation'], split_edge['valid']['tail']], axis=1),
        np.stack([split_edge['test']['head'], split_edge['test']['relation'], split_edge['test']['tail']], axis=1),
    ], axis=0)
    n_entities = int(triples_all[:, [0, 2]].max()) + 1
    n_relations = int(triples_all[:, 1].max()) + 1
    print(f"KG: {n_entities} entities, {n_relations} relations")

    # 3. Build adjacency list and convert to PyG Data
    adj_list = build_adj_list(triples_all, n_entities, n_relations)
    pyg_graph = ssp_multigraph_to_pyg(adj_list)

    heads_torch = torch.from_numpy(triples_all[:, 0])
    tails_torch = torch.from_numpy(triples_all[:, 2])
    rels_torch = torch.from_numpy(triples_all[:, 1])

    # Nối 2 chiều cho edge_index/edge_type
    edge_index = torch.cat([
        torch.stack([heads_torch, tails_torch], dim=0),
        torch.stack([tails_torch, heads_torch], dim=0)
    ], dim=1)
    edge_type = torch.cat([rels_torch, rels_torch], dim=0)
    num_nodes = pyg_graph.num_nodes

    # 4. Chọn backend (CPU/GPU)
    graph, backend = build_graph_backend(edge_index, edge_type, args.backend)

    # 5. Build subgraph cho từng split
    for split in args.split:
        edges = split_edge[split]
        # Lọc triple theo degree (giảm outlier)
        heads, tails, rels = filter_triples_by_degree(
            edges['head'], edges['tail'], edges['relation'], max_degree=500
        )
        triples = np.stack([heads, tails, rels], axis=1)
        db_path = os.path.join(args.db_root, f"lmdb_{split}")
        build_split_subgraph_parallel(
            split_name=split,
            triples=triples,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=num_nodes,
            db_path=db_path,
            num_neg_samples_per_link=args.num_neg_samples_per_link,
            k=args.k,
            tau=args.tau,
            num_workers=args.num_workers,
            max_links=args.max_links,
            backend=backend,
            graph=graph
        )

if __name__ == "__main__":
    main()
