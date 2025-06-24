import argparse
import os
import lmdb
import pickle
import numpy as np
from tqdm import tqdm
from ogb.linkproppred import LinkPropPredDataset
from utils.graph_utils import ssp_multigraph_to_pyg
from utils.cugraph_utils import build_cugraph, cugraph_shortest_dist  # Chỉ import khi cần
from extraction import (
    extract_relation_aware_subgraph,         # bản CPU cũ
    extract_relation_aware_subgraph_cugraph  # bản GPU (mới)
)
import multiprocessing as mp


import multiprocessing as mp
from tqdm import tqdm
import pickle
import lmdb
import os

import torch

def is_cuda_available():
    try:
        import cupy
        cuda_cnt = cupy.cuda.runtime.getDeviceCount()
        return cuda_cnt > 0
    except ImportError:
        import torch
        return torch.cuda.is_available()

def build_graph_backend(edge_index, edge_type):
    if is_cuda_available():
        print("[INFO] CUDA detected – using cuGraph (GPU) for subgraph extraction!")
        from utils.cugraph_utils import build_cugraph
        G = build_cugraph(edge_index, edge_type)
        return G, 'cugraph'
    else:
        print("[INFO] CUDA not found – using PyG (CPU) for subgraph extraction.")
        return (edge_index, edge_type), 'pyg'


global_edge_index = None
global_edge_type = None
global_num_nodes = None
global_k = None
global_tau = None
global_num_neg_samples_per_link = None
global_backend = None
global_graph = None

def _init_worker(edge_index, edge_type, num_nodes, k, tau, num_neg_samples_per_link, backend, graph):
    global global_edge_index, global_edge_type, global_num_nodes, global_k, global_tau
    global global_num_neg_samples_per_link, global_backend, global_graph
    global_edge_index = edge_index
    global_edge_type = edge_type
    global_num_nodes = num_nodes
    global_k = k
    global_tau = tau
    global_num_neg_samples_per_link = num_neg_samples_per_link
    global_backend = backend
    global_graph = graph

def extract_for_one_worker(triple):
    h, t, r = triple
    try:
        if global_backend == 'cugraph':
            filtered_nodes, sub_edge_index, sub_edge_type, node_label = extract_relation_aware_subgraph_cugraph(
                global_graph, int(h), int(t), int(r), global_k, global_tau
            )
        else:
            filtered_nodes, sub_edge_index, sub_edge_type, node_label = extract_relation_aware_subgraph(
                global_edge_index, global_edge_type, int(h), int(t), int(r),
                global_num_nodes, global_k, global_tau
            )
        subgraph_data = {'h': int(h), 't': int(t), 'r_label': int(r), 'g_label': 1, 'nodes': filtered_nodes.tolist()}
        # Negative sampling
        neg_samples = []
        for _ in range(global_num_neg_samples_per_link):
            for _ in range(10):
                t_neg = np.random.randint(0, global_num_nodes)
                if t_neg != t:
                    break
            else:
                t_neg = (int(t) + 1) % global_num_nodes
            if global_backend == 'cugraph':
                filtered_nodes_neg, sub_edge_index_neg, sub_edge_type_neg, node_label_neg = extract_relation_aware_subgraph_cugraph(
                    global_graph, int(h), int(t_neg), int(r), global_k, global_tau
                )
            else:
                filtered_nodes_neg, sub_edge_index_neg, sub_edge_type_neg, node_label_neg = extract_relation_aware_subgraph(
                    global_edge_index, global_edge_type, int(h), int(t_neg), int(r),
                    global_num_nodes, global_k, global_tau
                )
            neg_data = {'h': int(h), 't': int(t_neg), 'r_label': int(r), 'g_label': 0, 'nodes': filtered_nodes_neg.tolist()}
            neg_samples.append(neg_data)
        return subgraph_data, neg_samples
    except Exception as e:
        print(f"[WARNING] Skip triple ({h},{t},{r}) with error: {e}")
        return None


def build_split_subgraph_parallel(
    split_name, triples, edge_index, edge_type, num_nodes,
    db_path, num_neg_samples_per_link=1, k=2, tau=2,
    map_size=int(2e10), num_workers=4, max_links=None, backend=None, graph=None
):
    # Chuẩn bị LMDB
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    env = lmdb.open(db_path, map_size=map_size, max_dbs=4, lock=False)
    db_pos = env.open_db(b'positive')
    db_neg = env.open_db(b'negative')
    if max_links is not None:
        triples = triples[:max_links]
        print(f"Processing only {max_links} triples for split={split_name}")

    # Pool chuẩn spawn context cho MacOS/Linux/Windows
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
    args = parser.parse_args()

    # 1. Load OGB-BioKG và chia split
    dataset = LinkPropPredDataset(name='ogbl-biokg', root=args.ogb_root)
    split_edge = dataset.get_edge_split()
    print(f"Splits found: {[k for k in split_edge]}")

    # 2. Build full KG graph
    triples_all = np.concatenate([
        np.stack([split_edge['train']['head'], split_edge['train']['relation'], split_edge['train']['tail']], axis=1),
        np.stack([split_edge['valid']['head'], split_edge['valid']['relation'], split_edge['valid']['tail']], axis=1),
        np.stack([split_edge['test']['head'], split_edge['test']['relation'], split_edge['test']['tail']], axis=1),
    ], axis=0)
    n_entities = int(triples_all[:, [0,2]].max()) + 1
    n_relations = int(triples_all[:,1].max()) + 1
    print(f"KG: {n_entities} entities, {n_relations} relations")

    # Build adjacency list cho multi-rel
    from scipy.sparse import csr_matrix
    adj_list = []
    for rel in range(n_relations):
        mask = (triples_all[:,1] == rel)
        heads, tails = triples_all[mask, 0], triples_all[mask, 2]
        data = np.ones(len(heads), dtype=np.int8)
        adj = csr_matrix((data, (heads, tails)), shape=(n_entities, n_entities))
        adj_list.append(adj)
    # Chuyển sang PyG Data
    # pyg_graph = ssp_multigraph_to_pyg(adj_list)
    # edge_index, edge_type = pyg_graph.edge_index, pyg_graph.edge_type
    # num_nodes = pyg_graph.num_nodes

    # Chuyển sang PyG Data
    pyg_graph = ssp_multigraph_to_pyg(adj_list)
    edge_index, edge_type = pyg_graph.edge_index, pyg_graph.edge_type
    num_nodes = pyg_graph.num_nodes

    # Detect backend và build graph phù hợp
    graph, backend = build_graph_backend(edge_index, edge_type)

    # 3. Build subgraph cho từng split
    for split in args.split:
        edges = split_edge[split]
        triples = np.stack([edges['head'], edges['tail'], edges['relation']], axis=1)
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
            backend=backend,  # THÊM DÒNG NÀY
            graph=graph  # THÊM DÒNG NÀY
        )

if __name__ == "__main__":
    main()
