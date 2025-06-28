import argparse
import os
import lmdb
import pickle
import numpy as np
import torch
from tqdm import tqdm
from ogb.linkproppred import LinkPropPredDataset
from scipy.sparse import csr_matrix
from collections import Counter

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
import networkx as nx
import matplotlib.pyplot as plt

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
        # Chỉ skip nếu subgraph None, không skip nếu nhỏ
        if filtered_nodes is None:
            #print(f"Skip triple ({h},{t},{r}) - subgraph is None")
            return None

        print(f"Triple ({h},{t},{r}): subgraph size {len(filtered_nodes)}")

        subgraph_data = {
            'h': int(h), 't': int(t), 'r_label': int(r), 'g_label': 1, 'nodes': filtered_nodes.tolist()
        }

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
            # Nếu negative None, vẫn cho vào sample (hoặc skip cũng được)
            if filtered_nodes_neg is None:
                print(f"  Negative sample ({h},{t_neg},{r}) is None")
                continue
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
    print(f"Split {split_name}: {triples.shape[0]} triples after filtering")
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
        results = []
        fail_count = 0
        with ctx.Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(edge_index, edge_type, num_nodes, k, tau, num_neg_samples_per_link, backend, graph)
        ) as pool:
            for result in tqdm(pool.imap(extract_for_one_worker, triples), total=len(triples)):
                if result is not None:
                    results.append(result)
                else:
                    fail_count += 1
                    continue
        print("Number of valid subgraph samples:", len(results))
        print("Number of failed (None) triples:", fail_count)

        # Sau khi gather hết result ở process cha, mở env và ghi lại:
        with env.begin(write=True) as txn:
            for idx, (subgraph_data, neg_samples) in enumerate(results):
                txn.put(str(idx).encode(), pickle.dumps(subgraph_data), db=db_pos)
                txn.put(str(idx).encode(), pickle.dumps(neg_samples), db=db_neg)
        print(f"Saved {len(results)} positive/negative subgraphs to {db_path}")

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
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-rel", type=int, default=None)
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
    # --- Nếu truyền --top-rel, lọc relation phổ biến ---
    if args.top_rel is not None:
        rel_counts = Counter(triples_all[:, 1])
        top_rels = [r for r, _ in rel_counts.most_common(args.top_rel)]
        mask = np.isin(triples_all[:, 1], top_rels)
        triples_all = triples_all[mask]
        print(f"[INFO] Lấy {args.top_rel} relation phổ biến nhất: {top_rels}, triple count = {triples_all.shape[0]}")

    # --- Nếu truyền --top-k, lọc node nhiều degree nhất ---
    if args.top_k is not None:
        degrees = np.bincount(triples_all[:, 0]) + np.bincount(triples_all[:, 2])
        if len(degrees) < triples_all[:, [0, 2]].max() + 1:
            degrees = np.pad(degrees, (0, triples_all[:, [0, 2]].max() + 1 - len(degrees)))
        top_entities = np.argsort(-degrees)[:args.top_k]
        entity_set = set(top_entities)

        mask = np.array([(h in entity_set) and (t in entity_set) for h, _, t in triples_all])
        triples_all = triples_all[mask]

        # Chọn lại node nhiều degree nhất (nếu >1000 node)
        nodes_in_filtered = set(triples_all[:, 0]).union(triples_all[:, 2])
        print(f"Node thực sự có liên kết: {len(nodes_in_filtered)}")

        if len(nodes_in_filtered) > 1000:
            degrees_filtered = np.bincount(triples_all[:, 0], minlength=max(nodes_in_filtered) + 1) + \
                               np.bincount(triples_all[:, 2], minlength=max(nodes_in_filtered) + 1)
            nodes_in_filtered = np.array(list(nodes_in_filtered))
            idx = np.argsort(-degrees_filtered[nodes_in_filtered])[:1000]
            final_nodes = set(nodes_in_filtered[idx])
            final_mask = np.array([(h in final_nodes) and (t in final_nodes) for h, _, t in triples_all])
            triples_all = triples_all[final_mask]
        print(f"Final #nodes: {len(set(triples_all[:, 0]).union(triples_all[:, 2]))}")
        print(f"Final #triples: {triples_all.shape[0]}")

    # 1. Lọc node thực sự có trong tập triple nhỏ
    nodes_in_use = np.unique(np.concatenate([triples_all[:, 0], triples_all[:, 2]]))
    num_nodes = len(nodes_in_use)
    print("Num nodes (after shrink):", num_nodes)

    # 2. Tạo mapping node id gốc → node id mới
    node_id_map = {old: new for new, old in enumerate(nodes_in_use)}
    id2entity = {v: k for k, v in node_id_map.items()}

    # 3. Ánh xạ lại triple về id mới liên tục
    triples_reindexed = np.array([
        [node_id_map[h], r, node_id_map[t]]
        for h, r, t in triples_all
    ], dtype=np.int64)

    # 4. Build lại edge_index, edge_type như chuẩn GraIL
    heads_torch = torch.from_numpy(triples_reindexed[:, 0])
    tails_torch = torch.from_numpy(triples_reindexed[:, 2])
    rels_torch = torch.from_numpy(triples_reindexed[:, 1])

    edge_index = torch.cat([
        torch.stack([heads_torch, tails_torch], dim=0),
        torch.stack([tails_torch, heads_torch], dim=0)
    ], dim=1)
    edge_type = torch.cat([rels_torch, rels_torch], dim=0)
    # 5. Tạo mapping cho relation (nếu cần)
    relation_set = sorted(set(triples_all[:, 1]))
    relation2id = {old: new for new, old in enumerate(relation_set)}
    id2relation = {v: k for k, v in relation2id.items()}

    # 6. Lưu mapping ra file để dùng downstream
    os.makedirs(args.db_root, exist_ok=True)
    with open(os.path.join(args.db_root, "entity2id.pkl"), "wb") as f:
        pickle.dump(node_id_map, f)
    with open(os.path.join(args.db_root, "id2entity.pkl"), "wb") as f:
        pickle.dump(id2entity, f)
    with open(os.path.join(args.db_root, "relation2id.pkl"), "wb") as f:
        pickle.dump(relation2id, f)
    with open(os.path.join(args.db_root, "id2relation.pkl"), "wb") as f:
        pickle.dump(id2relation, f)

    print(f"Saved mapping: entity2id, id2entity, relation2id, id2relation to {args.db_root}")

    # 4. Chọn backend (CPU/GPU)
    graph, backend = build_graph_backend(edge_index, edge_type, args.backend)

    print(
        f"[Check] edge_index.min(): {edge_index.min().item()}, edge_index.max(): {edge_index.max().item()}, num_nodes: {num_nodes}")
    assert edge_index.min().item() >= 0
    assert edge_index.max().item() < num_nodes
    for split in args.split:
        edges = split_edge[split]
        heads, tails, rels = edges['head'], edges['tail'], edges['relation']
        triples = np.stack([heads, tails, rels], axis=1)

        # Khi lấy triple từ split_edge:
        heads, tails, rels = edges['head'], edges['tail'], edges['relation']
        print("Check head:", heads[:10])
        print("Check tail:", tails[:10])
        print("Check rel:", rels[:10])

        # Lấy triple đúng thứ tự [head, relation, tail]
        triples = np.stack([heads, rels, tails], axis=1)
        mask = np.array([(h in nodes_in_use) and (t in nodes_in_use) for h, _, t in triples])
        triples = triples[mask]

        # Mapping lại node id
        triples = np.array([
            [node_id_map[h], r, node_id_map[t]]
            for h, r, t in triples
            if (h in node_id_map and t in node_id_map)
        ], dtype=np.int64)

        print("[Check] Một vài triple đầu sau mapping:")
        print(triples[:10])
        print("Relation min:", triples[:, 1].min(), "max:", triples[:, 1].max())
        # Chỉ kiểm tra head và tail node id!
        assert triples[:, 0].min() >= 0, "head node id âm"
        assert triples[:, 0].max() < num_nodes, "head node id vượt quá num_nodes"
        assert triples[:, 2].min() >= 0, "tail node id âm"
        assert triples[:, 2].max() < num_nodes, "tail node id vượt quá num_nodes"

        print("Relation min:", triples[:, 1].min(), "max:", triples[:, 1].max())
        db_path = os.path.join(args.db_root, f"lmdb_{split}")

        #visualize_graph(triples, title=f"KG {split}")


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


def visualize_graph(triples, num_nodes=None, rel_names=None, title="KG Subgraph"):
    G = nx.MultiDiGraph()
    for h, r, t in triples:
        label = str(r) if rel_names is None else rel_names.get(r, str(r))
        G.add_edge(h, t, label=label)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(G.number_of_nodes()))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    main()
