import lmdb
import pickle
import numpy as np
from tqdm import tqdm
import os

def load_csr_graph(pkl_path):
    import pickle
    with open(pkl_path, 'rb') as f:
        csr = pickle.load(f)
    return csr

def count_edges_in_subgraph(csr_graph, nodes):
    # nodes: list[int] hoặc np.array
    nodes = np.asarray(nodes)
    node_set = set(nodes)
    count = 0
    for n in nodes:
        neighbors = csr_graph.indices[csr_graph.indptr[n]:csr_graph.indptr[n+1]]
        count += np.isin(neighbors, nodes).sum()
    # Nếu là undirected thì chia 2 (mỗi cạnh đếm 2 lần)
    return count // 2 if not hasattr(csr_graph, 'directed') or not csr_graph.directed else count

def lmdb_summary(lmdb_path, csr_graph_path, max_samples=None):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    csr_graph = load_csr_graph(csr_graph_path)
    subgraph_node_counts = []
    subgraph_edge_counts = []
    n_total = 0

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in tqdm(cursor, desc="Reading LMDB"):
            if key == b'_progress':
                continue
            if value and value[:1] in [b'\x80', b'\x81']:
                data = pickle.loads(value)
                nodes = data['nodes']
                if not nodes:  # Nếu subgraph rỗng thì bỏ qua
                    continue
                n_nodes = len(nodes)
                n_edges = count_edges_in_subgraph(csr_graph, nodes)
                subgraph_node_counts.append(n_nodes)
                subgraph_edge_counts.append(n_edges)
                n_total += 1
                if max_samples and n_total >= max_samples:
                    break

    node_counts = np.array(subgraph_node_counts)
    edge_counts = np.array(subgraph_edge_counts)
    print("===== Subgraph Extraction Summary =====")
    print(f"Total subgraphs: {n_total}")
    print(f"Nodes per subgraph: mean={node_counts.mean():.2f}, median={np.median(node_counts)}, min={node_counts.min()}, max={node_counts.max()}")
    print(f"Edges per subgraph: mean={edge_counts.mean():.2f}, median={np.median(edge_counts)}, min={edge_counts.min()}, max={edge_counts.max()}")
    print(f"5th, 25th, 75th, 95th percentiles (nodes): {np.percentile(node_counts, [5,25,75,95])}")
    print(f"5th, 25th, 75th, 95th percentiles (edges): {np.percentile(edge_counts, [5,25,75,95])}")

