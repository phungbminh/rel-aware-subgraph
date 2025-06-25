import torch
import numpy as np

def build_cugraph_simple(edge_index):
    import cudf
    import cugraph
    """
    Build a simple directed cugraph.Graph (no edge type) for BFS/k-hop extraction.
    """
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    df = cudf.DataFrame({'src': src, 'dst': dst})
    G = cugraph.Graph(directed=True)  # Đảm bảo đồ thị có hướng
    G.from_cudf_edgelist(df, source='src', destination='dst')
    return G

def edge_index_to_cudf(edge_index, edge_type):
    import cudf
    """
    Convert edge_index, edge_type (torch) to cudf DataFrame for cuGraph MultiGraph.
    """
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    rel = edge_type.cpu().numpy()
    df = cudf.DataFrame({'src': src, 'dst': dst, 'etype': rel})
    return df

def build_cugraph(edge_index, edge_type):
    import cugraph
    """
    Build a cuGraph.MultiGraph with edge types.
    """
    df_edges = edge_index_to_cudf(edge_index, edge_type)
    G = cugraph.MultiGraph()
    G.from_cudf_edgelist(df_edges, source='src', destination='dst', edge_attr='etype')
    return G

def cugraph_k_hop(G, nodes, k):
    import cugraph
    """
    Get all k-hop neighbors (union) for a list of nodes using cugraph.Graph.
    """
    all_nodes = []
    for node in nodes:
        bfs_result = cugraph.bfs(G, start=int(node), depth_limit=k)
        neighbors = bfs_result['vertex'].values_host
        all_nodes.append(neighbors)
    all_nodes = np.unique(np.concatenate(all_nodes))
    return all_nodes.astype(np.int64)
# Cập nhật trong utils/cugraph_utils.py:
def cugraph_k_hop_batch(G, seed_nodes, k):
    import cugraph
    bfs_result = cugraph.bfs(G, start=seed_nodes, depth_limit=k)
    all_nodes = bfs_result['vertex'].values_host
    return np.unique(all_nodes).astype(np.int64)


def filter_by_relation_tau(G, sub_nodes, rel, tau):
    """
    Filter nodes in sub_nodes having at least tau outgoing edges of type rel (using MultiGraph).
    """
    df = G.view_edge_list()
    mask = (df['etype'] == rel) & df['src'].isin(sub_nodes)
    rel_counts = df[mask].groupby('src').size().reset_index(name='count')
    if len(rel_counts) == 0:
        return np.array([], dtype=np.int64)
    keep_nodes = rel_counts[rel_counts['count'] >= tau]['src'].values_host
    return keep_nodes.astype(np.int64)

def extract_cugraph_subgraph(G, node_ids):
    """
    Extract subgraph from G containing only node_ids and relabel nodes to [0, num_sub-1].
    Returns (edge_index, edge_type, num_nodes, old2new mapping).
    """
    df = G.view_edge_list()
    mask = df['src'].isin(node_ids) & df['dst'].isin(node_ids)
    sub_edges = df[mask]
    old2new = {int(n): i for i, n in enumerate(node_ids)}
    src = sub_edges['src'].to_numpy()
    dst = sub_edges['dst'].to_numpy()
    etype = sub_edges['etype'].to_numpy()
    if len(src) == 0:
        # fallback: return empty tensors
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
    else:
        src_new = np.array([old2new[int(x)] for x in src])
        dst_new = np.array([old2new[int(x)] for x in dst])
        edge_index = torch.tensor(np.stack([src_new, dst_new]), dtype=torch.long)
        edge_type = torch.tensor(etype, dtype=torch.long)
    return edge_index, edge_type, len(node_ids), old2new

def cugraph_shortest_dist(G, source, node_list):
    import cugraph
    """
    Compute shortest path length from source to all nodes in node_list using cugraph.BFS.
    Returns -1 for unreachable nodes.
    """
    bfs_result = cugraph.bfs(G, start=int(source))
    dist_map = dict(zip(bfs_result['vertex'].values_host, bfs_result['distance'].values_host))
    # -1 cho unreachable node (hoặc dùng 999 để mask)
    return np.array([dist_map.get(int(n), -1) for n in node_list], dtype=np.int64)

from collections import Counter

def compute_degrees(heads, tails):
    out_deg = Counter(heads.tolist())
    in_deg = Counter(tails.tolist())
    return out_deg, in_deg

def filter_triples_by_degree(heads, tails, rels, max_degree=500):
    out_deg, in_deg = compute_degrees(heads, tails)
    mask = [(out_deg[int(h)] <= max_degree and in_deg[int(t)] <= max_degree)
            for h, t in zip(heads, tails)]
    mask = np.array(mask)
    return heads[mask], tails[mask], rels[mask]
