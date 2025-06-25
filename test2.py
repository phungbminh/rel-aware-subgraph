# test_cugraph_relation_khop.py

import torch
import numpy as np
from ogb.linkproppred import LinkPropPredDataset
from utils.cugraph_utils import (
    build_cugraph,                # MultiGraph cho filter/edge label
    filter_by_relation_tau,
    extract_cugraph_subgraph
)
import cudf
import cugraph

def build_cugraph_simple_relation(edge_index, edge_type, r):
    # Chỉ lấy cạnh đúng relation r
    mask = (edge_type.cpu().numpy() == r)
    src = edge_index[0].cpu().numpy()[mask]
    dst = edge_index[1].cpu().numpy()[mask]
    df = cudf.DataFrame({'src': src, 'dst': dst})
    G = cugraph.Graph()
    G.from_cudf_edgelist(df, source='src', destination='dst')
    return G

def cugraph_k_hop(G, nodes, k):
    all_nodes = []
    for node in nodes:
        bfs_result = cugraph.bfs(G, start=int(node), depth_limit=k)
        neighbors = bfs_result['vertex'].values_host
        all_nodes.append(neighbors)
    all_nodes = np.unique(np.concatenate(all_nodes))
    return all_nodes.astype(np.int64)

def cugraph_shortest_dist(G, source, node_list):
    bfs_result = cugraph.bfs(G, start=int(source))
    dist_map = dict(zip(bfs_result['vertex'].values_host, bfs_result['distance'].values_host))
    return np.array([dist_map.get(int(n), -1) for n in node_list], dtype=np.int64)

def extract_relation_aware_subgraph_cugraph(G_simple_r, G_full, h, t, r, k, tau):
    print(f"Try extract with h={h}, t={t}, r={r}")
    # 1. Kiểm tra node tồn tại trong G_simple_r
    def node_in_cugraph(G, node):
        df = G.view_edge_list()
        src = set(df['src'].to_pandas().tolist())
        dst = set(df['dst'].to_pandas().tolist())
        return (node in src) or (node in dst)
    in_h = node_in_cugraph(G_simple_r, h)
    in_t = node_in_cugraph(G_simple_r, t)
    print(f"h in G_simple_r: {in_h}, t in G_simple_r: {in_t}")

    if not in_h and not in_t:
        print("Neither h nor t exists in G_simple_r: subgraph = [h, t] only.")
        filtered_nodes = np.array([h, t], dtype=np.int64)
        sub_edge_index = torch.empty((2, 0), dtype=torch.long)
        sub_edge_type = torch.empty((0,), dtype=torch.long)
        node_label = torch.zeros((2, 2), dtype=torch.long)
        return torch.from_numpy(filtered_nodes), sub_edge_index, sub_edge_type, node_label

    # 2. K-hop neighbors (chỉ trên relation r, từ node tồn tại)
    seed_nodes = []
    if in_h: seed_nodes.append(h)
    if in_t: seed_nodes.append(t)
    sub_nodes = cugraph_k_hop(G_simple_r, seed_nodes, k)
    print(f"[extract_subgraph] k_hop (relation={r}): {len(sub_nodes)} nodes")
    # 3. Relation-aware filtering (trên G_full)
    filtered_nodes = filter_by_relation_tau(G_full, sub_nodes, r, tau)
    print(f"[extract_subgraph] Filtered nodes (rel=={r}, tau>={tau}): {len(filtered_nodes)}")
    if len(filtered_nodes) == 0:
        filtered_nodes = np.array([h, t], dtype=np.int64)
    # 4. Build subgraph & relabel
    sub_edge_index, sub_edge_type, num_nodes, old2new = extract_cugraph_subgraph(G_full, filtered_nodes)
    print(f"[extract_subgraph] Subgraph: {sub_edge_index.shape[1]} edges")
    # 5. Node labeling
    dist_h = cugraph_shortest_dist(G_simple_r, h, filtered_nodes) if in_h else np.full(len(filtered_nodes), -1)
    dist_t = cugraph_shortest_dist(G_simple_r, t, filtered_nodes) if in_t else np.full(len(filtered_nodes), -1)
    node_label = torch.from_numpy(np.stack([dist_h, dist_t], axis=1)).long()
    filtered_nodes = torch.from_numpy(filtered_nodes).long()
    print("[extract_subgraph] Node labeling done")
    return filtered_nodes, sub_edge_index, sub_edge_type, node_label

def main():
    dataset = LinkPropPredDataset(name='ogbl-biokg', root='./data/ogb/')
    split_edge = dataset.get_edge_split()
    heads = split_edge['train']['head']
    tails = split_edge['train']['tail']
    rels = split_edge['train']['relation']

    heads_torch = torch.from_numpy(heads)
    tails_torch = torch.from_numpy(tails)
    rels_torch = torch.from_numpy(rels)
    edge_index = torch.stack([heads_torch, tails_torch])
    edge_type = rels_torch

    G_full = build_cugraph(edge_index, edge_type)

    # Chọn 1 triplet bất kỳ
    idx = 0
    h = int(heads[idx])
    t = int(tails[idx])
    r = int(rels[idx])
    k = 1
    tau = 1

    # --- Kiểm tra degree của h và t trong toàn bộ KG (mọi relation) ---
    df = G_full.view_edge_list()
    deg_h_src = (df['src'] == h).sum()
    deg_h_dst = (df['dst'] == h).sum()
    deg_t_src = (df['src'] == t).sum()
    deg_t_dst = (df['dst'] == t).sum()
    print(f"Degree of h={h}: out={deg_h_src}, in={deg_h_dst}")
    print(f"Degree of t={t}: out={deg_t_src}, in={deg_t_dst}")

    # --- Xây graph chỉ với relation r ---
    G_simple_r = build_cugraph_simple_relation(edge_index, edge_type, r)

    # --- Extract relation-aware subgraph ---
    filtered_nodes, sub_edge_index, sub_edge_type, node_label = extract_relation_aware_subgraph_cugraph(
        G_simple_r, G_full, h, t, r, k, tau
    )

    print("Done! Subgraph node count:", filtered_nodes.shape[0])
    print("Subgraph edge count:", sub_edge_index.shape[1])
    print("Node labels shape:", node_label.shape)
    print("Sample node labels:", node_label[:min(5, len(node_label))])

if __name__ == "__main__":
    main()
