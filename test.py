# test_cugraph_subgraph.py

import torch
import numpy as np
from ogb.linkproppred import LinkPropPredDataset
from utils.cugraph_utils import (
    build_cugraph,
    filter_by_relation_tau,
    extract_cugraph_subgraph
)
import cudf
import cugraph

# Hàm build graph đơn giản cho BFS/k-hop
def build_cugraph_simple(edge_index):
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    df = cudf.DataFrame({'src': src, 'dst': dst})
    G = cugraph.Graph()  # mặc định directed
    G.from_cudf_edgelist(df, source='src', destination='dst')
    return G

# K-hop sử dụng G_simple
def cugraph_k_hop(G, nodes, k):
    all_nodes = []
    for node in nodes:
        bfs_result = cugraph.bfs(G, start=int(node), depth_limit=k)
        neighbors = bfs_result['vertex'].values_host
        all_nodes.append(neighbors)
    all_nodes = np.unique(np.concatenate(all_nodes))
    return all_nodes.astype(np.int64)

# Node labeling: shortest distance
def cugraph_shortest_dist(G, source, node_list):
    bfs_result = cugraph.bfs(G, start=int(source))
    dist_map = dict(zip(bfs_result['vertex'].values_host, bfs_result['distance'].values_host))
    return np.array([dist_map.get(int(n), -1) for n in node_list], dtype=np.int64)

def extract_relation_aware_subgraph_cugraph(G_simple, G_full, h, t, r, k, tau):
    print(f"Try extract with h={h}, t={t}, r={r}")
    sub_nodes = cugraph_k_hop(G_simple, [h, t], k)
    print(f"[extract_subgraph] k_hop: {len(sub_nodes)} nodes")
    filtered_nodes = filter_by_relation_tau(G_full, sub_nodes, r, tau)
    print(f"[extract_subgraph] Filtered nodes (rel=={r}, tau>={tau}): {len(filtered_nodes)}")
    if len(filtered_nodes) == 0:
        filtered_nodes = np.array([h, t], dtype=np.int64)
    sub_edge_index, sub_edge_type, num_nodes, old2new = extract_cugraph_subgraph(G_full, filtered_nodes)
    print(f"[extract_subgraph] Subgraph: {sub_edge_index.shape[1]} edges")
    dist_h = cugraph_shortest_dist(G_simple, h, filtered_nodes)
    dist_t = cugraph_shortest_dist(G_simple, t, filtered_nodes)
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

    # CHỈ 1 CHIỀU head -> tail
    heads_torch = torch.from_numpy(heads)
    tails_torch = torch.from_numpy(tails)
    rels_torch = torch.from_numpy(rels)
    edge_index = torch.stack([heads_torch, tails_torch])
    edge_type = rels_torch

    G_full = build_cugraph(edge_index, edge_type)
    G_simple = build_cugraph_simple(edge_index)

    h = int(heads[0])
    t = int(tails[0])
    r = int(rels[0])
    k = 1
    tau = 1

    filtered_nodes, sub_edge_index, sub_edge_type, node_label = extract_relation_aware_subgraph_cugraph(
        G_simple, G_full, h, t, r, k, tau
    )

    print("Done! Subgraph node count:", filtered_nodes.shape[0])
    print("Subgraph edge count:", sub_edge_index.shape[1])
    print("Node labels shape:", node_label.shape)
    print("Sample node labels:", node_label[:min(5, len(node_label))])

if __name__ == "__main__":
    main()
