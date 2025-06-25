# test_extract_cugraph.py

import torch
import numpy as np
from utils.cugraph_utils import build_cugraph, cugraph_k_hop, filter_by_relation_tau, extract_cugraph_subgraph, cugraph_shortest_dist

def extract_relation_aware_subgraph_cugraph(G, h, t, r, k, tau):
    print(f"Try extract with h={h}, t={t}, r={r}")
    # 1. K-hop neighbors
    sub_nodes = cugraph_k_hop(G, [h, t], k)
    print(f"[extract_subgraph] k_hop: {len(sub_nodes)} nodes")
    # 2. Relation-aware filtering
    filtered_nodes = filter_by_relation_tau(G, sub_nodes, r, tau)
    print(f"[extract_subgraph] Filtered nodes (rel=={r}, tau>={tau}): {len(filtered_nodes)}")
    if len(filtered_nodes) == 0:
        filtered_nodes = np.array([h, t], dtype=np.int64)
    # 3. Build subgraph & relabel
    sub_edge_index, sub_edge_type, num_nodes, old2new = extract_cugraph_subgraph(G, filtered_nodes)
    print(f"[extract_subgraph] Subgraph: {sub_edge_index.shape[1]} edges")
    # 4. Node labeling
    dist_h = cugraph_shortest_dist(G, h, filtered_nodes)
    dist_t = cugraph_shortest_dist(G, t, filtered_nodes)
    node_label = torch.from_numpy(np.stack([dist_h, dist_t], axis=1)).long()
    filtered_nodes = torch.from_numpy(filtered_nodes).long()
    print("[extract_subgraph] Node labeling done")
    return filtered_nodes, sub_edge_index, sub_edge_type, node_label

def main():
    # ==== THAY BẰNG ĐƯỜNG DẪN FILE EDGELIST (hoặc build bằng OGB) ====
    # Giả sử bạn đã có edge_index, edge_type dạng torch.tensor ở CPU (VD load từ PyG, OGB,...)
    # Dưới đây chỉ là ví dụ nhỏ, hãy load edge_index, edge_type thật của bạn!
    # edge_index = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
    # edge_type = torch.tensor([0,1,0], dtype=torch.long)
    # Hoặc dùng loader của bạn:
    # ... load edge_index, edge_type từ file (numpy hoặc torch) ...

    # Ví dụ load từ file .npz
    # data = np.load('your_edge_index_and_type.npz')
    # edge_index = torch.tensor(data['edge_index'], dtype=torch.long)
    # edge_type = torch.tensor(data['edge_type'], dtype=torch.long)

    # --------- GỢI Ý: Nếu dùng OGB -------------
    from ogb.linkproppred import LinkPropPredDataset
    dataset = LinkPropPredDataset(name='ogbl-biokg', root='./data/ogb/')
    split_edge = dataset.get_edge_split()
    heads = split_edge['train']['head']
    tails = split_edge['train']['tail']
    rels = split_edge['train']['relation']

    # Đổi sang torch
    heads_torch = torch.from_numpy(heads)
    tails_torch = torch.from_numpy(tails)
    rels_torch = torch.from_numpy(rels)

    edge_index = torch.stack([
        torch.cat([heads_torch, tails_torch]),
        torch.cat([tails_torch, heads_torch])
    ])
    edge_type = torch.cat([rels_torch, rels_torch])
    # -------------------------------------------

    # Build cuGraph MultiGraph
    G = build_cugraph(edge_index, edge_type)

    # ==== Lấy 1 triplet thật, ví dụ dòng đầu tiên train ====
    h, t, r = int(heads[0]), int(tails[0]), int(rels[0])
    print(f"Testing triple: h={h}, t={t}, r={r}")

    k = 1  # bán kính hop
    tau = 1  # relation-aware threshold

    # === Extract subgraph ===
    filtered_nodes, sub_edge_index, sub_edge_type, node_label = extract_relation_aware_subgraph_cugraph(
        G, h, t, r, k, tau
    )
    print("Done! Subgraph node count:", filtered_nodes.shape[0])
    print("Subgraph edge count:", sub_edge_index.shape[1])
    print("Node labels shape:", node_label.shape)
    print("Sample node labels:", node_label[:min(5, len(node_label))])

if __name__ == "__main__":
    main()
