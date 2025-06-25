
import torch
from torch_geometric.utils import k_hop_subgraph, subgraph
from collections import deque
import numpy as np


import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from utils.cugraph_utils import (
    build_cugraph,
    cugraph_k_hop,
    filter_by_relation_tau,
    extract_cugraph_subgraph,
    cugraph_shortest_dist
)

def bfs_shortest_dist(edge_index, num_nodes, source):
    # Build CSR matrix
    row, col = edge_index.cpu().numpy()
    data = np.ones(len(row), dtype=np.float32)
    csr = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    dist = shortest_path(csr, directed=False, indices=source)
    dist = torch.from_numpy(dist).long()
    dist[torch.isinf(dist)] = -1
    return dist
# def bfs_shortest_dist(edge_index, start, num_nodes, max_dist=5):
#     """
#     BFS để tính khoảng cách từ node start đến tất cả các node khác (d(h,v) hoặc d(t,v))
#     Trả về tensor [num_nodes] với khoảng cách, giá trị -1 nếu không reachable hoặc vượt quá max_dist.
#     """
#     dist = torch.full((num_nodes,), -1, dtype=torch.long)
#     dist[start] = 0
#     queue = deque([start])
#     while queue:
#         curr = queue.popleft()
#         for nb in edge_index[1][edge_index[0] == curr]:
#             if dist[nb] == -1:
#                 dist[nb] = dist[curr] + 1
#                 if dist[nb] < max_dist:
#                     queue.append(nb)
#     return dist


import time
import torch
from torch_geometric.utils import k_hop_subgraph, subgraph

def extract_relation_aware_subgraph(edge_index, edge_type, h, t, r, num_nodes, k, tau):
    """
    Trích xuất subgraph k-hop quanh h, t, chỉ giữ node có edge_type == r >= tau.
    - edge_index: [2, N] tensor
    - edge_type: [N] tensor
    - h, t, r: int
    - num_nodes: int
    - k: bán kính
    - tau: threshold relation-aware filter
    """
    t0 = time.time()
    # 1. K-hop subgraph quanh h và t
    subset_h, _, _, _ = k_hop_subgraph(torch.tensor([h], dtype=torch.long), k, edge_index, relabel_nodes=False, num_nodes=num_nodes)
    print(f"[extract_subgraph] k_hop_subgraph(h): {len(subset_h)} nodes, time {time.time()-t0:.3f}s")
    t1 = time.time()
    subset_t, _, _, _ = k_hop_subgraph(torch.tensor([t], dtype=torch.long), k, edge_index, relabel_nodes=False, num_nodes=num_nodes)
    print(f"[extract_subgraph] k_hop_subgraph(t): {len(subset_t)} nodes, time {time.time()-t1:.3f}s")
    t2 = time.time()
    subset = torch.unique(torch.cat([subset_h, subset_t]))
    print(f"[extract_subgraph] Union nodes: {len(subset)}")
    # 2. Lọc node có số cạnh relation r ≥ tau
    mask_r = (edge_type == r)
    rel_counts = torch.zeros(num_nodes, dtype=torch.long)
    rel_counts.scatter_add_(0, edge_index[0, mask_r], torch.ones(mask_r.sum(), dtype=torch.long))
    mask_subset = (rel_counts[subset] >= tau)
    filtered_nodes = subset[mask_subset]
    print(f"[extract_subgraph] Filtered nodes (rel=={r}, tau>={tau}): {len(filtered_nodes)} (time {time.time()-t2:.3f}s)")
    t3 = time.time()
    # 3. Node labeling
    #dist_h = bfs_shortest_dist(edge_index, h, num_nodes)[filtered_nodes]
    #dist_t = bfs_shortest_dist(edge_index, t, num_nodes)[filtered_nodes]

    dist_h = bfs_shortest_dist(edge_index, num_nodes, h)[filtered_nodes]
    dist_t = bfs_shortest_dist(edge_index, num_nodes, t)[filtered_nodes]
    node_label = torch.stack([dist_h, dist_t], dim=1)
    print(f"[extract_subgraph] Node labeling done (time {time.time()-t3:.3f}s)")
    t4 = time.time()
    # 4. Subgraph (edge_index/edge_type) cho node đã lọc
    sub_edge_index, sub_edge_type, edge_mask = subgraph(
        filtered_nodes, edge_index, edge_attr=edge_type, relabel_nodes=True, return_edge_mask=True
    )
    print(f"[extract_subgraph] Subgraph: {sub_edge_index.shape[1]} edges (time {time.time()-t4:.3f}s)")
    print(f"[extract_subgraph] TOTAL TIME: {time.time()-t0:.3f}s")
    return filtered_nodes, sub_edge_index, sub_edge_type, node_label


def extract_relation_aware_subgraph_cugraph(G_simple, G_full, h, t, r, k, tau, verbose=True):
    times = {}
    t0 = time.time()
    # 1. K-hop trên G_simple
    t_khop_start = time.time()
    sub_nodes = cugraph_k_hop(G_simple, [h, t], k)
    times['k_hop'] = time.time() - t_khop_start
    if verbose:
        print(f"[extract_subgraph] k_hop: {len(sub_nodes)} nodes (time {times['k_hop']:.3f}s)")

    # 2. Lọc relation trên G_full
    t_filter = time.time()
    filtered_nodes = filter_by_relation_tau(G_full, sub_nodes, r, tau)
    times['relation_filter'] = time.time() - t_filter
    if verbose:
        print(f"[extract_subgraph] Filtered nodes (rel=={r}, tau>={tau}): {len(filtered_nodes)} (time {times['relation_filter']:.3f}s)")
    if len(filtered_nodes) == 0:
        filtered_nodes = np.array([h, t], dtype=np.int64)

    # 3. Subgraph edges (và relabel)
    t_subg = time.time()
    sub_edge_index, sub_edge_type, num_nodes, old2new = extract_cugraph_subgraph(G_full, filtered_nodes)
    times['subgraph'] = time.time() - t_subg
    if verbose:
        print(f"[extract_subgraph] Subgraph: {sub_edge_index.shape[1]} edges (time {times['subgraph']:.3f}s)")

    # 4. Node labeling
    t_label = time.time()
    dist_h = cugraph_shortest_dist(G_simple, h, filtered_nodes)
    dist_t = cugraph_shortest_dist(G_simple, t, filtered_nodes)
    node_label = torch.from_numpy(np.stack([dist_h, dist_t], axis=1)).long()
    filtered_nodes = torch.from_numpy(filtered_nodes).long()
    times['label'] = time.time() - t_label
    if verbose:
        print(f"[extract_subgraph] Node labeling done (time {times['label']:.3f}s)")

    total = time.time() - t0
    if verbose:
        print(f"[extract_subgraph] TOTAL TIME: {total:.3f}s")
    return filtered_nodes, sub_edge_index, sub_edge_type, node_label


# def extract_relation_aware_subgraph(edge_index, edge_type, h, t, r, num_nodes, k, tau):
#     """
#     Trích xuất subgraph k-hop quanh h, t, chỉ giữ node có edge_type == r >= tau.
#     - edge_index: [2, N] tensor
#     - edge_type: [N] tensor
#     - h, t, r: int
#     - num_nodes: int
#     - k: bán kính
#     - tau: threshold relation-aware filter
#     """
#     # 1. K-hop subgraph quanh h và t
#     #subset_h, _, _, _ = k_hop_subgraph(h, k, edge_index, relabel_nodes=False, num_nodes=num_nodes)
#     #subset_t, _, _, _ = k_hop_subgraph(t, k, edge_index, relabel_nodes=False, num_nodes=num_nodes)
#     subset_h, _, _, _ = k_hop_subgraph(torch.tensor([h], dtype=torch.long), k, edge_index, relabel_nodes=False,
#                                        num_nodes=num_nodes)
#     subset_t, _, _, _ = k_hop_subgraph(torch.tensor([t], dtype=torch.long), k, edge_index, relabel_nodes=False,
#                                        num_nodes=num_nodes)
#
#     subset = torch.unique(torch.cat([subset_h, subset_t]))
#     # 2. Lọc node có số cạnh relation r ≥ tau
#     mask_r = (edge_type == r)
#     rel_counts = torch.zeros(num_nodes, dtype=torch.long)
#     rel_counts.scatter_add_(0, edge_index[0, mask_r], torch.ones(mask_r.sum(), dtype=torch.long))
#     mask_subset = (rel_counts[subset] >= tau)
#     filtered_nodes = subset[mask_subset]
#     # 3. Node labeling
#     dist_h = bfs_shortest_dist(edge_index, h, num_nodes)[filtered_nodes]
#     dist_t = bfs_shortest_dist(edge_index, t, num_nodes)[filtered_nodes]
#     node_label = torch.stack([dist_h, dist_t], dim=1)
#     # 4. Subgraph (edge_index/edge_type) cho node đã lọc
#     #sub_edge_index, edge_mask = subgraph(filtered_nodes, edge_index, relabel_nodes=True, return_edge_mask=True)
#     sub_edge_index, sub_edge_type, edge_mask = subgraph(
#         filtered_nodes, edge_index, edge_attr=edge_type,
#         relabel_nodes=True, return_edge_mask=True
#     )
#
#     sub_edge_type = edge_type[edge_mask]
#     return filtered_nodes, sub_edge_index, sub_edge_type, node_label

def sample_neg(edges, num_nodes, params):
    num_negs = params.num_negs
    neg_edges = []
    for src, dst in edges:
        negs = []
        for _ in range(num_negs):
            neg_dst = np.random.randint(0, num_nodes)
            negs.append((src, neg_dst))
        neg_edges.append(negs)
    return edges, neg_edges

