import cudf
import cugraph
import cupy as cp
import torch
import numpy as np

def edge_index_to_cudf(edge_index, edge_type):
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    rel = edge_type.cpu().numpy()
    df = cudf.DataFrame({
        'src': src,
        'dst': dst,
        'etype': rel
    })
    return df

def build_cugraph(edge_index, edge_type):
    df_edges = edge_index_to_cudf(edge_index, edge_type)
    G = cugraph.MultiGraph()
    G.from_cudf_edgelist(df_edges, source='src', destination='dst', edge_attr='etype')
    return G

def cugraph_k_hop(G, nodes, k):
    all_nodes = []
    for node in nodes:
        bfs_result = cugraph.bfs(G, start=int(node), depth_limit=k)
        neighbors = bfs_result['vertex'].values_host
        all_nodes.append(neighbors)
    all_nodes = np.unique(np.concatenate(all_nodes))
    return all_nodes.astype(np.int64)

def filter_by_relation_tau(G, sub_nodes, rel, tau):
    df = G.view_edge_list()
    mask = (df['etype'] == rel) & df['src'].isin(sub_nodes)
    rel_counts = df[mask].groupby('src').size().reset_index(name='count')
    keep_nodes = rel_counts[rel_counts['count'] >= tau]['src'].values_host
    return keep_nodes.astype(np.int64)

def extract_cugraph_subgraph(G, node_ids):
    df = G.view_edge_list()
    mask = df['src'].isin(node_ids) & df['dst'].isin(node_ids)
    sub_edges = df[mask]
    old2new = {int(n): i for i, n in enumerate(node_ids)}
    src = sub_edges['src'].to_numpy()
    dst = sub_edges['dst'].to_numpy()
    etype = sub_edges['etype'].to_numpy()
    src_new = np.array([old2new[int(x)] for x in src])
    dst_new = np.array([old2new[int(x)] for x in dst])
    edge_index = torch.tensor([src_new, dst_new], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    return edge_index, edge_type, len(node_ids), old2new

def cugraph_shortest_dist(G, source, node_list):
    bfs_result = cugraph.bfs(G, start=int(source))
    dist_map = dict(zip(bfs_result['vertex'].values_host, bfs_result['distance'].values_host))
    # -1 cho unreachable node (hoặc dùng 999 để mask)
    return np.array([dist_map.get(int(n), -1) for n in node_list], dtype=np.int64)
