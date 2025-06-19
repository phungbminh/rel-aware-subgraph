import struct
import logging
from tqdm import tqdm
import lmdb

import cudf, cupy as cp
import cugraph
from cupyx.scipy.sparse import csr_matrix as cupy_csr

from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask import config
import dask_cudf

from utils.graph_utils import serialize, get_edge_count
config.set({'distributed.dashboard.enabled': False})
# Khởi Dask-cuGraph cho 2 GPU
# cluster = LocalCUDACluster(n_workers=2, dashboard_address=None, scheduler_port=0 )
# client = Client(cluster)

# Chuyển mỗi adjacency list sang Graph trên GPU
gpu_graphs = None

def build_gpu_graphs(adj_list):
    gpu_graphs = []
    for A in adj_list:
        A_gpu = cupy_csr(A)  # SciPy CSR -> CuPy CSR
        src, dst = A_gpu.nonzero()
        df = cudf.DataFrame({'src': src, 'dst': dst})
        G = cugraph.Graph(directed=True)
        G.from_cudf_edgelist(df, source='src', destination='dst')
        gpu_graphs.append(G)
    return gpu_graphs


def init_gpu_graphs(adj_list):
    global gpu_graphs
    gpu_graphs = build_gpu_graphs(adj_list)


# GPU-negative sampling
def sample_neg(adj_list, edges, num_neg_samples_per_link=1, max_size=1_000_000, constrained_neg_prob=0.0):
    #edges_gpu = cp.asarray(edges, dtype=cp.int32)
    global gpu_graphs
    if gpu_graphs is None:
        gpu_graphs = build_gpu_graphs(adj_list)
    edges_gpu = cp.asarray(edges, dtype=cp.int32)
    M = edges_gpu.shape[0]
    if max_size < M:
        perm = cp.random.permutation(M)[:max_size]
        edges_gpu = edges_gpu[perm]
        M = edges_gpu.shape[0]

    # distribution relation trên CPU (nhỏ)
    edge_count = get_edge_count(adj_list)  # numpy array
    θ = 0.001
    rel_dist = cp.asarray(edge_count, dtype=cp.float32)
    idx = edge_count.nonzero()
    rel_dist[idx] = cp.exp(θ * cp.asarray(edge_count[idx]))
    rel_dist = rel_dist / rel_dist.sum()

    # valid heads/tails trên GPU
    valid_heads = [cp.asarray(A.tocoo().row, dtype=cp.int32) for A in adj_list]
    valid_tails = [cp.asarray(A.tocoo().col, dtype=cp.int32) for A in adj_list]

    neg = []
    pbar = tqdm(total=M * num_neg_samples_per_link)
    i = 0
    n = adj_list[0].shape[0]

    while len(neg) < M * num_neg_samples_per_link:
        h, t, r = int(edges_gpu[i, 0]), int(edges_gpu[i, 1]), int(edges_gpu[i, 2])
        if cp.random.rand() < constrained_neg_prob:
            if cp.random.rand() < 0.5:
                h = int(cp.random.choice(valid_heads[r], size=1)[0])
            else:
                t = int(cp.random.choice(valid_heads[r], size=1)[0])
        else:
            if cp.random.rand() < 0.5:
                h = int(cp.random.randint(0, n))
            else:
                t = int(cp.random.randint(0, n))

        # kiểm tra cạnh tồn tại trên GPU graph
        if h != t and not gpu_graphs[r].has_edge(h, t):
            neg.append((h, t, r))
            pbar.update(1)
        i = (i + 1) % M

    pbar.close()
    neg = cp.asnumpy(cp.array(neg, dtype=cp.int32))
    return cp.asnumpy(edges_gpu), neg

# Hàm extract + label subgraph sử dụng cuGraph ego_graph
def extract_partition(df_part, hop, enclosing, max_label_value):
    import cudf, cupy as cp, cugraph
    results = []
    for row in df_part.itertuples():
        u, v, r = int(row.h), int(row.t), int(row.r)
        G = gpu_graphs[r]
        sg = cugraph.ego_graph(G, u, radius=hop, center=True)

        # Chuyển adjacency ego sang định dạng scipy để tái sử dụng node_label
        src = sg.edgelist.edgelist_df['src'].to_pandas().values
        dst = sg.edgelist.edgelist_df['dst'].to_pandas().values
        # build scipy sparse adjacency
        from scipy.sparse import csr_matrix
        n_nodes = int(sg.number_of_vertices())
        A_sub = csr_matrix((np.ones(len(src)), (src, dst)), shape=(n_nodes, n_nodes))

        # gọi node_label gốc (CPU) trên A_incidence từ incidence_matrix
        labels, idxs = node_label(cugraph.utilities.convert_from_sparse(A_sub), max_distance=hop)
        if max_label_value is not None:
            labels = cp.minimum(cp.asarray(labels), max_label_value).get()

        datum = {
            'nodes': sg.nodes().to_array()[idxs].to_pandas().tolist(),
            'n_labels': labels.tolist(),
            'subgraph_size': int(len(idxs)),
            'enc_ratio': float(len(idxs) / (A_sub.shape[0] + 1e-6)),
            'num_pruned_nodes': int(A_sub.shape[0] - len(idxs))
        }
        str_id = f"{row.Index:08d}".encode('ascii')
        results.append({'id': str_id, 'datum': serialize(datum)})
    return cudf.DataFrame(results)


# Giữ nguyên cấu trúc cũ, chỉ dispatch qua Dask
def links2subgraphs(adj_list, graphs, params, max_label_value=None):
    # init GPU graphs
    init_gpu_graphs(adj_list)

    # build DataFrame links
    all_rows = []
    for split_name, split in graphs.items():
        for sign in ['pos', 'neg']:
            g_label = 1 if sign == 'pos' else 0
            for (h, t, r) in split[sign]:
                all_rows.append((h, t, r, split_name))
    df = cudf.DataFrame(all_rows, columns=['h', 't', 'r', 'split'])
    ddf = dask_cudf.from_cudf(df, npartitions=2)

    # submit extract_partition lên Dask
    parts = ddf.map_partitions(
        lambda dfp: extract_partition(dfp, params.hop, params.enclosing_sub_graph, max_label_value)
    ).compute()

    # ghi LMDB theo split, giữ nguyên logic serialize
    env = lmdb.open(params.db_path, map_size=params.map_size, max_dbs=6)
    for pdf in parts.to_pandas().itertuples():
        split = pdf.datum  # split name
        db = env.open_db(split.encode())
        with env.begin(write=True, db=db) as txn:
            txn.put(pdf.id, pdf.datum)

# import struct
# import logging
# from tqdm import tqdm
# import lmdb
# import multiprocessing as mp
# import numpy as np
# import scipy.sparse as ssp
# from scipy.special import softmax
# from utils.dgl_utils import _bfs_relational
# from utils.graph_utils import incidence_matrix, remove_nodes, serialize, get_edge_count
#
#
#
# def sample_neg(adj_list, edges, num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0):
#     pos_edges = edges
#     neg_edges = []
#
#     # if max_size is set, randomly sample train links
#     if max_size < len(pos_edges):
#         perm = np.random.permutation(len(pos_edges))[:max_size]
#         pos_edges = pos_edges[perm]
#
#     # sample negative links for train/test
#     n, r = adj_list[0].shape[0], len(adj_list)
#
#     # distribution of edges across reelations
#     theta = 0.001
#     edge_count = get_edge_count(adj_list)
#     rel_dist = np.zeros(edge_count.shape)
#     idx = np.nonzero(edge_count)
#     rel_dist[idx] = softmax(theta * edge_count[idx])
#
#     # possible head and tails for each relation
#     valid_heads = [adj.tocoo().row.tolist() for adj in adj_list]
#     valid_tails = [adj.tocoo().col.tolist() for adj in adj_list]
#
#     pbar = tqdm(total=len(pos_edges))
#     while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
#         neg_head, neg_tail, rel = pos_edges[pbar.n % len(pos_edges)][0], pos_edges[pbar.n % len(pos_edges)][1], pos_edges[pbar.n % len(pos_edges)][2]
#         if np.random.uniform() < constrained_neg_prob:
#             if np.random.uniform() < 0.5:
#                 neg_head = np.random.choice(valid_heads[rel])
#             else:
#                 neg_tail = np.random.choice(valid_tails[rel])
#         else:
#             if np.random.uniform() < 0.5:
#                 neg_head = np.random.choice(n)
#             else:
#                 neg_tail = np.random.choice(n)
#
#         if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
#             neg_edges.append([neg_head, neg_tail, rel])
#             pbar.update(1)
#
#     pbar.close()
#
#     neg_edges = np.array(neg_edges)
#     return pos_edges, neg_edges
#
#
# def links2subgraphs(A, graphs, params, max_label_value=None):
#     '''
#     extract enclosing subgraphs, write map mode + named dbs
#     '''
#     max_n_label = {'value': np.array([0, 0])}
#     subgraph_sizes = []
#     enc_ratios = []
#     num_pruned_nodes = []
#
#     BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['pos'], A, params) * 1.5
#     links_length = 0
#     for split_name, split in graphs.items():
#         links_length += (len(split['pos']) + len(split['neg'])) * 2
#     map_size = links_length * BYTES_PER_DATUM
#     map_size = int(map_size)
#     print("Opening LMDB at", params.db_path, "with map_size =", map_size)
#
#     env = lmdb.open(params.db_path, map_size=map_size, max_dbs=6)
#
#     def extraction_helper(A, links, g_labels, split_env):
#
#         with env.begin(write=True, db=split_env) as txn:
#             txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))
#         n = mp.cpu_count()
#         print("Cores:", n)
#         with mp.Pool(processes=None, initializer=intialize_worker, initargs=(A, params, max_label_value)) as p:
#             args_ = zip(range(len(links)), links, g_labels)
#             for (str_id, datum) in tqdm(p.imap(extract_save_subgraph, args_), total=len(links)):
#                 max_n_label['value'] = np.maximum(np.max(datum['n_labels'], axis=0), max_n_label['value'])
#                 subgraph_sizes.append(datum['subgraph_size'])
#                 enc_ratios.append(datum['enc_ratio'])
#                 num_pruned_nodes.append(datum['num_pruned_nodes'])
#
#                 with env.begin(write=True, db=split_env) as txn:
#                     #txn.put(str_id, pickle.dumps(datum, protocol=4))
#                     txn.put(str_id, serialize(datum))
#
#     for split_name, split in graphs.items():
#         print(f"Extracting enclosing subgraphs for positive links in {split_name} set")
#         labels = np.ones(len(split['pos']))
#         db_name_pos = split_name + '_pos'
#         split_env = env.open_db(db_name_pos.encode())
#         extraction_helper(A, split['pos'], labels, split_env)
#
#         print(f"Extracting enclosing subgraphs for negative links in {split_name} set")
#         labels = np.zeros(len(split['neg']))
#         db_name_neg = split_name + '_neg'
#         split_env = env.open_db(db_name_neg.encode())
#         extraction_helper(A, split['neg'], labels, split_env)
#
#     max_n_label['value'] = max_label_value if max_label_value is not None else max_n_label['value']
#
#     with env.begin(write=True) as txn:
#         bit_len_label_sub = int.bit_length(int(max_n_label['value'][0]))
#         bit_len_label_obj = int.bit_length(int(max_n_label['value'][1]))
#         txn.put('max_n_label_sub'.encode(), (int(max_n_label['value'][0])).to_bytes(bit_len_label_sub, byteorder='little'))
#         txn.put('max_n_label_obj'.encode(), (int(max_n_label['value'][1])).to_bytes(bit_len_label_obj, byteorder='little'))
#
#         txn.put('avg_subgraph_size'.encode(), struct.pack('f', float(np.mean(subgraph_sizes))))
#         txn.put('min_subgraph_size'.encode(), struct.pack('f', float(np.min(subgraph_sizes))))
#         txn.put('max_subgraph_size'.encode(), struct.pack('f', float(np.max(subgraph_sizes))))
#         txn.put('std_subgraph_size'.encode(), struct.pack('f', float(np.std(subgraph_sizes))))
#
#         txn.put('avg_enc_ratio'.encode(), struct.pack('f', float(np.mean(enc_ratios))))
#         txn.put('min_enc_ratio'.encode(), struct.pack('f', float(np.min(enc_ratios))))
#         txn.put('max_enc_ratio'.encode(), struct.pack('f', float(np.max(enc_ratios))))
#         txn.put('std_enc_ratio'.encode(), struct.pack('f', float(np.std(enc_ratios))))
#
#         txn.put('avg_num_pruned_nodes'.encode(), struct.pack('f', float(np.mean(num_pruned_nodes))))
#         txn.put('min_num_pruned_nodes'.encode(), struct.pack('f', float(np.min(num_pruned_nodes))))
#         txn.put('max_num_pruned_nodes'.encode(), struct.pack('f', float(np.max(num_pruned_nodes))))
#         txn.put('std_num_pruned_nodes'.encode(), struct.pack('f', float(np.std(num_pruned_nodes))))
#
#
# def get_average_subgraph_size(sample_size, links, A, params):
#     total_size = 0
#     for (n1, n2, r_label) in links[np.random.choice(len(links), sample_size)]:
#         nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A, params.hop, params.enclosing_sub_graph, params.max_nodes_per_hop)
#         datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
#         total_size += len(serialize(datum))
#     return total_size / sample_size
#
#
# def intialize_worker(A, params, max_label_value):
#     global A_, params_, max_label_value_
#     A_, params_, max_label_value_ = A, params, max_label_value
#
#
# def extract_save_subgraph(args_):
#     idx, (n1, n2, r_label), g_label = args_
#     nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A_, params_.hop, params_.enclosing_sub_graph, params_.max_nodes_per_hop)
#
#     # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
#     if max_label_value_ is not None:
#         n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in n_labels])
#
#     datum = {'nodes': nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
#     str_id = '{:08}'.format(idx).encode('ascii')
#
#     return (str_id, datum)
#
#
# def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
#     bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
#     lvls = list()
#     for _ in range(h):
#         try:
#             lvls.append(next(bfs_generator))
#         except StopIteration:
#             pass
#     return set().union(*lvls)
#
#
# def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_node_label_value=None):
#     # extract the h-hop enclosing subgraphs around link 'ind'
#     A_incidence = incidence_matrix(A_list)
#     A_incidence += A_incidence.T
#
#     root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
#     root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)
#
#     subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
#     subgraph_nei_nodes_un = root1_nei.union(root2_nei)
#
#     # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
#     if enclosing_sub_graph:
#         subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
#     else:
#         subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)
#
#     subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]
#
#     labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h)
#
#     pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
#     pruned_labels = labels[enclosing_subgraph_nodes]
#     # pruned_subgraph_nodes = subgraph_nodes
#     # pruned_labels = labels
#
#     if max_node_label_value is not None:
#         pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])
#
#     subgraph_size = len(pruned_subgraph_nodes)
#     enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
#     num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)
#
#     return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes
#
#
# def node_label(subgraph, max_distance=1):
#     # implementation of the node labeling scheme described in the paper
#     roots = [0, 1]
#     sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
#     dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
#     dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)
#
#     target_node_labels = np.array([[0, 1], [1, 0]])
#     labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels
#
#     enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
#     return labels, enclosing_subgraph_nodes
