#!/usr/bin/env python3
import os
import struct
import logging
from tqdm import tqdm
import lmdb
import multiprocessing as mp

import numpy as np
import scipy.sparse as ssp
from scipy.special import softmax
import os
from utils.dgl_utils import _bfs_relational
from utils.graph_utils import incidence_matrix, remove_nodes, serialize, get_edge_count

# Configure logger
logger = logging.getLogger(__name__)

import os
import struct
import logging
from tqdm import tqdm
import lmdb

import numpy as np
import scipy.sparse as ssp
from scipy.special import softmax

import dgl
import torch

from utils.graph_utils import serialize, get_edge_count, incidence_matrix, remove_nodes
from utils.dgl_utils import _bfs_relational

# Logger
logger = logging.getLogger(__name__)


def sample_neg(adj_list, edges,
               num_neg_samples_per_link=1,
               max_size=1_000_000,
               constrained_neg_prob=0.0):
    """
    CPU-based negative sampling for training split only.
    """
    pos = edges.copy()
    if 0 < max_size < len(pos):
        perm = np.random.permutation(len(pos))[:max_size]
        pos = pos[perm]

    n_nodes = adj_list[0].shape[0]
    neg = []
    for (h, t, r) in tqdm(pos, desc="Sampling negatives", leave=False):
        for _ in range(num_neg_samples_per_link):
            if np.random.rand() < constrained_neg_prob:
                if np.random.rand() < 0.5:
                    h = np.random.choice(adj_list[r].tocoo().row)
                else:
                    t = np.random.choice(adj_list[r].tocoo().col)
            else:
                if np.random.rand() < 0.5:
                    h = np.random.randint(n_nodes)
                else:
                    t = np.random.randint(n_nodes)
            if h != t and adj_list[r][h, t] == 0:
                neg.append([h, t, r])
    return pos, np.array(neg, dtype=int)


def build_gpu_graph(adj_list):
    """
    Construct a single DGLGraph on GPU from per-relation SciPy matrices.
    """
    srcs, dsts, rels = [], [], []
    for r, A in enumerate(adj_list):
        coo = A.tocoo()
        srcs.append(torch.from_numpy(coo.row).cuda())
        dsts.append(torch.from_numpy(coo.col).cuda())
        rels.append(torch.full((coo.nnz,), r, dtype=torch.long).cuda())
    src = torch.cat(srcs)
    dst = torch.cat(dsts)
    etype = torch.cat(rels)
    g = dgl.heterograph({('node','rel','node'): (src, dst)},
                         num_nodes_dict={'node': adj_list[0].shape[0]})
    g.edata['rel_type'] = etype
    return g


def extract_subgraph_gpu(g, u, v, h, enclosing=True):
    """
    GPU-based h-hop enclosing/union subgraph extraction.
    Returns node list (on CPU) and GPU subgraph.
    """
    # Ensure seed indices are int32 for DGL
    device = g.device
    src_seeds = torch.tensor([u], dtype=torch.int32, device=device)
    dst_seeds = torch.tensor([v], dtype=torch.int32, device=device)
    # Perform k-hop subgraph extraction on GPU
    sg_u, nid_u = dgl.khop_in_subgraph(g, src_seeds, h)
    sg_v, nid_v = dgl.khop_in_subgraph(g, dst_seeds, h)
    set_u, set_v = set(nid_u.tolist()), set(nid_v.tolist())
    core = (set_u & set_v) if enclosing else (set_u | set_v)
    # Assemble node list on CPU
    nodes = [u, v] + [n for n in core if n not in (u, v)]
    # Convert to int32 tensor on GPU for node_subgraph
    nodes_tensor = torch.tensor(nodes, dtype=torch.int32, device=device)
    subg = dgl.node_subgraph(g, nodes_tensor)
    return nodes, subg



def extract_and_label(nodes, adj_list, params, max_label=None):
    """
    CPU-based node labeling on pruned subgraph nodes.
    """
    inc = incidence_matrix(adj_list)
    inc += inc.T
    roots = nodes[:2]
    lvl_sets = []
    bfs = _bfs_relational(inc, set(roots), params.max_nodes_per_hop)
    for _ in range(params.hop):
        try:
            lvl_sets.append(next(bfs))
        except StopIteration:
            break
    keep = [i for i, n in enumerate(nodes) if n in set().union(*lvl_sets)]

    sg = inc[keep][:, keep]
    dist1 = ssp.csgraph.dijkstra(remove_nodes(sg, [0]), indices=[0])[0]
    dist2 = ssp.csgraph.dijkstra(remove_nodes(sg, [1]), indices=[0])[0]
    labels = np.vstack((np.zeros((2,2), dtype=int),
                        np.stack((dist1[1:], dist2[1:]), axis=1).astype(int)))
    if max_label is not None:
        labels = np.minimum(labels, max_label)
    pr_nodes = [nodes[i] for i in keep]
    enc = len(set(nodes) & set(lvl_sets[0])) / (len(set().union(*lvl_sets)) + 1e-6)
    return pr_nodes, labels, len(pr_nodes), enc, len(nodes) - len(pr_nodes)


def links2subgraphs(adj_list, graphs, params, max_label=None, use_cache=True):
    """
    Full pipeline: negative sampling (train only), GPU build, GPU extract, CPU label, LMDB store.
    """
    # 1) Negative sampling only for train
    pos, neg = sample_neg(adj_list,
                           graphs['train']['pos'],
                           params.num_neg_samples_per_link,
                           graphs['train'].get('max_size', 0),
                           params.constrained_neg_prob)
    graphs['train']['pos'], graphs['train']['neg'] = pos, neg

    # 2) Check cache
    if use_cache and os.path.isdir(params.db_path):
        try:
            env = lmdb.open(params.db_path, readonly=True, lock=False, max_dbs=1)
            db = env.open_db(b'train_pos')
            with env.begin(db=db) as txn:
                cnt = int.from_bytes(txn.get(b'num_graphs'), 'little')
            logger.info(f"[CACHE] found {cnt}, skip extraction.")
            return
        except:
            pass

    # 3) Prepare LMDB
    if os.path.isdir(params.db_path): os.system(f"rm -rf {params.db_path}")
    os.makedirs(params.db_path, exist_ok=True)
    env = lmdb.open(params.db_path, map_size=int(1e9), max_dbs=1)
    # Build graph on GPU
    g_gpu = build_gpu_graph(adj_list)

    # 4) Extract & store
    db = env.open_db(b'train_pos')
    with env.begin(write=True, db=db) as txn:
        txn.put(b'num_graphs', len(graphs['train']['pos']).to_bytes(4, 'little'))
    for idx, (h, t, r) in enumerate(tqdm(graphs['train']['pos'], desc='train_pos')):
        # GPU extraction
        nodes, subg = extract_subgraph_gpu(g_gpu, int(h), int(t), params.hop, params.enclosing_sub_graph)
        # CPU labeling
        pr_nodes, labels, sz, enc, pruned = extract_and_label(nodes, adj_list, params, max_label)
        datum = {
            'nodes': pr_nodes,
            'r_label': int(r),
            'g_label': 1,
            'n_labels': labels,
            'subgraph_size': sz,
            'enc_ratio': enc,
            'num_pruned_nodes': pruned
        }
        sid = f"{idx:08}".encode()
        with env.begin(write=True, db=db) as txn:
            txn.put(sid, serialize(datum))

    logger.info("✅ Done: GPU used for subgraph extraction.")

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
