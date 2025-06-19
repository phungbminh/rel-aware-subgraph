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


def sample_neg(adj_list, edges,
               num_neg_samples_per_link=1,
               max_size=1_000_000,
               constrained_neg_prob=0.0):
    """
    Generate negative edge samples for each positive link.

    Args:
        adj_list: list of scipy.sparse adjacency matrices, one per relation.
        edges: array of shape (num_pos, 3) with (head, tail, rel).
        num_neg_samples_per_link: number of negative samples per positive link.
        max_size: if >0 and less than num_edges, randomly subsample positives.
        constrained_neg_prob: probability to sample from same-head/tail distribution.

    Returns:
        pos_edges: (M,3) array of positive edges used.
        neg_edges: (M*num_neg_samples_per_link,3) array of negative samples.
    """
    pos_edges = edges.copy()
    if max_size > 0 and max_size < len(pos_edges):
        perm = np.random.permutation(len(pos_edges))[:max_size]
        pos_edges = pos_edges[perm]

    n_nodes = adj_list[0].shape[0]
    # relation distribution (unused here but kept for extensibility)
    theta = 0.001
    edge_count = get_edge_count(adj_list)
    rel_dist = np.zeros_like(edge_count, dtype=float)
    idx = np.nonzero(edge_count)
    rel_dist[idx] = softmax(theta * edge_count[idx])

    valid_heads = [adj.tocoo().row.tolist() for adj in adj_list]
    valid_tails = [adj.tocoo().col.tolist() for adj in adj_list]

    neg_edges = []
    pbar = tqdm(total=len(pos_edges), desc="Sampling negatives")
    while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
        i = pbar.n % len(pos_edges)
        h, t, r = pos_edges[i]
        if np.random.rand() < constrained_neg_prob:
            if np.random.rand() < 0.5:
                h = np.random.choice(valid_heads[r])
            else:
                t = np.random.choice(valid_tails[r])
        else:
            if np.random.rand() < 0.5:
                h = np.random.randint(n_nodes)
            else:
                t = np.random.randint(n_nodes)

        if h != t and adj_list[r][h, t] == 0:
            neg_edges.append([h, t, r])
            pbar.update(1)
    pbar.close()

    return pos_edges, np.array(neg_edges, dtype=int)


def estimate_avg_size(sample_size, links, adj_list, params):
    """
    Estimate the average serialized size (in bytes) of a subgraph.
    """
    total = 0
    for n1, n2, r in links[np.random.choice(len(links), sample_size)]:
        nodes, labels, size, enc_ratio, pruned = extract_and_label((n1, n2), r,
                                                                     adj_list,
                                                                     params.hop,
                                                                     params.enclosing_sub_graph,
                                                                     params.max_nodes_per_hop)
        datum = {'nodes': nodes,
                 'r_label': r,
                 'g_label': 0,
                 'n_labels': labels,
                 'subgraph_size': size,
                 'enc_ratio': enc_ratio,
                 'num_pruned_nodes': pruned}
        total += len(serialize(datum))
    return total / sample_size


def _init_worker(adj_list, params, max_label_val):
    global _A, _params, _max_label_val
    _A = adj_list
    _params = params
    _max_label_val = max_label_val


def _extract_serialize(args):
    idx, ((n1, n2, r), g_lbl) = args
    nodes, labels, size, enc_ratio, pruned = extract_and_label((int(n1), int(n2)), int(r),
                                                                _A,
                                                                _params.hop,
                                                                _params.enclosing_sub_graph,
                                                                _params.max_nodes_per_hop,
                                                                _max_label_val)
    if _max_label_val is not None:
        labels = np.minimum(labels, _max_label_val)
    datum = {
        'nodes': nodes,
        'r_label': r,
        'g_label': g_lbl,
        'n_labels': labels,
        'subgraph_size': size,
        'enc_ratio': enc_ratio,
        'num_pruned_nodes': pruned
    }
    sid = f"{idx:08}".encode('ascii')
    return sid, datum


def links2subgraphs(adj_list, graph_splits, params, max_label_val=None, use_cache=True):
    """
    Extract & cache h-hop subgraphs into LMDB under splits 'train','valid','test'.

    Args:
      adj_list: list of scipy.sparse adjacency per relation.
      graph_splits: dict of splits with keys 'train','valid','test', each a dict with:
                    'pos': np.array of pos edges,
                    'max_size': int max positives to sample per split,
                    'neg': will be set inside,
                    (and optionally existing 'neg').
      params: attributes db_path, hop, enclosing_sub_graph,
              max_nodes_per_hop, num_neg_samples_per_link,
              constrained_neg_prob.
      max_label_val: cap for node labels.
      use_cache: skip if LMDB exists.
    """
    # 1) Negative sampling for all splits
    for split_name, info in graph_splits.items():
        pos_edges = info['pos']
        _, neg_edges = sample_neg(
            adj_list=adj_list,
            edges=pos_edges,
            num_neg_samples_per_link=params.num_neg_samples_per_link,
            max_size=info.get('max_size', 0),
            constrained_neg_prob=params.constrained_neg_prob
        )
        info['neg'] = neg_edges

    # 2) Cache check
    if use_cache and os.path.isdir(params.db_path):
        try:
            env = lmdb.open(params.db_path, readonly=True, lock=False, max_dbs=6)
            db = env.open_db(b'train_pos')
            with env.begin(db=db) as txn:
                cnt = int.from_bytes(txn.get(b'num_graphs'), byteorder='little')
            logger.info(f"[CACHE] {cnt} train_pos found, skip.")
            return
        except:
            logger.info("[CACHE] invalid LMDB, rebuilding...")

    # 3) Prepare LMDB
    if os.path.isdir(params.db_path): os.system(f"rm -rf {params.db_path}")
    os.makedirs(params.db_path, exist_ok=True)
    size_est = estimate_avg_size(min(100, len(graph_splits['train']['pos'])),
                                  graph_splits['train']['pos'],
                                  adj_list, params) * 1.5
    total_links = sum(len(graph_splits[s]['pos']) + len(graph_splits[s]['neg']) for s in graph_splits)
    env = lmdb.open(params.db_path,
                    map_size=int(total_links * size_est),
                    max_dbs=6)

    # 4) Extract & store helper
    def _store(name, links, labels):
        db = env.open_db(name.encode())
        with env.begin(write=True, db=db) as txn:
            txn.put(b'num_graphs', len(links).to_bytes(len(links).bit_length(), 'little'))
        args = zip(range(len(links)), zip(links, labels))
        with mp.Pool(initializer=_init_worker,
                     initargs=(adj_list, params, max_label_val)) as pool:
            for sid, datum in tqdm(pool.imap(_extract_serialize, args),
                                   total=len(links), desc=name):
                with env.begin(write=True, db=db) as txn:
                    txn.put(sid, serialize(datum))

    # 5) Run per split
    for split in ['train','valid','test']:
        _store(f"{split}_pos", graph_splits[split]['pos'],
               np.ones(len(graph_splits[split]['pos']), dtype=np.int8))
        _store(f"{split}_neg", graph_splits[split]['neg'],
               np.zeros(len(graph_splits[split]['neg']), dtype=np.int8))

    logger.info(f"✅ LMDB at {params.db_path}" )


def get_hop_neighbors(roots, inc_mat, h=1, max_p=None):
    bfs = _bfs_relational(inc_mat, roots, max_p)
    levels = []
    for _ in range(h):
        try: levels.append(next(bfs))
        except StopIteration: break
    return set().union(*levels)


def extract_and_label(ind, rel, adj_list,
                      h=1, enclosing=True,
                      max_p=None,
                      max_label=None):
    inc = incidence_matrix(adj_list)
    inc += inc.T
    n1, n2 = ind
    nei1 = get_hop_neighbors({n1}, inc, h, max_p)
    nei2 = get_hop_neighbors({n2}, inc, h, max_p)
    core = (nei1 & nei2) if enclosing else (nei1 | nei2)
    all_nodes = [n1, n2] + list(core - {n1,n2})
    sub_adjs = [adj[all_nodes,:][:,all_nodes] for adj in adj_list]
    lab, keep = node_label(incidence_matrix(sub_adjs), max_distance=h)
    pr_nodes = np.array(all_nodes)[keep].tolist()
    pr_labels = lab[keep]
    if max_label is not None: pr_labels = np.minimum(pr_labels, max_label)
    size = len(pr_nodes)
    enc = len(nei1 & nei2) / (len(nei1 | nei2) + 1e-6)
    pruned = len(all_nodes) - size
    return pr_nodes, pr_labels, size, enc, pruned


def node_label(incidence, max_distance=1):
    roots=[0,1]
    sgs=[remove_nodes(incidence,[r]) for r in roots]
    dists=[np.clip(ssp.csgraph.dijkstra(sg,indices=[0],directed=False,unweighted=True,limit=1e7)[:,1:],0,1e7) for sg in sgs]
    dp=np.stack((dists[0][0],dists[1][0]),axis=1).astype(int)
    base=np.array([[0,1],[1,0]])
    lab=np.vstack((base,dp)) if dp.size else base
    keep_idx=np.where(lab.max(axis=1)<=max_distance)[0]
    return lab, keep_idx



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
