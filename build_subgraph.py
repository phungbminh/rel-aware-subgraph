import argparse
import os
import lmdb
import pickle
import numpy as np
import time
import json
from tqdm import tqdm
import logging
import multiprocessing as mp
from queue import Empty
from ogb.linkproppred import LinkPropPredDataset
from scipy import sparse
from numba import njit, prange
from utils import CSRGraph
from verify_subgraph import lmdb_summary

# ======= Logger chuẩn hóa ===========
def setup_logger(output_dir: str):
    logger = logging.getLogger('subgraph_extraction')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh = logging.FileHandler(os.path.join(output_dir, 'extraction.log'))
    fh.setFormatter(formatter)
    class ColorFormatter(logging.Formatter):
        COLORS = {
            logging.INFO: '\033[94m',
            logging.WARNING: '\033[93m',
            logging.ERROR: '\033[91m',
            logging.CRITICAL: '\033[95m'
        }
        RESET = '\033[0m'
        def format(self, record):
            msg = super().format(record)
            return f"{self.COLORS.get(record.levelno, '')}{msg}{self.RESET}"
    ch = logging.StreamHandler()
    ch.setFormatter(ColorFormatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ======= BFS song song với Numba ===========
@njit(nogil=True)
def numba_bfs(adj_indptr, adj_indices, start, max_hops):
    num_nodes = adj_indptr.shape[0] - 1
    distances = np.full(num_nodes, 127, dtype=np.int8)
    queue = np.empty(num_nodes, dtype=np.int32)
    start_ptr, end_ptr = 0, 0
    queue[end_ptr] = start
    end_ptr += 1
    distances[start] = 0
    while start_ptr < end_ptr:
        current = queue[start_ptr]
        start_ptr += 1
        current_dist = distances[current]
        if current_dist < max_hops:
            for idx in range(adj_indptr[current], adj_indptr[current + 1]):
                neighbor = adj_indices[idx]
                if distances[neighbor] > current_dist + 1:
                    distances[neighbor] = current_dist + 1
                    queue[end_ptr] = neighbor
                    end_ptr += 1
    return distances

@njit(nogil=True, parallel=True)
def batch_bfs(adj_indptr, adj_indices, start_nodes, max_hops):
    num_nodes = adj_indptr.shape[0] - 1
    num_starts = len(start_nodes)
    dist_matrix = np.full((num_starts, num_nodes), 127, dtype=np.int8)
    for i in prange(num_starts):
        dist_matrix[i] = numba_bfs(adj_indptr, adj_indices, start_nodes[i], max_hops)
    return dist_matrix

# ======= Tính bậc quan hệ dạng sparse ===========
def compute_relation_degree_sparse(triples, num_nodes, num_relations):
    """Sparse matrix: node x relation (degree count)"""
    rows = np.concatenate([triples[:, 0], triples[:, 2]])
    cols = np.concatenate([triples[:, 1], triples[:, 1]])
    data = np.ones(2 * len(triples), dtype=np.int32)
    rel_degree = sparse.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_relations), dtype=np.int32).tocsr()
    return rel_degree

# ======= WorkerContext chứa graph, degree, param ===========
class WorkerContext:
    __slots__ = ('csr_graph', 'rel_degree', 'rel_degree_dense', 'use_dense', 'k', 'tau')
    def __init__(self, csr_graph, rel_degree, rel_degree_dense, use_dense, k, tau):
        self.csr_graph = csr_graph
        self.rel_degree = rel_degree
        self.rel_degree_dense = rel_degree_dense
        self.use_dense = use_dense
        self.k = k
        self.tau = tau

# ======= Trích xuất subgraph từng batch, dùng getcol nhanh ===========
def extract_batch_subgraphs(triples_batch, ctx: WorkerContext):
    """Trích xuất subgraph k-hop cho từng triple. Truy vấn rel_degree theo cột (r), cực nhanh."""
    sources = triples_batch[:, 0].astype(np.int32)
    targets = triples_batch[:, 2].astype(np.int32)
    relations = triples_batch[:, 1].astype(np.int32)
    if sources.max() >= ctx.csr_graph.num_nodes or targets.max() >= ctx.csr_graph.num_nodes:
        raise ValueError("Node index out of bound in batch")
    src_dists = batch_bfs(ctx.csr_graph.indptr, ctx.csr_graph.indices, sources, ctx.k)
    tgt_dists = batch_bfs(ctx.csr_graph.indptr, ctx.csr_graph.indices, targets, ctx.k)
    results = []
    for i in range(len(triples_batch)):
        h, r, t = sources[i], relations[i], targets[i]
        in_range = (src_dists[i] <= ctx.k) | (tgt_dists[i] <= ctx.k)
        candidate_nodes = np.where(in_range)[0]
        if ctx.use_dense:
            # Dùng dense matrix (cực nhanh, tốn RAM nếu graph lớn)
            rel_counts = ctx.rel_degree_dense[candidate_nodes, r]
        else:
            # Dùng getcol để truy vấn cột (nhanh hơn nhiều so với getrow)
            col_r = ctx.rel_degree.getcol(r).toarray().ravel()
            rel_counts = col_r[candidate_nodes]
        valid_mask = rel_counts >= ctx.tau
        if not np.any(valid_mask):
            valid_mask = rel_counts > 0
        filtered_nodes = candidate_nodes[valid_mask]
        results.append({
            'triple': (int(h), int(r), int(t)),
            'nodes': filtered_nodes.tolist(),
            's_dist': src_dists[i, filtered_nodes].tolist(),
            't_dist': tgt_dists[i, filtered_nodes].tolist()
        })
    return results

def process_batch(args):
    idx, batch, ctx = args
    try:
        res = extract_batch_subgraphs(batch, ctx)
        return idx, res
    except Exception as e:
        # Không crash toàn bộ pipeline nếu 1 batch lỗi
        return idx, [{"error": str(e), "batch_idx": idx}]

# ======= Negative sampling: batch-wise, memory safe ===========
def negative_sample_batches(positive_triples, num_negatives, num_nodes, batch_size):
    """Sinh negatives từng batch nhỏ, memory safe"""
    n_pos = len(positive_triples)
    n_batches = (n_pos + batch_size - 1) // batch_size
    for i in range(n_batches):
        batch = positive_triples[i*batch_size : (i+1)*batch_size]
        negatives = []
        for h, r, t in batch:
            for _ in range(num_negatives):
                # Tail negative
                t_neg = np.random.randint(0, num_nodes)
                while t_neg == t:
                    t_neg = np.random.randint(0, num_nodes)
                negatives.append([h, r, t_neg])
                # Head negative
                h_neg = np.random.randint(0, num_nodes)
                while h_neg == h:
                    h_neg = np.random.randint(0, num_nodes)
                negatives.append([h_neg, r, t])
        yield np.array(negatives, dtype=np.int32)

# ======= Async writer: log đầy đủ, close queue chuẩn ===========
def async_writer(queue, output_path, total_items, logger, progress_key=b'_progress'):
    env = lmdb.open(output_path, map_size=1024 ** 4, max_dbs=1)
    db = env.open_db()
    count = 0
    txn = env.begin(write=True, db=db)
    try:
        while True:
            try:
                key_data = queue.get(timeout=30)
                if key_data is None:
                    logger.info("Writer received sentinel, exiting.")
                    break
                key, data = key_data
                txn.put(key, pickle.dumps(data))
                count += 1
                if count % 1000 == 0:
                    txn.put(progress_key, str(count).encode())
                    txn.commit()
                    txn = env.begin(write=True, db=db)
            except Empty:
                logger.info("Writer queue is empty, exiting.")
                break
            except Exception as e:
                logger.error(f"Writer error: {str(e)}")
                break
        txn.put(progress_key, b'COMPLETED')
        txn.commit()
        logger.info(f"Writer process finished for {output_path}. Wrote {count} items.")
    finally:
        env.close()


# ======= parallel_extraction: log mỗi vài batch, close queue ===========
def parallel_extraction(
        triples: np.ndarray,
        csr_graph: CSRGraph,
        rel_degree: sparse.csr_matrix,
        rel_degree_dense: np.ndarray,
        use_dense: bool,
        k: int,
        tau: int,
        num_workers: int,
        output_path: str,
        batch_size: int = 1000,
        logger: logging.Logger = None
):
    ctx = WorkerContext(csr_graph, rel_degree, rel_degree_dense, use_dense, k, tau)
    num_batches = (len(triples) + batch_size - 1) // batch_size
    batches = [
        (i, triples[i * batch_size: (i + 1) * batch_size], ctx)
        for i in range(num_batches)
    ]
    writer_queue = mp.Queue(maxsize=10000)
    writer_process = mp.Process(
        target=async_writer,
        args=(writer_queue, output_path, len(triples), logger)
    )
    writer_process.start()
    completed = 0
    start_time = time.time()
    with mp.Pool(num_workers) as pool:
        results = pool.imap_unordered(process_batch, batches)
        with tqdm(total=len(triples), desc="Extracting subgraphs") as pbar:
            for idx, batch_results in results:
                for j, res in enumerate(batch_results):
                    global_idx = idx * batch_size + j
                    writer_queue.put((f"{global_idx:010d}".encode(), res))
                completed += len(batch_results)
                pbar.update(len(batch_results))
                if logger and (idx % 2 == 0 or idx == num_batches-1):  # Log mỗi 2 batch
                    elapsed = time.time() - start_time
                    speed = completed / elapsed if elapsed > 0 else 0
                    #logger.info(f"Processed batch {idx}/{num_batches} - Speed: {speed:.2f} triples/sec")
    writer_queue.put(None)  # Gửi sentinel (kết thúc writer)
    writer_process.join()
    writer_queue.close()
    writer_queue.join_thread()
    return completed

# ======= Lưu mapping file + metadata ===========
def save_mappings(output_dir, all_triples, args, split_sizes):
    output_dir = output_dir + "/mappings"
    os.makedirs(output_dir + "/mappings", exist_ok=True)
    entity_set = set(all_triples[:,0]) | set(all_triples[:,2])
    relation_set = set(all_triples[:,1])
    entity2id = {eid: i for i, eid in enumerate(sorted(entity_set))}
    relation2id = {rid: i for i, rid in enumerate(sorted(relation_set))}
    id2entity = {i: eid for eid, i in entity2id.items()}
    id2relation = {i: rid for rid, i in relation2id.items()}
    with open(os.path.join(output_dir, "entity2id.pkl"), "wb") as f:
        pickle.dump(entity2id, f)
    with open(os.path.join(output_dir, "relation2id.pkl"), "wb") as f:
        pickle.dump(relation2id, f)
    with open(os.path.join(output_dir, "id2entity.pkl"), "wb") as f:
        pickle.dump(id2entity, f)
    with open(os.path.join(output_dir, "id2relation.pkl"), "wb") as f:
        pickle.dump(id2relation, f)
    metadata = {
        "num_entities": len(entity2id),
        "num_relations": len(relation2id),
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "split_sizes": split_sizes,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved mapping and metadata files to {output_dir}")



# ======= Main pipeline tối ưu tốc độ + robust ===========
def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Subgraph Extraction Pipeline (Optimized)")
    parser.add_argument("--ogb-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-triples", type=int, default=None)
    parser.add_argument("--max-eval-triples", type=int, default=None)
    parser.add_argument("--num-negatives", type=int, default=2)
    parser.add_argument("--undirected", action="store_true", help="If set, build undirected graph (default: directed)")
    parser.add_argument("--rel-degree-dense", action="store_true", help="Convert relation degree to dense (faster, use more RAM)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)
    logger.info("Starting subgraph extraction pipeline")
    logger.info(f"Arguments: {vars(args)}")

    # Tải dữ liệu OGB
    logger.info("Loading OGB-BioKG dataset...")
    dataset = LinkPropPredDataset(name='ogbl-biokg', root=args.ogb_root)
    split_edge = dataset.get_edge_split()
    logger.info("Building full graph...")

    all_triples = []
    split_sizes = {}
    for split in ['train', 'valid', 'test']:
        edges = split_edge[split]
        triples = np.stack([edges['head'], edges['relation'], edges['tail']], axis=1).astype(np.int32)
        all_triples.append(triples)
        split_sizes[split] = int(len(triples))
    all_triples = np.concatenate(all_triples, axis=0)
    num_nodes = int(np.max(all_triples[:, [0, 2]]) + 1)
    num_relations = int(np.max(all_triples[:, 1]) + 1)

    save_mappings(args.output_dir, all_triples, args, split_sizes)

    # Đồ thị hướng hoặc không hướng
    if args.undirected:
        edge_sources = np.concatenate([all_triples[:, 0], all_triples[:, 2]])
        edge_targets = np.concatenate([all_triples[:, 2], all_triples[:, 0]])
    else:
        edge_sources = all_triples[:, 0]
        edge_targets = all_triples[:, 2]
    edge_index = np.vstack([edge_sources, edge_targets]).astype(np.int32)

    logger.info("Creating CSR graph...")
    csr_graph = CSRGraph(edge_index, num_nodes)
    with open(os.path.join(args.output_dir + "/mappings", "global_graph.pkl"), "wb") as f:
        pickle.dump(csr_graph, f)
    print(f"[INFO] Saved global graph to {os.path.join(args.output_dir, 'global_graph.pkl')}")

    logger.info("Computing relation degrees (sparse)...")
    rel_degree = compute_relation_degree_sparse(all_triples, num_nodes, num_relations)
    rel_degree_dense = rel_degree.toarray() if args.rel_degree_dense else None

    # Xử lý train
    logger.info("Processing training triples...")
    train_triples = np.stack([
        split_edge['train']['head'],
        split_edge['train']['relation'],
        split_edge['train']['tail']
    ], axis=1).astype(np.int32)
    if args.max_triples and len(train_triples) > args.max_triples:
        train_triples = train_triples[:args.max_triples]
    train_output = os.path.join(args.output_dir, "train.lmdb")
    train_count = parallel_extraction(
        train_triples, csr_graph, rel_degree, rel_degree_dense, args.rel_degree_dense,
        args.k, args.tau, args.num_workers, train_output, args.batch_size, logger
    )
    logger.info(f"Extracted {train_count} training subgraphs.")

    # Xử lý valid/test với negatives
    for split in ['valid', 'test']:
        logger.info(f"Processing {split} split...")
        split_triples = np.stack([
            split_edge[split]['head'],
            split_edge[split]['relation'],
            split_edge[split]['tail']
        ], axis=1).astype(np.int32)
        if args.max_eval_triples and len(split_triples) > args.max_eval_triples:
            split_triples = split_triples[:args.max_eval_triples]
        all_triples = [split_triples]
        for negs in negative_sample_batches(split_triples, args.num_negatives, num_nodes, args.batch_size):
            all_triples.append(negs)
        all_split_triples = np.vstack(all_triples)
        split_output = os.path.join(args.output_dir, f"{split}.lmdb")
        split_count = parallel_extraction(
            all_split_triples, csr_graph, rel_degree, rel_degree_dense, args.rel_degree_dense,
            args.k, args.tau, args.num_workers, split_output, args.batch_size, logger
        )
        logger.info(f"Extracted {split_count} subgraphs for {split}")

    logger.info("Pipeline finished successfully!")

    lmdb_summary(train_output, args.output_dir + "/mappings/global_graph.pkl", max_samples=10000)

if __name__ == "__main__":
    main()
