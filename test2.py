import lmdb, pickle
import numpy as np

def analyze_lmdb(db_path):
    env = lmdb.open(db_path, readonly=True, lock=False, max_dbs=2)
    db_pos = env.open_db(b'positive')
    with env.begin(db=db_pos) as txn:
        cursor = txn.cursor()
        node_counts = []
        edge_counts = []
        for key, value in cursor:
            subgraph = pickle.loads(value)
            nodes = subgraph['nodes']
            node_counts.append(len(nodes))
            # Nếu subgraph có lưu edges thì lấy, không thì skip
            if 'edges' in subgraph:
                edge_counts.append(len(subgraph['edges']))
        print(f"Number of subgraphs: {len(node_counts)}")
        print(f"Subgraph node count: min={min(node_counts)}, max={max(node_counts)}, mean={np.mean(node_counts):.1f}")
        if edge_counts:
            print(f"Subgraph edge count: min={min(edge_counts)}, max={max(edge_counts)}, mean={np.mean(edge_counts):.1f}")

# Example:
analyze_lmdb('./data/subgraph_db/lmdb_train')
