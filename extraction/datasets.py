import os
import lmdb
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from collections import OrderedDict
from utils import collate_pyg

class SubGraphDataset(Dataset):
    """
    Dataset cho LMDB subgraph.
    - Tự động cache sample để tăng tốc truy cập.
    - Hỗ trợ negative sampling động nếu cần.
    - Tái sử dụng global graph để truy xuất cạnh subgraph.
    """
    def __init__(self, db_path, mapping_dir, global_graph=None, num_negatives=0, split='train', cache_size=4096):
        # Nạp mapping entity/relation
        self.entity2id = pickle.load(open(os.path.join(mapping_dir, 'entity2id.pkl'), 'rb'))
        self.relation2id = pickle.load(open(os.path.join(mapping_dir, 'relation2id.pkl'), 'rb'))
        self.id2entity = pickle.load(open(os.path.join(mapping_dir, 'id2entity.pkl'), 'rb'))
        self.id2relation = pickle.load(open(os.path.join(mapping_dir, 'id2relation.pkl'), 'rb'))

        # LMDB: luôn lưu keys dạng bytes!
        self.env = lmdb.open(db_path, readonly=True, lock=False, readahead=False)
        with self.env.begin() as txn:
            self.keys = [k for k, _ in txn.cursor() if k != b'_progress']

        self.global_graph = global_graph
        self.num_negatives = num_negatives
        self.split = split
        self.cache = OrderedDict()
        self.cache_size = cache_size

        self.valid_entity_ids = set(self.entity2id.keys())
        print(f"[INFO] Loaded {len(self)} subgraphs from {db_path}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if idx in self.cache:
            # LRU cache: đưa lên cuối
            item = self.cache.pop(idx)
            self.cache[idx] = item
            return item

        key = self.keys[idx]
        with self.env.begin() as txn:
            raw = txn.get(key)
        if raw is None:
            raise ValueError(f"Key {key} not found in LMDB {self.env}")

        data = pickle.loads(raw)
        graph = self.create_graph(data['triple'], data['nodes'], data['s_dist'], data['t_dist'])

        # Sinh negative sample (nếu training và num_negatives > 0)
        neg_graphs = []
        if self.num_negatives > 0 and self.split == "train":
            neg_graphs = self.sample_negatives(data['triple'], data['nodes'], data['s_dist'], data['t_dist'])

        item = {'graph': graph, 'relation': graph.relation_label, 'neg_graphs': neg_graphs}
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        self.cache[idx] = item
        return item

    def create_graph(self, triple, nodes, s_dist, t_dist):
        """Tạo PyG Data object cho subgraph"""
        h_raw, r_raw, t_raw = triple
        h = self.entity2id.get(h_raw, -1)
        r = self.relation2id.get(r_raw, -1)
        t = self.entity2id.get(t_raw, -1)

        # Map node ids, nếu không có thì -1
        nodes_id = [self.entity2id.get(n, -1) for n in nodes]

        # Tạo mask cho head/tail
        # head_mask = torch.tensor([n == h_raw for n in nodes], dtype=torch.bool)
        # tail_mask = torch.tensor([n == t_raw for n in nodes], dtype=torch.bool)
        head_mask = torch.tensor(np.array(nodes) == h_raw, dtype=torch.bool)
        tail_mask = torch.tensor(np.array(nodes) == t_raw, dtype=torch.bool)
        head_idx = head_mask.nonzero(as_tuple=True)[0].item() if head_mask.any() else -1
        tail_idx = tail_mask.nonzero(as_tuple=True)[0].item() if tail_mask.any() else -1

        edge_index = self.get_subgraph_edges(nodes)

        return Data(
            x=torch.stack([torch.tensor(s_dist, dtype=torch.float),
                           torch.tensor(t_dist, dtype=torch.float)], dim=1),
            edge_index=edge_index,
            head_idx=head_idx,
            tail_idx=tail_idx,
            head_mask=head_mask,
            tail_mask=tail_mask,
            original_nodes=torch.tensor(nodes_id, dtype=torch.long),
            relation_label=r,
            num_nodes=len(nodes)
        )

    def get_subgraph_edges(self, nodes):
        if self.global_graph is None:
            return torch.empty(2, 0, dtype=torch.long)
        node_set = set(nodes)
        edges = []
        node2idx = {n: i for i, n in enumerate(nodes)}
        for node in nodes:
            if node not in self.valid_entity_ids:
                continue
            n_id = self.entity2id[node]
            # FIX: kiểm tra chỉ số hợp lệ
            if n_id < 0 or n_id + 1 >= len(self.global_graph.indptr):
                continue  # Bỏ node này, tránh out of bounds
            start = self.global_graph.indptr[n_id]
            end = self.global_graph.indptr[n_id + 1]
            neighbors = self.global_graph.indices[start:end]
            for nb in neighbors:
                nb_raw = self.id2entity[nb]
                if nb_raw in node_set and nb_raw in node2idx:
                    edges.append((node2idx[node], node2idx[nb_raw]))
        if not edges:
            return torch.empty(2, 0, dtype=torch.long)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def sample_negatives(self, triple, nodes, s_dist, t_dist):
        """Negative sampling: sinh negative triples, tái sử dụng structure như positive."""
        h_raw, r_raw, t_raw = triple
        all_entities = list(self.entity2id.keys())
        negatives = []
        for _ in range(self.num_negatives):
            if np.random.rand() < 0.5:
                # Negative head
                while True:
                    h_neg = np.random.choice(all_entities)
                    if h_neg != h_raw: break
                neg_triple = (h_neg, r_raw, t_raw)
            else:
                # Negative tail
                while True:
                    t_neg = np.random.choice(all_entities)
                    if t_neg != t_raw: break
                neg_triple = (h_raw, r_raw, t_neg)
            negatives.append(self.create_graph(neg_triple, nodes, s_dist, t_dist))
        return negatives

# --- Demo (chạy thử nghiệm nhỏ) ---
if __name__ == "__main__":
    class DummyCSR:
        def __init__(self, num_nodes=200):
            self.indptr = np.arange(0, num_nodes+1)
            self.indices = np.arange(num_nodes)


    dataset = SubGraphDataset(
        db_path="/Users/minhbui/Personal/Project/Master/rel-aware-subgraph/debug_subgraph_db/train.lmdb/",
        mapping_dir="/Users/minhbui/Personal/Project/Master/rel-aware-subgraph/debug_subgraph_db/",
        global_graph=DummyCSR(),
        num_negatives=2,
        split='train'
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_pyg)
    for batch_all, relations, batch_size, num_negs in loader:
        print("Batch x:", batch_all.x.shape)
        print("Relations:", relations.shape)
        print("Batch_size:", batch_size, "Num negatives:", num_negs)
        break
