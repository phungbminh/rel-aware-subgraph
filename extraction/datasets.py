import os
import lmdb
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from collections import OrderedDict
from utils import debug_tensor

#####################
# 1. COLLATE FN
#####################

def collate_pyg(batch):
    """
    Chuẩn hóa batch cho PyG:
    batch: list of dict {'graph', 'relation', 'neg_graphs'}
    """
    pos_graphs = []
    neg_graphs_list = []
    relations = []
    max_num_negs = max(len(item['neg_graphs']) for item in batch)

    for item in batch:
        pos_graphs.append(item['graph'])
        relations.append(item['relation'])
        negs = item['neg_graphs']
        # Pad cho đều
        if len(negs) < max_num_negs:
            negs = negs + [Data()] * (max_num_negs - len(negs))
        neg_graphs_list.append(negs)

    return pos_graphs, relations, neg_graphs_list, max_num_negs

#####################
# 2. DATASET
#####################

class SubGraphDataset(Dataset):
    def __init__(self, db_path, mapping_dir, global_graph=None, num_negatives=0, split='train', cache_size=4096, is_debug=False):
        # Load entity/relation mapping
        self.entity2id = pickle.load(open(os.path.join(mapping_dir, 'entity2id.pkl'), 'rb'))
        self.relation2id = pickle.load(open(os.path.join(mapping_dir, 'relation2id.pkl'), 'rb'))
        self.id2entity = pickle.load(open(os.path.join(mapping_dir, 'id2entity.pkl'), 'rb'))
        self.id2relation = pickle.load(open(os.path.join(mapping_dir, 'id2relation.pkl'), 'rb'))

        # LMDB keys - CHỈ lấy key hợp lệ (value là pickle)
        self.env = lmdb.open(db_path, readonly=True, lock=False, readahead=False)
        with self.env.begin() as txn:
            self.keys = []
            n_bad = 0
            for k, v in txn.cursor():
                if v and v[:1] in [b'\x80', b'\x81']:
                    self.keys.append(k)
                else:
                    n_bad += 1
            print(f"[INFO] Loaded {len(self.keys)} valid subgraphs from {db_path}, {n_bad} key lỗi đã bị loại.")

        self.global_graph = global_graph
        self.num_negatives = num_negatives
        self.split = split
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.valid_entity_ids = set(self.entity2id.keys())
        self.is_debug = is_debug
        if self.is_debug:
            print(f"[DEBUG] Dataset initialized with {len(self)} samples")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        print(f"[DEBUG][Dataset] __getitem__ idx={idx}")
        # Check cache trước
        if idx in self.cache:
            item = self.cache.pop(idx)
            self.cache[idx] = item
            if self.is_debug and idx < 5:
                print(f"[DEBUG][CACHE] idx={idx}, keys: {list(item.keys())}")
            return item

        key = self.keys[idx]
        with self.env.begin() as txn:
            raw = txn.get(key)
        if raw is None:
            raise ValueError(f"Key {key} not found in LMDB {self.env}")

        data = pickle.loads(raw)

        # Debug dữ liệu raw
        if self.is_debug and idx < 5:
            print(f"[DEBUG][LMDB] Sample {idx}: key={key}, data keys: {list(data.keys())}")
            if 'edge_index' in data:
                debug_tensor(data['edge_index'], f"Sample{idx}/edge_index", 5)
            if 'node_label' in data:
                debug_tensor(data['node_label'], f"Sample{idx}/node_label", 5)
            if 'h_idx' in data:
                print(f"Sample{idx}: h_idx={data['h_idx']}, t_idx={data['t_idx']}")
            print("-----------------")

        # Tạo PyG Data object
        graph = self.create_graph(data['triple'], data['nodes'], data['s_dist'], data['t_dist'])

        # Debug graph structure
        if self.is_debug and idx < 5:
            print(f"[DEBUG][Graph] idx={idx} | x shape: {graph.x.shape} | edge_index: {graph.edge_index.shape}")
            debug_tensor(graph.x, f"Sample{idx}/x", 5)
            debug_tensor(graph.edge_index, f"Sample{idx}/edge_index", 10)

        # Sinh negative sample nếu cần
        neg_graphs = []
        if self.num_negatives > 0 and self.split == "train":
            neg_graphs = self.sample_negatives(data['triple'], data['nodes'], data['s_dist'], data['t_dist'])
            if self.is_debug and idx < 5:
                print(f"[DEBUG][Negatives] idx={idx}, num_negatives={len(neg_graphs)}")
                for i, g in enumerate(neg_graphs[:2]):  # chỉ debug 2 negative đầu
                    print(f"  [DEBUG] Negative {i}: x shape {g.x.shape if hasattr(g, 'x') else 'N/A'}")

        # Đóng gói lại
        item = {
            'graph': graph,
            'relation': graph.relation_label,
            'neg_graphs': neg_graphs
        }
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        self.cache[idx] = item
        return item

    def create_graph(self, triple, nodes, s_dist, t_dist):
        """
        Tạo PyG Data object cho subgraph
        """
        h_raw, r_raw, t_raw = triple
        h = self.entity2id.get(h_raw, -1)
        r = self.relation2id.get(r_raw, -1)
        t = self.entity2id.get(t_raw, -1)
        nodes_id = [self.entity2id.get(n, -1) for n in nodes]

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
            if n_id < 0 or n_id + 1 >= len(self.global_graph.indptr):
                continue  # out of bounds
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
        """
        Sinh negative triples, tái sử dụng structure như positive.
        """
        h_raw, r_raw, t_raw = triple
        all_entities = list(self.entity2id.keys())
        negatives = []
        for _ in range(self.num_negatives):
            if np.random.rand() < 0.5:
                while True:
                    h_neg = np.random.choice(all_entities)
                    if h_neg != h_raw: break
                neg_triple = (h_neg, r_raw, t_raw)
            else:
                while True:
                    t_neg = np.random.choice(all_entities)
                    if t_neg != t_raw: break
                neg_triple = (h_raw, r_raw, t_neg)
            negatives.append(self.create_graph(neg_triple, nodes, s_dist, t_dist))
        return negatives

#####################
# 3. DEMO TEST LOADER
#####################

if __name__ == "__main__":
    # Dummy global graph structure cho test nhanh
    class DummyCSR:
        def __init__(self, num_nodes=200):
            self.indptr = np.arange(0, num_nodes+1)
            self.indices = np.arange(num_nodes)

    dataset = SubGraphDataset(
        db_path="/Users/minhbui/Personal/Project/Master/rel-aware-subgraph/debug_subgraph_db/train.lmdb",
        mapping_dir="/Users/minhbui/Personal/Project/Master/rel-aware-subgraph/debug_subgraph_db/mappings/",
        global_graph=DummyCSR(),
        num_negatives=2,
        split='train',
        is_debug = True
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_pyg
    )

    for pos_graphs, relations, negatives_list, num_negs in loader:
        if getattr(dataset, 'is_debug', False):
            print(f"[DEBUG][Loader] Got batch. Batch size: {len(pos_graphs)}")
            print(f"[DEBUG][Loader] Relations: {relations}")
            print(f"[DEBUG][Loader] Num negatives per pos: {num_negs}")
            debug_tensor(pos_graphs[0].x, "First pos_graph/x")

        break

    count = 0
    for pos_graphs, relations, negatives_list, num_negs in loader:
        print(f"[BATCH {count}] Batch size: {len(pos_graphs)}")
        count += 1
