from ogb.linkproppred import LinkPropPredDataset
from utils.graph_utils import ssp_multigraph_to_pyg
from extraction.graph_sampler import extract_relation_aware_subgraph
import numpy as np

dataset = LinkPropPredDataset(name='ogbl-biokg', root="./data/ogb/")
split_edge = dataset.get_edge_split()

# Build edge_index, edge_type toàn đồ thị
triples_all = np.concatenate([
    np.stack([split_edge['train']['head'], split_edge['train']['relation'], split_edge['train']['tail']], axis=1),
    np.stack([split_edge['valid']['head'], split_edge['valid']['relation'], split_edge['valid']['tail']], axis=1),
    np.stack([split_edge['test']['head'], split_edge['test']['relation'], split_edge['test']['tail']], axis=1),
], axis=0)
n_entities = int(triples_all[:, [0,2]].max()) + 1
n_relations = int(triples_all[:,1].max()) + 1

from scipy.sparse import csr_matrix
adj_list = []
for rel in range(n_relations):
    mask = (triples_all[:,1] == rel)
    heads, tails = triples_all[mask, 0], triples_all[mask, 2]
    data = np.ones(len(heads), dtype=np.int8)
    adj = csr_matrix((data, (heads, tails)), shape=(n_entities, n_entities))
    adj_list.append(adj)
pyg_graph = ssp_multigraph_to_pyg(adj_list)
edge_index, edge_type = pyg_graph.edge_index, pyg_graph.edge_type
num_nodes = pyg_graph.num_nodes

# Lấy 1 triplet đầu tiên
edges = split_edge['train']
triples = np.stack([edges['head'], edges['tail'], edges['relation']], axis=1)
h, t, r = triples[0]
print(f"Try extract with h={h}, t={t}, r={r}")

import time
start = time.time()
filtered_nodes, sub_edge_index, sub_edge_type, node_label = extract_relation_aware_subgraph(
    edge_index, edge_type, int(h), int(t), int(r), num_nodes, k=1, tau=1
)
print("Done, time elapsed:", time.time() - start)
print(f"Subgraph nodes: {len(filtered_nodes)}")
