import numpy as np
import torch
import scipy.sparse as ssp
from torch_geometric.data import Data

def ssp_multigraph_to_pyg(adj_list, n_feats=None):
    """
    Chuyển list scipy sparse adjacency matrix (mỗi matrix là một relation) sang PyG Data object,
    với các cạnh chứa trường edge_type để đánh dấu loại quan hệ.
    """
    edge_indices = []
    edge_types = []

    for rel, adj in enumerate(adj_list):
        coo = adj.tocoo()
        if coo.nnz == 0:
            continue
        edge_indices.append(np.stack([coo.row, coo.col], axis=0))  # [2, num_edges_rel]
        edge_types.append(np.full(coo.row.shape, rel, dtype=np.int64))  # [num_edges_rel]

    if len(edge_indices) == 0:
        raise ValueError("No edges found in any relation!")

    edge_index = np.concatenate(edge_indices, axis=1)  # [2, num_edges_all]
    edge_type = np.concatenate(edge_types, axis=0)     # [num_edges_all]

    pyg_data = Data(
        edge_index=torch.from_numpy(edge_index).long(),
        edge_type=torch.from_numpy(edge_type).long(),
        num_nodes=adj_list[0].shape[0]
    )
    if n_feats is not None:
        pyg_data.x = torch.from_numpy(n_feats).float()
    return pyg_data

def serialize(data):
    # Tùy bạn đang dùng pickle/json gì, giữ nguyên hoặc điều chỉnh cho PyG
    import pickle
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)

def deserialize(data):
    import pickle
    keys = ('nodes', 'r_label', 'g_label', 'n_label')
    data_tuple = pickle.loads(data)
    return dict(zip(keys, data_tuple))

# Utility cho node labeling
def node_label_embedding(node_labels, num_classes, emb_dim):
    """
    node_labels: (N, 2) tensor, (d(h,v), d(t,v))
    Trả về: Tensor (N, emb_dim)
    """
    # Mỗi distance được embed riêng, sau đó concat
    emb = torch.nn.Embedding(num_classes, emb_dim)
    out = torch.cat([emb(node_labels[:, 0]), emb(node_labels[:, 1])], dim=1)
    return out
