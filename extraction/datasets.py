from torch.utils.data import Dataset
import torch
import lmdb
import struct
import json
import numpy as np
from utils.graph_utils import deserialize, ssp_multigraph_to_pyg
from utils.pyg_utils import collate_pyg, move_batch_to_device_pyg
from .graph_sampler import extract_relation_aware_subgraph, sample_neg
from ogb.linkproppred import LinkPropPredDataset
from tqdm import tqdm
import os
from utils import process_files


class SubgraphDataset(Dataset):
    """
    PyG Dataset for RASG: Each item is a tuple (pos_data, g_label, r_label, neg_data_list, neg_g_labels, neg_r_labels)
    """
    def __init__(
        self, db_path, db_name_pos, db_name_neg, raw_data_paths, included_relations=None,
        add_transpose_rels=False, num_neg_samples_per_link=1, use_kge_embeddings=False,
        dataset='', kge_model='', file_name='',
        relation_emb_dim=200, node_label_emb_dim=16, max_dist=10
    ):
        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.file_name = file_name

        # Xử lý features (nếu dùng embedding từ knowledge graph)
        self.node_features = None
        self.kge_entity2id = None
        if use_kge_embeddings:
            self.node_features, self.kge_entity2id = get_kge_embeddings(dataset, kge_model)
        # Xây dựng graph từ file raw (adjacency list)
        ssp_graph, __, __, __, id2entity, id2relation = process_files(raw_data_paths, included_relations)
        if add_transpose_rels:
            ssp_graph += [adj.T for adj in ssp_graph]
        self.num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_pyg(ssp_graph, self.node_features)
        self.ssp_graph = ssp_graph
        self.id2entity = id2entity
        self.id2relation = id2relation

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')
            # Đọc các thông số mô tả subgraph nếu cần

    def __len__(self):
        with self.main_env.begin() as txn:
            num_pos = txn.stat(db=self.db_pos)["entries"]
        return num_pos

    def __getitem__(self, idx):
        """
        Trả về:
            pos_data: Data (subgraph positive)
            g_label: tensor
            r_label: tensor
            neg_data_list: List[Data]
            neg_g_labels: List[tensor]
            neg_r_labels: List[tensor]
        """
        # Đọc subgraph dương
        with self.main_env.begin() as txn:
            key = str(idx).encode()
            data_pos = txn.get(key, db=self.db_pos)
            pos_dict = deserialize(data_pos)
        # Đọc subgraph âm
        with self.main_env.begin() as txn:
            neg_key = str(idx).encode()
            data_neg = txn.get(neg_key, db=self.db_neg)
            neg_list = json.loads(data_neg)
        # Tạo subgraph positive bằng relation-aware extraction
        h, t, r = pos_dict['h'], pos_dict['t'], pos_dict['r_label']
        filtered_nodes, sub_edge_index, sub_edge_type, node_label = extract_relation_aware_subgraph(
            self.graph.edge_index, self.graph.edge_type, h, t, r,
            num_nodes=self.graph.num_nodes, k=2, tau=2
        )
        from torch_geometric.data import Data
        # Node features (label + relation embedding) sẽ thêm ở bước DataLoader/Model
        pos_data = Data(
            edge_index=sub_edge_index,
            edge_type=sub_edge_type,
            num_nodes=filtered_nodes.size(0),
            node_label=node_label,  # (N, 2)
            h_idx=(filtered_nodes == h).nonzero(as_tuple=True)[0][0],   # chỉ số node h trong subgraph
            t_idx=(filtered_nodes == t).nonzero(as_tuple=True)[0][0],   # chỉ số node t trong subgraph
        )

        # Negative samples
        neg_data_list, neg_g_labels, neg_r_labels = [], [], []
        for neg_dict in neg_list:
            h_neg, t_neg, r_neg = neg_dict['h'], neg_dict['t'], neg_dict['r_label']
            filtered_nodes_neg, sub_edge_index_neg, sub_edge_type_neg, node_label_neg = extract_relation_aware_subgraph(
                self.graph.edge_index, self.graph.edge_type, h_neg, t_neg, r_neg,
                num_nodes=self.graph.num_nodes, k=2, tau=2
            )
            neg_data = Data(
                edge_index=sub_edge_index_neg,
                edge_type=sub_edge_type_neg,
                num_nodes=filtered_nodes_neg.size(0),
                node_label=node_label_neg,
                h_idx=(filtered_nodes_neg == h_neg).nonzero(as_tuple=True)[0][0],
                t_idx=(filtered_nodes_neg == t_neg).nonzero(as_tuple=True)[0][0],
            )
            neg_data_list.append(neg_data)
            neg_g_labels.append(torch.tensor(neg_dict.get('g_label', -1)))
            neg_r_labels.append(torch.tensor(neg_dict.get('r_label', -1)))

        g_label = torch.tensor(pos_dict['g_label'])
        r_label = torch.tensor(pos_dict['r_label'])

        return pos_data, g_label, r_label, neg_data_list, neg_g_labels, neg_r_labels


def get_kge_embeddings(dataset, kge_model):
    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}
    return node_features, kge_entity2id
