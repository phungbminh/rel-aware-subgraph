from torch.utils.data import Dataset
import json
import torch
import os, sys, types

os.environ["DGL_DISABLE_GRAPHBOLT"] = "1"
stub = types.ModuleType("dgl.graphbolt")
stub.load_graphbolt = lambda *args, **kwargs: None
sys.modules["dgl.graphbolt"] = stub
import dgl
from ogb.linkproppred import LinkPropPredDataset
from utils import ssp_multigraph_to_dgl, deserialize
from utils import process_files
from scipy.sparse import csr_matrix
from .graph_sampler import *
from tqdm import tqdm


def generate_subgraph_datasets(params, splits=['train', 'valid', 'test'], max_label_value=None):
    """
    Dedicated pipeline for OGB-BioKG (ogbl-biokg) subgraph extraction.

    Steps:
      1. Load ogbl-biokg from params.main_dir
      2. Subsample positives up to params.max_links
      3. Build per-relation adjacency list from train positives
      4. Sample negatives with sample_neg
      5. Extract enclosing subgraphs around pos/neg links via links2subgraphs
      6. Store everything in LMDB at params.db_path

    Required params attributes:
      - main_dir: str, root dir for OGB data
      - db_path: str, path to write LMDB (params.db_path)
      - max_links: int, max positives per split
      - num_neg_samples_per_link: int
      - constrained_neg_prob: float
    """
    # 1. Load dataset and splits
    dataset = LinkPropPredDataset(name='ogbl-biokg', root=params.main_dir)
    split_edge = dataset.get_edge_split()
    # n_entities = sum(dataset[0]['num_nodes_dict'].values())
    # print(f"n_entities = {n_entities}")
    # # 2. Tính số entities và relations từ split data
    all_nodes = np.concatenate([
        split_edge['train']['head'], split_edge['train']['tail'],
        split_edge['valid']['head'], split_edge['valid']['tail'],
        split_edge['test']['head'], split_edge['test']['tail'],
    ], axis=0)
    n_entities = int(all_nodes.max()) + 1
    print(f"n_entities = {n_entities}")

    all_rels = np.concatenate([
        split_edge['train']['relation'],
        split_edge['valid']['relation'],
        split_edge['test']['relation'],
    ], axis=0)
    # train_triples, valid_triples, test_triples = split_edge["train"], split_edge["valid"], split_edge["test"]
    # n_relations = int(max(train_triples['relation']))+1
    # print(f"n_relations = {n_relations}")
    n_relations = int(all_rels.max().item()) + 1
    print(f"n_relations = {n_relations}")
    # 2. Collect positive triplets per split
    graphs = {}
    #for split in splits:
    for split in tqdm(splits, desc="Subsampling positives"):
        edges = split_edge[split]
        triplets = np.stack([edges['head'], edges['tail'], edges['relation']], axis=1)
        if params.max_links is not None and triplets.shape[0] > params.max_links:
            idx = np.random.choice(triplets.shape[0], params.max_links, replace=False)
            triplets = triplets[idx]
        graphs[split] = {'pos': triplets, 'max_size': params.max_links}

    # 3. Build adjacency list from training positives
    raw_train = split_edge['train']
    adj_list = []
    #for rel in range(n_relations):
    for rel in tqdm(range(n_relations), desc="Building adj list"):
        mask = raw_train['relation'] == rel
        heads = raw_train['head'][mask]
        tails = raw_train['tail'][mask]
        data = np.ones(heads.shape[0], dtype=np.int8)
        adj = csr_matrix((data, (heads, tails)), shape=(n_entities, n_entities))
        adj_list.append(adj)

    # 4. Negative sampling
    for split_name, info in graphs.items():
    #for split_name, info in tqdm(graphs.items(), desc="Sampling negatives"):
    #for split_name, info in tqdm(graphs.items(), desc="Sampling negatives", total=len(graphs)):
        #print(f"Sampling negative links for OGB-BioKG split '{split_name}'")
        pos_edges = info['pos']
        _, neg_edges = sample_neg(
            adj_list=adj_list,
            edges=pos_edges,
            num_neg_samples_per_link=params.num_neg_samples_per_link,
            max_size=info['max_size'],
            constrained_neg_prob=params.constrained_neg_prob
        )
        info['neg'] = neg_edges

    # 5. Subgraph extraction and storage
    links2subgraphs(adj_list, graphs, params, max_label_value)


def get_kge_embeddings(dataset, kge_model):
    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, db_path, db_name_pos, db_name_neg, raw_data_paths, included_relations=None,
                 add_traspose_rels=False, num_neg_samples_per_link=1, use_kge_embeddings=False, dataset='',
                 kge_model='', file_name=''):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.node_features, self.kge_entity2id = get_kge_embeddings(dataset, kge_model) if use_kge_embeddings else (
            None, None)
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.file_name = file_name

        ssp_graph, __, __, __, id2entity, id2relation = process_files(raw_data_paths, included_relations)
        self.num_rels = len(ssp_graph)

        # Add transpose matrices to handle both directions of relations.
        if add_traspose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.ssp_graph = ssp_graph
        self.id2entity = id2entity
        self.id2relation = id2relation

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
            self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
            self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
            self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))

            self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
            self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
            self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
            self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))

            self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
            self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
            self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
            self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

        print(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")

        print('=====================')
        print(
            f"Subgraph size stats: \n Avg size {self.avg_subgraph_size}, \n Min size {self.min_subgraph_size}, \n Max size {self.max_subgraph_size}, \n Std {self.std_subgraph_size}")

        print('=====================')
        print(
            f"Enclosed nodes ratio stats: \n Avg size {self.avg_enc_ratio}, \n Min size {self.min_enc_ratio}, \n Max size {self.max_enc_ratio}, \n Std {self.std_enc_ratio}")

        print('=====================')
        print(
            f"# of pruned nodes stats: \n Avg size {self.avg_num_pruned_nodes}, \n Min size {self.min_num_pruned_nodes}, \n Max size {self.max_num_pruned_nodes}, \n Std {self.std_num_pruned_nodes}")

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        self.__getitem__(0)

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
            # lookup the real key (1-based) from pos_keys
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)
        subgraphs_neg = []
        r_labels_neg = []
        g_labels_neg = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                subgraphs_neg.append(self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg))
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)

        return subgraph_pos, g_label_pos, r_label_pos, subgraphs_neg, g_labels_neg, r_labels_neg

    def __len__(self):
        return self.num_graphs_pos

    def _prepare_subgraphs(self, nodes, r_label, n_labels):
        # subgraph = dgl.DGLGraph(self.graph.subgraph(nodes))
        # subgraph.edata['type'] = self.graph.edata['type'][self.graph.subgraph(nodes).parent_eid]
        # subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)
        #
        # edges_btw_roots = subgraph.edge_id(0, 1)
        # rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        # if rel_link.squeeze().nelement() == 0:
        #     subgraph.add_edge(0, 1)
        #     subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
        #     subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)

        # 1) Tạo subgraph và lưu lại ID gốc của edges
        #    relabel_nodes=True để nodes đánh lại từ 0, store_ids=True để lưu dgl.EID
        subgraph = dgl.node_subgraph(self.graph, nodes,
                               relabel_nodes=True,
                               store_ids=True)

        # 2) Lấy bản đồ từ edge trong subgraph -> edge gốc
        orig_eids = subgraph.edata[dgl.EID]  # tensor dài = số edge trong sg

        # 3) Gán kiểu relation và label
        subgraph.edata['type'] = self.graph.edata['type'][orig_eids]
        subgraph.edata['label'] = subgraph.edata['type']  # hoặc tạo tensor r_label tương ứng

        # 4) Đảm bảo có đường nối giữa 2 root nodes (0 và 1)
        if not subgraph.has_edges_between(0, 1):
            # 2) Thêm cạnh mới
            subgraph.add_edges([0], [1])
            # 3) Lấy EID của cạnh vừa thêm
            new_eid = subgraph.num_edges() - 1
            # 4) Gán type & label
            subgraph.edata['type'][new_eid] = r_label
            subgraph.edata['label'][new_eid] = r_label

        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features_new(subgraph, n_labels, n_feats)
        # === GraIL++: đính kèm query-relation ID lên mỗi node ===
        n_nodes = subgraph.number_of_nodes()
        subgraph.ndata['query_rel'] = torch.full(
            (n_nodes,),  # 1 giá trị r_label cho mỗi node
            r_label,  # chính là relation của triple (h,r,t)
            dtype=torch.long
        )
        return subgraph

    def _prepare_features(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1))
        label_feats[np.arange(n_nodes), n_labels] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), 0] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph
