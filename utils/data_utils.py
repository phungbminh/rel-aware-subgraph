import os
import pdb
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import gzip
import csv
from scipy.sparse import csr_matrix

def plot_rel_dist(adj_list, filename):
    rel_count = []
    for adj in adj_list:
        rel_count.append(adj.count_nonzero())

    fig = plt.figure(figsize=(12, 8))
    plt.plot(rel_count)
    fig.savefig(filename, dpi=fig.dpi)


# def process_files(files, saved_relation2id=None):
#     '''
#     files: Dictionary map of file paths to read the triplets from.
#     saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
#     '''
#     entity2id = {}
#     relation2id = {} if saved_relation2id is None else saved_relation2id
#
#     triplets = {}
#
#     ent = 0
#     rel = 0
#
#     for file_type, file_path in files.items():
#
#         data = []
#         with open(file_path) as f:
#             file_data = [line.split() for line in f.read().split('\n')[:-1]]
#
#         for triplet in file_data:
#             if triplet[0] not in entity2id:
#                 entity2id[triplet[0]] = ent
#                 ent += 1
#             if triplet[2] not in entity2id:
#                 entity2id[triplet[2]] = ent
#                 ent += 1
#             if not saved_relation2id and triplet[1] not in relation2id:
#                 relation2id[triplet[1]] = rel
#                 rel += 1
#
#             # Save the triplets corresponding to only the known relations
#             if triplet[1] in relation2id:
#                 data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])
#
#         triplets[file_type] = np.array(data)
#
#     id2entity = {v: k for k, v in entity2id.items()}
#     id2relation = {v: k for k, v in relation2id.items()}
#
#     # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
#     adj_list = []
#     for i in range(len(relation2id)):
#         idx = np.argwhere(triplets['train'][:, 2] == i)
#         adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))
#
#     return adj_list, triplets, entity2id, relation2id, id2entity, id2relation
#
#
def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')


def process_files(files, saved_relation2id=None):
    """
    files: dict với key 'relations' trỏ tới thư mục raw/relations
           (mỗi folder con là một quan hệ, chứa edge.csv.gz, edge_reltype.csv.gz, num-edge-list.csv.gz)
    saved_relation2id: (tuỳ chọn) dict name->id nếu bạn muốn dùng lại mapping cố định

    Trả về:
      - adj_list       : list of csr_matrix, mỗi phần tử là adjacency matrix của một relation
      - triplets       : {}  (unused downstream)
      - entity2id      : dict raw_id -> contig_id
      - relation2id    : dict rel_name -> rel_idx
      - id2entity      : dict contig_id -> raw_id
      - id2relation    : dict rel_idx -> rel_name
    """
    relations_dir = files['relations']
    # 1) liệt kê tên các folder con (mỗi folder là 1 quan hệ)
    rel_folders = sorted([
        d for d in os.listdir(relations_dir)
        if os.path.isdir(os.path.join(relations_dir, d))
    ])

    # 2) build relation2id / id2relation
    if saved_relation2id:
        relation2id = saved_relation2id
        # ensure id2relation in sync
        id2relation = {v: k for k, v in relation2id.items()}
    else:
        relation2id = {name: idx for idx, name in enumerate(rel_folders)}
        id2relation = {idx: name for name, idx in relation2id.items()}

    # 3) đọc tất cả các edges để tìm max entity ID
    max_entity = 0
    edges_per_rel = {}
    for rel_name in rel_folders:
        path = os.path.join(relations_dir, rel_name, 'edge.csv.gz')
        with gzip.open(path, 'rt') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header nếu có
            edges = []
            for row in reader:
                h, t = int(row[0]), int(row[1])
                edges.append((h, t))
                max_entity = max(max_entity, h, t)
        edges_per_rel[rel_name] = edges

    # 4) build entity2id / id2entity (chúng ta map raw_id -> itself, cho contiguous)
    n_entities = max_entity + 1
    entity2id = {i: i for i in range(n_entities)}
    id2entity = {i: i for i in range(n_entities)}

    # 5) build adj_list
    adj_list = []
    for rel_name in rel_folders:
        edges = edges_per_rel[rel_name]
        heads = [entity2id[h] for h, _ in edges]
        tails = [entity2id[t] for _, t in edges]
        data = np.ones(len(heads), dtype=np.uint8)
        adj = csr_matrix((data, (heads, tails)),
                         shape=(n_entities, n_entities))
        adj_list.append(adj)

    # 6) trả về
    triplets = {}  # không dùng downstream
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation