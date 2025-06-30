import os
import pdb
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import gzip
import csv
from scipy.sparse import csr_matrix
import pickle

def plot_rel_dist(adj_list, filename):
    rel_count = []
    for adj in adj_list:
        rel_count.append(adj.count_nonzero())

    fig = plt.figure(figsize=(12, 8))
    plt.plot(rel_count)
    fig.savefig(filename, dpi=fig.dpi)


def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')


def build_adj_mtx(triples_path, entity2id, relation2id):
    """
    Xây dựng adj_list (list csr_matrix, mỗi quan hệ một matrix) từ triples đã mapping.
    - triples_path: file npy chứa các triple đã mapping (head, rel, tail)
    - n_entities: tổng số entity (len(entity2id))
    - n_relations: tổng số relation (len(relation2id))
    """
    n_entities = len(entity2id)
    n_relations = len(relation2id)
    triples = np.load(triples_path)  # shape [num_triple, 3] (h, r, t)
    adj_list = []
    for rel in range(n_relations):
        mask = (triples[:, 1] == rel)
        heads, tails = triples[mask, 0], triples[mask, 2]
        data = np.ones(len(heads), dtype=np.int8)
        adj = csr_matrix((data, (heads, tails)), shape=(n_entities, n_entities))
        adj_list.append(adj)

    return adj_list

def debug_tensor(tensor, name=None, num_values=10):
    """
    In ra thông tin chi tiết về một tensor để debug.
    Args:
        tensor: torch.Tensor
        name: Tên tensor (string) in ra cho dễ nhận diện
        num_values: Số giá trị đầu tiên sẽ print ra (mặc định 10)
    """
    if name:
        print(f"[DEBUG] Tensor '{name}':")
    else:
        print("[DEBUG] Tensor:")

    # Thông tin chung
    print(f"  Shape: {tuple(tensor.shape)}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")

    # Nếu tensor ở trên GPU, chuyển về CPU để print
    if tensor.device.type == 'cuda':
        tensor = tensor.cpu()

    # Flatten và lấy vài giá trị đầu tiên (có thể dùng .numpy() luôn)
    arr = tensor.detach().cpu().flatten()
    n = min(num_values, arr.shape[0])
    print(f"  Values: {arr[:n].numpy()}" + (f" ... ({arr.shape[0]} values)" if arr.shape[0] > n else ""))
    print("")