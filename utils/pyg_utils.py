import torch
from torch_geometric.data import Batch
from torch_geometric.data import Batch, Data

def collate_pyg(batch):
    """
    Chuẩn hóa batch:
        - batch: List các dict với keys: 'graph', 'relation', 'neg_graphs'
        - Ghép positive và negative thành danh sách, pad negative nếu thiếu
    Trả về:
        - batch_all: Batch PyG
        - relations: Tensor relation ids
        - batch_size: Số positive
        - num_negs: Số negative mỗi positive
        - negatives_list: List negative của từng positive (giữ cho create_mega_batch)
    """
    pos_graphs = []
    neg_graphs_list = []
    relations = []

    # Tìm số lượng negative tối đa
    max_num_negs = max(len(item['neg_graphs']) for item in batch)

    for item in batch:
        pos_graphs.append(item['graph'])
        relations.append(item['relation'])

        negs = item['neg_graphs']
        # Pad bằng Data() nếu thiếu
        if len(negs) < max_num_negs:
            negs = negs + [Data()] * (max_num_negs - len(negs))
        neg_graphs_list.append(negs)

    # Dạng [batch_size, num_negs] => [batch_size * num_negs]
    # (sẽ flatten ở create_mega_batch)
    return pos_graphs, relations, neg_graphs_list, max_num_negs


def move_batch_to_device_pyg(batch, device):
    """
    Đưa toàn bộ batch data lên device (CPU/GPU).
    """
    batch_pos, g_label, r_label, batch_negs, batch_neg_g_labels, batch_neg_r_labels = batch

    batch_pos = batch_pos.to(device)
    batch_negs = [b.to(device) if b is not None else None for b in batch_negs]

    g_label = g_label.to(device)
    r_label = r_label.to(device)
    batch_neg_g_labels = [g.to(device) if g is not None else None for g in batch_neg_g_labels]
    batch_neg_r_labels = [r.to(device) if r is not None else None for r in batch_neg_r_labels]

    return batch_pos, g_label, r_label, batch_negs, batch_neg_g_labels, batch_neg_r_labels
