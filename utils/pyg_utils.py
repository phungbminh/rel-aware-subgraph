import torch
from torch_geometric.data import Batch

def collate_pyg(samples):
    pos_data, g_label, r_label, neg_data_list, neg_g_labels, neg_r_labels = zip(*samples)
    batch_pos = Batch.from_data_list(pos_data)
    batch_negs = [Batch.from_data_list(neg_list) for neg_list in neg_data_list]
    g_label = torch.stack(g_label)
    r_label = torch.stack(r_label)
    batch_neg_g_labels = [torch.stack(g) for g in neg_g_labels]
    batch_neg_r_labels = [torch.stack(r) for r in neg_r_labels]
    return batch_pos, g_label, r_label, batch_negs, batch_neg_g_labels, batch_neg_r_labels

def move_batch_to_device_pyg(batch, device):
    batch_pos, g_label, r_label, batch_negs, batch_neg_g_labels, batch_neg_r_labels = batch
    batch_pos = batch_pos.to(device)
    batch_negs = [b.to(device) for b in batch_negs]
    g_label = g_label.to(device)
    r_label = r_label.to(device)
    batch_neg_g_labels = [x.to(device) for x in batch_neg_g_labels]
    batch_neg_r_labels = [x.to(device) for x in batch_neg_r_labels]
    return batch_pos, g_label, r_label, batch_negs, batch_neg_g_labels, batch_neg_r_labels
