import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GlobalAttention
from torch_geometric.utils import softmax

class NodeLabelEmbedding(nn.Module):
    def __init__(self, max_dist, emb_dim):
        super().__init__()
        self.label_emb1 = nn.Embedding(max_dist+1, emb_dim)
        self.label_emb2 = nn.Embedding(max_dist+1, emb_dim)
    def forward(self, node_label):
        # node_label: [N, 2]  (d(h,v), d(t,v))
        return torch.cat([
            self.label_emb1(node_label[:,0].clamp(max=self.label_emb1.num_embeddings-1)),
            self.label_emb2(node_label[:,1].clamp(max=self.label_emb2.num_embeddings-1))
        ], dim=-1)   # [N, emb_dim*2]

class RelationEmbedding(nn.Module):
    def __init__(self, num_rels, emb_dim):
        super().__init__()
        self.rel_emb = nn.Embedding(num_rels, emb_dim)
    def forward(self, rel_id, size):
        # rel_id: int (target relation), size: num_nodes (for broadcast)
        e = self.rel_emb(torch.tensor(rel_id, device=self.rel_emb.weight.device))
        return e.unsqueeze(0).repeat(size, 1)  # [num_nodes, emb_dim]

# ----------- CompGCN layer (1 block) -------------
class CompGCNConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_rels, act=F.relu):
        super().__init__(aggr='add')
        self.lin_node = nn.Linear(in_dim, out_dim)
        self.lin_rel = nn.Embedding(num_rels, out_dim)
        self.act = act
    def forward(self, x, edge_index, edge_type):
        # x: [N, d], edge_index: [2, E], edge_type: [E]
        rel_emb = self.lin_rel(edge_type)  # [E, d]
        # GNN: truyền thông điệp (neighbor + rel)
        return self.propagate(edge_index, x=x, rel_emb=rel_emb)
    def message(self, x_j, rel_emb):
        # x_j: neighbor feature [E, d], rel_emb [E, d]
        return x_j + rel_emb
    def update(self, aggr_out, x):
        return self.act(self.lin_node(aggr_out) + x)

# ----------- Attention Pooling -----------
class AttPool(nn.Module):
    def __init__(self, in_dim, att_dim=64):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(in_dim, att_dim), nn.LeakyReLU(0.2), nn.Linear(att_dim, 1)
        )
    def forward(self, x):
        att_score = self.gate_nn(x)   # [N, 1]
        att_weight = torch.softmax(att_score, dim=0)  # [N, 1]
        return (att_weight * x).sum(dim=0)            # [in_dim]

# -------------- Full RASG Model -------------------
class RASGModel(nn.Module):
    def __init__(self, num_rels, node_label_max_dist, node_label_emb_dim=16,
                 rel_emb_dim=32, gnn_hidden_dim=128, num_layers=3, att_dim=64, out_dim=1):
        super().__init__()
        self.node_label_emb = NodeLabelEmbedding(node_label_max_dist, node_label_emb_dim)
        self.rel_emb = RelationEmbedding(num_rels, rel_emb_dim)
        self.input_dim = node_label_emb_dim*2 + rel_emb_dim
        self.comp_layers = nn.ModuleList([
            CompGCNConv(self.input_dim if i==0 else gnn_hidden_dim, gnn_hidden_dim, num_rels)
            for i in range(num_layers)
        ])
        self.att_pool = AttPool(gnn_hidden_dim, att_dim)
        self.final_linear = nn.Linear(gnn_hidden_dim*3, out_dim)

    def forward(self, data, rel_id):
        # data: PyG Data,  1 subgraph
        # rel_id: int, id relation của triple này (lấy từ data.r_label)
        # Node label embedding
        node_label_emb = self.node_label_emb(data.node_label)    # [N, label_emb*2]
        rel_emb = self.rel_emb(rel_id, data.num_nodes)           # [N, rel_emb_dim]
        x = torch.cat([node_label_emb, rel_emb], dim=1)          # [N, d]
        # CompGCN encode
        for layer in self.comp_layers:
            x = layer(x, data.edge_index, data.edge_type)
        # Attention pooling toàn subgraph
        z_subgraph = self.att_pool(x)    # [hidden_dim]
        # Lấy embedding của node h và t (trong subgraph, idx relabel sau extract)
        h_vec = x[data.h_idx]
        t_vec = x[data.t_idx]
        final_vec = torch.cat([z_subgraph, h_vec, t_vec], dim=-1)   # [hidden_dim*3]
        score = self.final_linear(final_vec).squeeze(-1)            # [1] (nếu out_dim=1)
        return score
