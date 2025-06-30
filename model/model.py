# rasg_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch

class NodeLabelEmbedding(nn.Module):
    """Embeds node topological labels (d(h,v), d(t,v)) bằng continuous encoding."""
    def __init__(self, max_dist=10, emb_dim=16, dropout=0.1):
        super().__init__()
        self.max_dist = max_dist
        self.emb = nn.Sequential(
            nn.Linear(2, emb_dim*2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(emb_dim*2, emb_dim)
        )
        nn.init.xavier_uniform_(self.emb[0].weight)
        nn.init.zeros_(self.emb[0].bias)
        nn.init.xavier_uniform_(self.emb[3].weight)
        nn.init.zeros_(self.emb[3].bias)

    def forward(self, node_labels):
        # Normalize: chia cho max_dist (giúp học tốt hơn)
        norm = node_labels.float() / (self.max_dist + 1e-5)
        return self.emb(norm)

class RelationEmbedding(nn.Module):
    """Embedding cho quan hệ. Hỗ trợ LayerNorm và Dropout."""
    def __init__(self, num_rels, emb_dim=32, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(num_rels + 1, emb_dim)  # +1 cho PAD/UNK
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.LayerNorm(emb_dim)
        )
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, rel_ids):
        return self.proj(self.emb(rel_ids))

class CompGCNConv(MessagePassing):
    """
    CompGCN Layer tối ưu cho multi-relational subgraph.
    Cho phép chọn composition: 'sub', 'mult', 'corr'
    """
    def __init__(self, in_dim, out_dim, composition='sub', dropout=0.1):
        super().__init__(aggr='add')
        self.comp = composition
        self.w_rel = nn.Linear(in_dim, out_dim, bias=False)
        self.w_node = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index, edge_attr):
        # edge_attr: (E,) relation id cho mỗi cạnh (đã mapping)
        if edge_attr is None:
            rel_emb = torch.zeros(edge_index.size(1), x.size(1), device=x.device)
        else:
            rel_emb = edge_attr  # Đã embedding ngoài, truyền vào đây
        return self.propagate(edge_index, x=x, rel_emb=rel_emb)

    def message(self, x_j, rel_emb):
        if self.comp == 'sub':
            out = x_j - rel_emb
        elif self.comp == 'mult':
            out = x_j * rel_emb
        elif self.comp == 'corr':
            out = F.normalize(x_j * rel_emb, p=2, dim=-1)
        else:
            out = x_j + rel_emb
        return self.dropout(self.w_rel(out))

    def update(self, aggr_out, x):
        out = aggr_out + self.w_node(x)
        if out.size(0) > 1:
            out = self.bn(out)
        return F.leaky_relu(out, 0.1)

class AttentionPooling(nn.Module):
    """Multi-head attention pooling cho subgraph-level readout, hỗ trợ multi-graph PyG batch."""
    def __init__(self, in_dim, att_dim=64, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.att_dim = att_dim
        self.proj = nn.Linear(in_dim, heads * att_dim, bias=False)
        self.query = nn.Parameter(torch.randn(1, heads, att_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(heads * att_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.normal_(self.query)

    def forward(self, x, batch):
        """
        x: (N, in_dim)
        batch: (N,) graph index cho mỗi node (chuẩn PyG)
        Return: (B, heads*att_dim)   (B = số graph)
        """
        N = x.size(0)
        keys = self.proj(x).view(N, self.heads, self.att_dim)        # (N, heads, att_dim)
        att = (keys * self.query).sum(-1)                            # (N, heads)
        from torch_scatter import scatter_softmax, scatter_sum

        pooled_outs = []
        for h in range(self.heads):
            att_h = att[:, h]
            attn_h = scatter_softmax(att_h, batch)                   # (N,)
            attn_h = self.dropout(attn_h)
            key_h = keys[:, h, :]                                    # (N, att_dim)
            pooled = scatter_sum(attn_h.unsqueeze(-1) * key_h, batch, dim=0)  # (num_graph, att_dim)
            pooled_outs.append(pooled)
        pooled_cat = torch.cat(pooled_outs, dim=-1)                  # (num_graph, heads*att_dim)
        return self.norm(pooled_cat)

class RASG(nn.Module):
    """
    Mô hình Relation-Aware Subgraph Extraction (RASG)
    - Relation-aware subgraph (được extract ngoài)
    - NodeLabelEmbedding + RelationEmbedding (early fusion)
    - Multi-layer CompGCN (composition)
    - AttentionPooling + head/tail concat để tính score
    """
    def __init__(self, num_rels, max_dist=10, node_emb_dim=32, rel_emb_dim=32,
                 gnn_hidden=128, num_layers=3, att_dim=64, att_heads=4, dropout=0.1, composition='sub'):
        super().__init__()
        self.node_emb = NodeLabelEmbedding(max_dist, node_emb_dim, dropout)
        self.rel_emb = RelationEmbedding(num_rels, rel_emb_dim, dropout)
        self.input_dim = node_emb_dim + rel_emb_dim
        self.gnn_layers = nn.ModuleList([
            CompGCNConv(self.input_dim if i == 0 else gnn_hidden, gnn_hidden, composition, dropout)
            for i in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(gnn_hidden) for _ in range(num_layers)])
        self.att_pool = AttentionPooling(gnn_hidden, att_dim, att_heads, dropout)
        self.scorer = nn.Sequential(
            nn.Linear(att_heads * att_dim + 2 * gnn_hidden, gnn_hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden, gnn_hidden // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(gnn_hidden // 2, 1)
        )

    def forward(self, data: Batch, rel_ids: torch.Tensor):
        """
        data: Batch với các field:
            x: (N, 2), edge_index, edge_attr, batch, head_idx, tail_idx
        rel_ids: (B,), relation id của mỗi graph

        """

        def forward(self, data: Batch, rel_ids: torch.Tensor):
            print("[DEBUG][forward] data.x device:", data.x.device)
            print("[DEBUG][forward] rel_ids device:", rel_ids.device)

        x = self.node_emb(data.x)                    # (N, node_emb_dim)
        # Expand rel_ids thành (N,) để concat
        if rel_ids.dim() == 0:
            rel_ids = rel_ids.expand(data.num_graphs)
        node2graph = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        #rel_feat = self.rel_emb(rel_ids[node2graph]) # (N, rel_emb_dim)
        rel_input = rel_ids[node2graph]
        if torch.any(rel_input < 0) or torch.any(rel_input >= self.rel_emb.emb.num_embeddings):
            print("[ERROR] Invalid rel_ids in RASG.forward")
            print("min =", rel_input.min().item(), "max =", rel_input.max().item())
            print("Embedding size =", self.rel_emb.emb.num_embeddings)
            raise ValueError("rel_ids out of embedding range")
        rel_feat = self.rel_emb(rel_input)

        h = torch.cat([x, rel_feat], dim=-1)         # (N, in_dim)
        edge_emb = self.rel_emb(data.edge_attr) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None

        # Multi-layer GNN
        for i, (layer, norm) in enumerate(zip(self.gnn_layers, self.norms)):
            h = layer(h, data.edge_index, edge_emb)
            h = norm(h)
            if i < len(self.gnn_layers) - 1:
                h = F.leaky_relu(h, 0.1)

        # Attention pooling: mask = all ones (hoặc mask head/tail tuỳ chọn)
        mask = torch.ones(h.size(0), dtype=torch.bool, device=h.device)
        #z_graph = self.att_pool(h, mask=mask)          # (1, pool_dim)
        z_graph = self.att_pool(h, data.batch)

        # Collect head/tail indices cho từng graph (giả sử đã padding -1 nếu không có)
        # Chú ý: Nếu batch, head_idx/tail_idx là (B,), lấy h[head_idx], h[tail_idx]
        B = data.num_graphs
        head_idx = getattr(data, 'head_idx', None)
        tail_idx = getattr(data, 'tail_idx', None)
        if head_idx is None or tail_idx is None:
            raise RuntimeError("Missing head_idx/tail_idx in data")
        # Nếu dùng padding -1, cần mask
        head_repr = torch.stack([h[idx] if idx >= 0 else torch.zeros_like(h[0]) for idx in head_idx], dim=0)
        tail_repr = torch.stack([h[idx] if idx >= 0 else torch.zeros_like(h[0]) for idx in tail_idx], dim=0)
        # z_graph (1, pool_dim) -> (B, pool_dim)
        z_graph = z_graph.expand(B, -1)
        features = torch.cat([z_graph, head_repr, tail_repr], dim=-1)  # (B, pool+2*hidden)
        score = self.scorer(features).squeeze(-1)                      # (B,)
        return score
