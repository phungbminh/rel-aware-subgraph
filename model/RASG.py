import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import RelGraphConv


class RASGModel(nn.Module):
    """
    GraIL++ model for link prediction on enclosing subgraphs.

    - input: a (batched) DGLGraph with
        ndatas:
          'feat'       : Tensor[N_total, in_feat_dim]   (one‐hot distance labels ± optional KGE feats)
          'query_rel'  : LongTensor[N_total]             (relation id of the target link)
        edata:
          'type'       : LongTensor[E_total]             (edge‐type ids)
    - output: scores per graph: Tensor[batch_size]
    """

    def __init__(self,
                 in_feat_dim: int,
                 rel_emb_dim: int,
                 hidden_dim: int,
                 num_rels: int,
                 num_bases: int = None,
                 num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers

        # 1) embedding cho quan hệ truy vấn
        self.rel_emb = nn.Embedding(num_rels, rel_emb_dim)

        # 2) project input (feat ⊕ rel_emb) → hidden_dim
        self.input_proj = nn.Linear(in_feat_dim + rel_emb_dim, hidden_dim)

        # 3) relational GNN layers (CompGCN / R-GCN style)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # basis decomposition R-GCN
            self.convs.append(
                RelGraphConv(in_feat=hidden_dim,
                             out_feat=hidden_dim,
                             num_rels=num_rels,
                             regularizer='basis',
                             num_bases=num_bases or num_rels)
            )


        # 4) attention‐based pooling conditioned on e_r
        #    hidden_dim + rel_emb_dim -> attn score
        self.attn_w = nn.Linear(hidden_dim + rel_emb_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)

        # 5) final scoring head
        self.scoring = nn.Linear(hidden_dim, 1)

    def forward(self, bg: dgl.DGLGraph) -> torch.Tensor:
        # unpack batched graph into a list so we can pool per subgraph
        graphs = dgl.unbatch(bg)
        out_scores = []
        for g in graphs:
            h = g.ndata['feat']  # [N, in_feat_dim]
            qrels = g.ndata['query_rel']  # [N]
            etypes = g.edata['type']  # [E]

            # a) relation embedding per node
            er = self.rel_emb(qrels)  # [N, rel_emb_dim]

            # b) initial projection
            h = torch.cat([h, er], dim=-1)  # [N, in+rel]
            h = F.relu(self.input_proj(h))  # [N, hidden]

            # c) relational message‐passing
            for conv in self.convs:
                h = F.relu(conv(g, h, etypes))

            # d) attention pooling
            cat = torch.cat([h, er], dim=-1)  # [N, hidden+rel]
            a = torch.tanh(self.attn_w(cat))  # [N, hidden]
            a = self.attn_score(a).squeeze(-1)  # [N]
            alpha = F.softmax(a, dim=0)  # [N]
            z = torch.sum(h * alpha.unsqueeze(-1), dim=0)  # [hidden]

            # e) score for this subgraph
            score = self.scoring(z)  # [1]
            out_scores.append(score)

        # stack → [batch_size, 1] → squeeze → [batch_size]
        return torch.stack(out_scores, dim=0).squeeze(-1)
