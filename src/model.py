import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import degree

class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super().__init__(aggr='add')
        self.linear       = nn.Linear(emb_dim, emb_dim)
        self.root_emb     = nn.Embedding(1, emb_dim)
        self.edge_encoder = nn.Linear(2, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x        = self.linear(x)
        edge_emb = self.edge_encoder(edge_attr.float())
        row, _   = edge_index
        deg          = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5).clamp(max=1e9)
        norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
        agg  = self.propagate(edge_index, x=x, edge_attr=edge_emb, norm=norm)
        return agg + F.relu(x + self.root_emb.weight) / deg.view(-1,1).clamp(min=1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1,1) * F.relu(x_j + edge_attr)
    
class GCN(nn.Module):
    def __init__(self, num_vocab, max_seq_len, node_encoder, num_layer=5, emb_dim=300, drop_ratio=0.5):
        super().__init__()
        self.convs      = nn.ModuleList([GCNConv(emb_dim) for _ in range(num_layer)])
        self.bns        = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])
        self.node_encoder = node_encoder
        self.drop_ratio = drop_ratio
        self.num_layer  = num_layer

        self.pred_heads = nn.ModuleList([
            nn.Linear(emb_dim, num_vocab) for _ in range(max_seq_len)
        ])
        self.max_seq_len = max_seq_len

    def forward(self, data):
        h = self.node_encoder(data.x, data.node_depth.view(-1))
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = bn(conv(h, data.edge_index, data.edge_attr))
            h = F.dropout(F.relu(h) if i < self.num_layer-1 else h,
                          p=self.drop_ratio, training=self.training)
        h_graph = global_mean_pool(h, data.batch)
        return [head(h_graph) for head in self.pred_heads]