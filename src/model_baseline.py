import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import degree


class GCNConv(MessagePassing):
    """VN/centroid 트리 엣지 없이 원본 데이터셋의 edge_index만 사용하는 순수 GCN 레이어.
    edge_dim을 인자로 받아 ogbg-code2(augment_edge의 2차원)와 ogbg-ppa(STRING 채널 7차원)
    양쪽에 대응한다."""

    def __init__(self, emb_dim, edge_dim):
        super().__init__(aggr='add')
        self.linear       = nn.Linear(emb_dim, emb_dim)
        self.root_emb     = nn.Embedding(1, emb_dim)
        self.edge_encoder = nn.Linear(edge_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x        = self.linear(x)
        edge_emb = self.edge_encoder(edge_attr.float())
        row, _   = edge_index
        deg          = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5).clamp(max=1e9)
        norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
        agg  = self.propagate(edge_index, x=x, edge_attr=edge_emb, norm=norm)
        return agg + F.relu(x + self.root_emb.weight) / deg.view(-1, 1).clamp(min=1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)


class GCNGraphClassifier(nn.Module):
    """VN/centroid 기작이 전혀 없는 baseline GCN. 그래프 하나당 단일 라벨(num_tasks 클래스)을
    예측한다. ogbg-ppa처럼 노드 입력 피처가 없는 데이터셋은 node_encoder=None으로 두면
    모든 노드에 동일한 학습 가능한 더미 임베딩을 부여해 처리한다."""

    def __init__(self, cfg, num_tasks, edge_dim, node_encoder=None):
        super().__init__()

        train_cfg = cfg.get('train', {})
        model_cfg = cfg.get('model', {})

        emb_dim    = train_cfg.get('emb_dim', 300)
        num_layers = model_cfg.get('num_layers', 5)
        drop_ratio = train_cfg.get('drop_ratio', 0.5)

        self.node_encoder = node_encoder
        if self.node_encoder is None:
            self.dummy_node_emb = nn.Embedding(1, emb_dim)

        self.convs = nn.ModuleList([GCNConv(emb_dim, edge_dim) for _ in range(num_layers)])
        self.bns   = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layers)])
        self.drop_ratio = drop_ratio
        self.num_layer  = num_layers

        self.pred_head = nn.Linear(emb_dim, num_tasks)

    def forward(self, data):
        if self.node_encoder is not None:
            h = self.node_encoder(data.x, data.node_depth.view(-1))
        else:
            h = self.dummy_node_emb(data.x)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = bn(conv(h, data.edge_index, data.edge_attr))
            h = F.dropout(F.relu(h) if i < self.num_layer - 1 else h,
                          p=self.drop_ratio, training=self.training)

        h_graph = global_mean_pool(h, data.batch)
        return self.pred_head(h_graph)
