import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset


def _add_dummy_node_feature(data):
    # ogbg-ppa는 노드 입력 피처가 없다(x=None). 모든 노드를 동일한 더미 인덱스(0)로
    # 채워서 모델의 dummy_node_emb(nn.Embedding(1, emb_dim))가 처리하게 한다.
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def build_loaders(cfg):
    """VN/centroid 전처리를 전혀 거치지 않고 원본 edge_index/edge_attr을 그대로 사용한다."""
    dataset = PygGraphPropPredDataset(
        name=cfg['data']['dataset_name'],
        transform=_add_dummy_node_feature,
    )
    split_idx = dataset.get_idx_split()

    train_loader = DataLoader(
        Subset(dataset, split_idx['train']),
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
    )
    val_loader = DataLoader(
        Subset(dataset, split_idx['valid']),
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['train']['num_workers'],
    )
    test_loader = DataLoader(
        Subset(dataset, split_idx['test']),
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['train']['num_workers'],
    )

    num_tasks = int(dataset.num_classes)
    edge_dim = dataset[0].edge_attr.size(-1)

    return train_loader, val_loader, test_loader, num_tasks, edge_dim
