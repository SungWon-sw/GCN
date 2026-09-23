import builtins
import contextlib

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset


def _add_dummy_node_feature(data):
    # ogbg-ppa는 노드 입력 피처가 없다(x=None). 모든 노드를 동일한 더미 인덱스(0)로
    # 채워서 모델의 dummy_node_emb(nn.Embedding(1, emb_dim))가 처리하게 한다.
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


@contextlib.contextmanager
def _auto_decline_dataset_update_prompt():
    """ogb가 '데이터셋이 갱신됐다'며 input()으로 재다운로드 여부를 물어보는데,
    nohup 등 stdin이 없는 환경에서는 그 input() 자체가 OSError(Bad file
    descriptor)로 죽는다. 여기선 무조건 'N'(기존 캐시 유지)으로 자동 응답한다."""
    original_input = builtins.input

    def _auto_no(prompt=''):
        print(prompt, end='')
        print('N  (자동 응답 — 비대화형 환경이라 기존 캐시 데이터셋을 그대로 사용)')
        return 'N'

    builtins.input = _auto_no
    try:
        yield
    finally:
        builtins.input = original_input


def build_loaders(cfg):
    """VN/centroid 전처리를 전혀 거치지 않고 원본 edge_index/edge_attr을 그대로 사용한다."""
    with _auto_decline_dataset_update_prompt():
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
