import random

import numpy as np
import yaml
import torch
import torch.nn as nn
from ogb.graphproppred import Evaluator

from model_baseline import GCNGraphClassifier
from dataset_ppa import build_loaders


def set_seed(seed):
    """RNG 시드 고정. original 브랜치(935a5eb)와 동일한 정책: None이면 건너뜀."""
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 안전한 모델 로드를 위한 패치
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load


def load_config(config_path="configs/config_ppa.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = torch.zeros((), device=device)
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        label = data.y.view(-1).to(torch.long)
        loss = criterion(model(data), label)

        loss.backward()
        optimizer.step()
        total_loss += loss.detach()  # 배치마다 .item()으로 동기화하지 않고 GPU에 누적
    return (total_loss / len(loader)).item()  # 동기화는 epoch 끝에 한 번만


@torch.no_grad()
def evaluate(model, loader, evaluator, device):
    model.eval()
    y_true_batches, y_pred_batches = [], []
    for data in loader:
        data = data.to(device)
        pred = model(data).argmax(dim=-1, keepdim=True)  # GPU에 둔 채로 모음
        y_pred_batches.append(pred)
        y_true_batches.append(data.y.view(-1, 1))

    y_true = torch.cat(y_true_batches, dim=0).cpu()  # 동기화(.cpu())는 전체 배치 끝난 뒤 한 번만
    y_pred = torch.cat(y_pred_batches, dim=0).cpu()
    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})['acc']


def main():
    cfg = load_config()
    seed = cfg['train'].get('seed', 0)
    set_seed(seed)
    cuda_num = cfg['cuda']['cuda_number']
    device = torch.device(f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # VN/centroid 전처리 없이 원본 ogbg-ppa 그래프 그대로 로드
    train_loader, val_loader, test_loader, num_tasks, edge_dim = build_loaders(cfg)
    print(f'num_tasks(classes): {num_tasks}, edge_dim: {edge_dim}')

    model = GCNGraphClassifier(cfg=cfg, num_tasks=num_tasks, edge_dim=edge_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    evaluator = Evaluator(cfg['data']['dataset_name'])
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    best_acc = 0.0
    model_name = cfg['model']['model_name']
    for epoch in range(1, cfg['train']['epochs'] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, evaluator, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_name)

        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Best: {best_acc:.4f}')

    model.load_state_dict(torch.load(model_name))
    test_acc = evaluate(model, test_loader, evaluator, device)
    print(f'==> Final Test Acc: {test_acc:.4f}')


if __name__ == "__main__":
    main()
