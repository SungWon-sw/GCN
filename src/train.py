import yaml
import torch
import torch.nn as nn
from ogb.graphproppred import Evaluator

from model import GCN
from dataset import build_loaders  # 방금 만든 함수 임포트
from utils import decode_arr_to_seq, ASTNodeEncoder  # 평가 및 엔코더에 필요한 것만 유지

# 안전한 모델 로드를 위한 패치
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load


def load_config(config_path="../configs/config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        label = data.y_arr.to(torch.long)
        loss = sum(criterion(pred, label[:, i]) for i, pred in enumerate(model(data)))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, idx2vocab, evaluator, device):
    model.eval()
    refs, preds = [], []
    for data in loader:
        data = data.to(device)
        pred_arr = torch.stack([p.argmax(-1) for p in model(data)], dim=1).cpu()
        
        for i in range(pred_arr.size(0)):
            preds.append(decode_arr_to_seq(pred_arr[i], idx2vocab))
            refs.append(decode_arr_to_seq(data.y_arr[i].cpu(), idx2vocab))
    return evaluator.eval({'seq_ref': refs, 'seq_pred': preds})['F1']


def main():
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # 1. 데이터 파트: 복잡한 로직은 src/dataset.py가 처리하고 로더와 메타데이터만 받음
    train_loader, val_loader, test_loader, idx2vocab, num_nodetypes, num_nodeattributes = build_loaders(cfg)
    print(f'num_nodetypes: {num_nodetypes}, num_nodeattributes: {num_nodeattributes}')

    # 2. 모델 파트
    node_encoder = ASTNodeEncoder(
        cfg['model']['emb_dim'], 
        num_nodetypes, 
        num_nodeattributes, 
        max_depth=cfg['model']['max_depth']
    )

    model = GCN(
        num_tasks=len(idx2vocab), 
        max_seq_len=cfg['train']['max_seq_len'], 
        node_encoder=node_encoder, 
        drop_ratio=cfg['model']['drop_ratio']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    evaluator = Evaluator(cfg['data']['dataset_name'])
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # 3. 학습 루프 파트
    best_f1 = 0.0
    for epoch in range(1, cfg['train']['epochs'] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_f1 = evaluate(model, val_loader, idx2vocab, evaluator, device)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), cfg['train']['model_save_path'])
            
        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Val F1: {val_f1:.4f} | Best: {best_f1:.4f}')

    # 4. 최종 테스트 평가
    model.load_state_dict(torch.load(cfg['train']['model_save_path']))
    test_f1 = evaluate(model, test_loader, idx2vocab, evaluator, device)
    print(f'==> Final Test F1: {test_f1:.4f}')


if __name__ == "__main__":
    main()