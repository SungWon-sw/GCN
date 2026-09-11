import yaml
import torch
import torch.nn as nn
from ogb.graphproppred import Evaluator

from model import GCN
from dataset import build_loaders  # 방금 만든 함수 임포트
from utils import decode_arr_to_seq, ASTNodeEncoder  # 평가 및 엔코더에 필요한 것만 유지
import random
import numpy as np

def set_seed(seed):
    """RNG 시드 고정. multi-seed 평균±std 용도라 bitwise 결정성까지는 안 감
    (cudnn.benchmark 유지). None 이면 시드 고정을 건너뜀."""
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


def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = torch.zeros((), device=device)
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        label = data.y_arr.to(torch.long)
        loss = sum(criterion(pred, label[:, i]) for i, pred in enumerate(model(data)))

        loss.backward()
        optimizer.step()
        total_loss += loss.detach()  # 배치마다 .item()으로 동기화하지 않고 GPU에 누적
    return (total_loss / len(loader)).item()  # 동기화는 에폭 끝에 한 번만


@torch.no_grad()
def evaluate(model, loader, idx2vocab, evaluator, device):
    model.eval()
    refs = []
    pred_batches = []
    for data in loader:
        data = data.to(device)
        pred_batches.append(torch.stack([p.argmax(-1) for p in model(data)], dim=1))  # GPU에 둔 채로 모음
        refs.extend(data.y)

    pred_arr = torch.cat(pred_batches, dim=0).cpu()  # 동기화(.cpu())는 전체 배치 끝난 뒤 한 번만
    pred_lists = pred_arr.tolist()  # 텐서 인덱싱 대신 한 번에 파이썬 리스트로 변환
    preds = [decode_arr_to_seq(row, idx2vocab) for row in pred_lists]

    return evaluator.eval({'seq_ref': refs, 'seq_pred': preds})['F1']


def main():
    cfg = load_config()
    seed = cfg['train'].get('seed', 0)
    set_seed(seed)
    cuda_num = cfg['cuda']['cuda_number']  # config에서 번호 직접 추출 (예: 7)
    device = torch.device(f'cuda:{cuda_num}')
    print('Using device:', device)

    # 1. 데이터 파트: 복잡한 로직은 src/dataset.py가 처리하고 로더와 메타데이터만 받음
    train_loader, val_loader, test_loader, idx2vocab, num_nodetypes, num_nodeattributes = build_loaders(cfg)
    print(f'num_nodetypes: {num_nodetypes}, num_nodeattributes: {num_nodeattributes}')

    # 2. 모델 파트
    node_encoder = ASTNodeEncoder(
        cfg['train']['emb_dim'], 
        num_nodetypes, 
        num_nodeattributes, 
        max_depth=cfg['train']['max_depth']
    )

    model = GCN(
        cfg=cfg, 
        node_encoder=node_encoder, 
        num_tasks=len(idx2vocab)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    evaluator = Evaluator(cfg['data']['dataset_name'])
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # 3. 학습 루프 파트
    best_f1 = 0.0
    model_name=cfg['model']['model_name']
    for epoch in range(1, cfg['train']['epochs'] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_f1 = evaluate(model, val_loader, idx2vocab, evaluator, device)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_name)
            
        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Val F1: {val_f1:.4f} | Best: {best_f1:.4f}')

    # 4. 최종 테스트 평가
    model.load_state_dict(torch.load(model_name))
    test_f1 = evaluate(model, test_loader, idx2vocab, evaluator, device)
    print(f'==> Final Test F1: {test_f1:.4f}')


if __name__ == "__main__":
    main()