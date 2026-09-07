import yaml
import torch
import torch.nn as nn
from ogb.graphproppred import Evaluator

from model import GCN
from dataset import build_loaders  # 방금 만든 함수 임포트
from utils import PPANodeEncoder

# 안전한 모델 로드를 위한 패치
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load


def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def print_vn_stats(data, graph_index):
    """Report direct real-node assignments for a single, unbatched graph."""
    num_vn = data.vn_batch.numel()
    counts = torch.bincount(data.node2vn, minlength=num_vn)
    max_count, vn_index = counts.max(dim=0)
    print(
        f'VN stats (first train graph, dataset index={graph_index}) | '
        f'Nodes: {data.num_nodes} | VNs: {num_vn} | '
        f'Max nodes per VN: {max_count.item()} (VN index={vn_index.item()})',
        flush=True,
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_graphs = 0

    for data in loader:
        data = data.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(data)              # [B, 37]
        labels = data.y.view(-1).long()   # [B]

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_graphs += batch_size

    return total_loss / total_graphs


@torch.no_grad()
def evaluate(model, loader, evaluator, device):
    model.eval()

    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)

        logits = model(data)  # [B, 37]
        pred = logits.argmax(dim=1, keepdim=True)  # [B, 1]

        y_true.append(data.y.view(-1, 1).cpu())
        y_pred.append(pred.cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    result = evaluator.eval({
        'y_true': y_true,
        'y_pred': y_pred,
    })

    return result['acc']

def select_device(cfg):
    cuda_num = int(cfg['cuda']['cuda_number'])
    if cuda_num == -1:
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError(
            f'CUDA를 사용할 수 없습니다 (torch={torch.__version__}, '
            f'CUDA build={torch.version.cuda}). '
            '가상환경에서 uv pip install -r requirements.txt를 실행하고 '
            'nvidia-smi로 GPU 접근을 확인하세요. '
            'CPU 실행은 cuda_number: -1로 명시하세요.'
        )
    if not 0 <= cuda_num < torch.cuda.device_count():
        raise ValueError(
            f'cuda_number={cuda_num}: 사용 가능한 GPU 번호는 '
            f'0~{torch.cuda.device_count() - 1}입니다.'
        )
    device = torch.device(f'cuda:{cuda_num}')
    # 데이터 로드 전에 실제 CUDA 초기화와 메모리 할당 확인
    torch.empty(1, device=device)
    return device


def main():
    cfg = load_config()
    device = select_device(cfg)
    print('Using device:', device)

    # 1. 데이터 파트: 복잡한 로직은 src/dataset.py가 처리하고 로더와 메타데이터만 받음
    train_loader, val_loader, test_loader, num_tasks, num_classes = build_loaders(cfg)
    print(f'num_tasks: {num_tasks}, num_classes: {num_classes}')
    sample_graph = train_loader.dataset[0]
    print_vn_stats(sample_graph, int(train_loader.dataset.indices[0]))

    node_encoder = PPANodeEncoder(cfg['train']['emb_dim'])

    model = GCN(
        cfg=cfg,
        node_encoder=node_encoder,
        num_classes=num_classes,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    evaluator = Evaluator(cfg['data']['dataset_name'])
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    best_acc = float('-inf')

    for epoch in range(1, cfg['train']['epochs'] + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_acc = evaluate(model, val_loader, evaluator, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')

        print(
            f'Epoch {epoch:03d} | Loss: {loss:.4f} | '
            f'Val Acc: {val_acc:.4f} | Best: {best_acc:.4f}'
        )

    state_dict = torch.load(
        'best_model.pt',
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    test_acc = evaluate(model, test_loader, evaluator, device)
    print(f'==> Final Test Accuracy: {test_acc:.4f}')


if __name__ == "__main__":
    main()
