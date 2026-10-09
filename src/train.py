import math
import random
import warnings
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, update_bn, get_ema_multi_avg_fn
from ogb.graphproppred import Evaluator

from model import GCN
from dataset import build_loaders  # 방금 만든 함수 임포트
from utils import PPANodeEncoder

# torch.load(weights_only=False) 가 구버전 체크포인트를 풀 때 나오는 내부 경고.
# 동작에는 영향 없어 메시지만 무음 처리.
warnings.filterwarnings('ignore', message='TypedStorage is deprecated')

# 안전한 모델 로드를 위한 패치
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# Ampere+ 에서 공짜 가속
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


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


def make_scheduler(optimizer, warmup_epochs, total_epochs):
    """warmup_epochs 선형 warmup 후 cosine 으로 0 까지 감소 (epoch 단위)."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def to_device(data, device):
    """배치를 GPU 로 비동기 전송. 모델이 필요로 하는 VN 총 개수는 CPU 텐서일 때
    미리 파이썬 int 로 뽑아 둬서 forward 안의 GPU->CPU 동기화(.max()/int())를 제거."""
    if getattr(data, 'num_vn', None) is not None:
        data.num_vn_total = int(data.num_vn.sum())
    return data.to(device, non_blocking=True)


def train_one_epoch(model, loader, optimizer, criterion, device,
                    amp_dtype=None, grad_clip=None, ema=None):
    model.train()

    # 배치마다 loss.item() 을 부르면 매번 CUDA 동기화 -> GPU 텐서로 누적, 에폭 끝에 1회만 .item()
    total_loss = torch.zeros((), device=device)
    total_graphs = 0

    for data in loader:
        data = to_device(data, device)

        optimizer.zero_grad(set_to_none=True)

        if amp_dtype is not None:
            with torch.autocast('cuda', dtype=amp_dtype):
                logits = model(data)                 # [B, 37]
                labels = data.y.view(-1).long()      # [B]
                loss = criterion(logits, labels)
        else:
            logits = model(data)
            labels = data.y.view(-1).long()
            loss = criterion(logits, labels)

        loss.backward()                              # autocast 밖; bf16 은 GradScaler 불필요
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update_parameters(model)

        batch_size = labels.size(0)                  # 정적 shape 이라 동기화 없음
        total_loss += loss.detach().float() * batch_size
        total_graphs += batch_size

    return total_loss.item() / total_graphs


@torch.no_grad()
def evaluate(model, loader, evaluator, device, amp_dtype=None):
    model.eval()

    y_true = []
    y_pred = []

    for data in loader:
        data = to_device(data, device)

        if amp_dtype is not None:
            with torch.autocast('cuda', dtype=amp_dtype):
                logits = model(data)  # [B, 37]
        else:
            logits = model(data)
        pred = logits.argmax(dim=1, keepdim=True)  # [B, 1]

        # GPU 에 모아뒀다가 루프 밖에서 한 번에 CPU 로 (배치별 .cpu() 동기화 제거)
        y_true.append(data.y.view(-1, 1))
        y_pred.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu()
    y_pred = torch.cat(y_pred, dim=0).cpu()

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


def build_model_and_optimizer(cfg, num_classes, device):
    tcfg = cfg['train']
    model = GCN(
        cfg=cfg,
        node_encoder=PPANodeEncoder(tcfg['emb_dim']),
        num_classes=num_classes,
    ).to(device)
    # adamw: GCN+ 와 동일한 decoupled weight decay (Adam 의 L2 와 다름). 기본값은 기존 동작(adam).
    opt_cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[str(tcfg.get('optimizer', 'adam')).lower()]
    optimizer = opt_cls(model.parameters(), lr=tcfg['lr'], weight_decay=tcfg['weight_decay'],
                        fused=(device.type == 'cuda'))
    return model, optimizer, nn.CrossEntropyLoss()


def run_training(cfg, model, optimizer, criterion, train_loader, val_loader,
                 evaluator, device, epochs=None, model_name='best_model.pt',
                 on_epoch_end=None, ema=None):
    """학습 루프. val acc 최고 체크포인트를 model_name 에 저장하고 best val acc 반환.
    on_epoch_end(epoch, val_acc) 는 Optuna pruning 등에 사용."""
    tcfg = cfg['train']
    epochs        = int(epochs if epochs is not None else tcfg.get('epochs', 120))
    warmup_epochs = min(int(tcfg.get('warmup_epochs', 5)), epochs)
    grad_clip     = tcfg.get('grad_clip', 1.0)
    use_amp       = bool(tcfg.get('amp', True)) and device.type == 'cuda'
    amp_dtype     = torch.bfloat16 if use_amp else None
    scheduler = make_scheduler(optimizer, warmup_epochs, epochs)

    best_acc = float('-inf')
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            amp_dtype=amp_dtype, grad_clip=grad_clip,
            ema=(ema if ema is not None and epoch > warmup_epochs else None),
        )
        val_acc = evaluate(model, val_loader, evaluator, device, amp_dtype=amp_dtype)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_name)

        lr_now = optimizer.param_groups[0]['lr']
        print(
            f'Epoch {epoch:03d} | Loss: {loss:.4f} | '
            f'Val Acc: {val_acc:.4f} | Best: {best_acc:.4f} | lr: {lr_now:.2e}'
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch, val_acc)
    return best_acc


def main():
    cfg = load_config()
    seed = cfg['train'].get('seed', 0)
    set_seed(seed)
    device = select_device(cfg)
    print('Using device:', device, '| seed:', seed)

    tcfg = cfg['train']
    epochs        = int(tcfg.get('epochs', 120))
    warmup_epochs = int(tcfg.get('warmup_epochs', 5))
    grad_clip     = tcfg.get('grad_clip', 1.0)
    ema_decay     = tcfg.get('ema_decay', 0.999)
    ema_decay     = None if ema_decay is None else float(ema_decay)
    use_amp       = bool(tcfg.get('amp', True)) and device.type == 'cuda'
    amp_dtype     = torch.bfloat16 if use_amp else None

    train_loader, val_loader, test_loader, num_tasks, num_classes = build_loaders(cfg)
    print(f'num_tasks: {num_tasks}, num_classes: {num_classes}')

    model, optimizer, criterion = build_model_and_optimizer(cfg, num_classes, device)
    evaluator = Evaluator(cfg['data']['dataset_name'])

    # 가중치 EMA: 파라미터만 평균, BN 통계는 학습 후 update_bn 으로 재계산.
    # ema_decay: null 이면 EMA 전체를 건너뜀 (raw 체크포인트만 사용).
    ema = None
    if ema_decay is not None:
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))

    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'epochs={epochs} warmup={warmup_epochs} '
          f'amp={"bf16" if use_amp else "off"} grad_clip={grad_clip} ema_decay={ema_decay}')

    run_training(cfg, model, optimizer, criterion, train_loader, val_loader,
                 evaluator, device, epochs=epochs, model_name='best_model.pt', ema=ema)

    # --- raw best (val 로 고른 체크포인트) ---
    model.load_state_dict(torch.load(
        'best_model.pt', map_location=device, weights_only=True,
    ))
    val_raw  = evaluate(model, val_loader, evaluator, device, amp_dtype=amp_dtype)
    test_raw = evaluate(model, test_loader, evaluator, device, amp_dtype=amp_dtype)

    if ema is None:
        print(f'raw : val {val_raw:.4f} | test {test_raw:.4f}')
        print(f'==> Final Test Accuracy: {test_raw:.4f} (raw)')
        return

    # --- EMA (BN running stats 를 train 데이터로 재계산한 뒤 평가) ---
    update_bn(train_loader, ema, device=device)
    val_ema  = evaluate(ema, val_loader, evaluator, device, amp_dtype=amp_dtype)
    test_ema = evaluate(ema, test_loader, evaluator, device, amp_dtype=amp_dtype)
    torch.save(ema.module.state_dict(), 'best_ema.pt')

    use_ema = val_ema >= val_raw
    print(f'raw : val {val_raw:.4f} | test {test_raw:.4f}')
    print(f'ema : val {val_ema:.4f} | test {test_ema:.4f}')
    print(f'==> Final Test Accuracy: {(test_ema if use_ema else test_raw):.4f} '
          f'({"ema" if use_ema else "raw"}, selected by val)')


if __name__ == "__main__":
    main()