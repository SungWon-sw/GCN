import torch

_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

import torch.nn as nn
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from model import GCN
from utils import get_vocab_mapping, encode_y_to_arr, ASTNodeEncoder, augment_edge, decode_arr_to_seq, convert_into_easy;

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)

NUM_VOCAB, MAX_SEQ_LEN = 5000, 5
BATCH_SIZE, LR, EPOCHS = 32, 1e-3, 30
from torch.utils.data import Subset
dataset   = PygGraphPropPredDataset(name='ogbg-code2')
split_idx = dataset.get_idx_split()

vocab2idx, idx2vocab = get_vocab_mapping([dataset[i].y for i in split_idx['train']], NUM_VOCAB)


def transform_fn(data):
    data2 = encode_y_to_arr(augment_edge(data), vocab2idx, MAX_SEQ_LEN)
    return data2

dataset.transform = transform_fn

dataset = convert_into_easy(dataset)

train_loader = DataLoader(Subset(dataset, split_idx['train']), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(Subset(dataset, split_idx['valid']), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(Subset(dataset, split_idx['test']),  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

num_nodetypes      = int(dataset.data.x[:, 0].max().item()) + 1
num_nodeattributes = int(dataset.data.x[:, 1].max().item()) + 1
print(f'num_nodetypes: {num_nodetypes}, num_nodeattributes: {num_nodeattributes}')

node_encoder = ASTNodeEncoder(300, num_nodetypes, num_nodeattributes, max_depth=20)

model = GCN(len(idx2vocab), MAX_SEQ_LEN, node_encoder, drop_ratio=0.6).to(DEVICE)
model.load_state_dict(torch.load('best_model.pt'))  # 이 줄 추가
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
evaluator = Evaluator('ogbg-code2')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

def train(loader):
    model.train()
    total_loss = 0.0
    for i, data in enumerate(loader):
        data = data.to(DEVICE)
        optimizer.zero_grad()
        label = data.y_arr.to(torch.long)
        loss  = sum(criterion(pred, label[:,i]) for i, pred in enumerate(model(data)))
        loss.backward(); optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    refs, preds = [], []
    for data in loader:
        data     = data.to(DEVICE)
        pred_arr = torch.stack([p.argmax(-1) for p in model(data)], dim=1).cpu()
        for i in range(pred_arr.size(0)):
            preds.append(decode_arr_to_seq(pred_arr[i],        idx2vocab))
            refs.append( decode_arr_to_seq(data.y_arr[i].cpu(), idx2vocab))
    return evaluator.eval({'seq_ref': refs, 'seq_pred': preds})['F1']
 
best = 0.0
for epoch in range(31, 61):
    loss = train(train_loader)
    f1   = evaluate(val_loader)
    if f1 > best:
        best = f1; torch.save(model.state_dict(), 'best_model.pt')
    print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Val F1: {f1:.4f} | Best: {best:.4f}')

model.load_state_dict(torch.load('best_model.pt'))
print(f'Test F1: {evaluate(test_loader):.4f}')