import multiprocessing as mp

import torch
import numpy as np

from utils_centroid import get_centroid_tree

class ASTNodeEncoder(torch.nn.Module):
    """
    x[:, 0] : node type index
    x[:, 1] : node attribute index
    depth    : depth of the node in AST (별도 텐서)
    """
    def __init__(self, emb_dim, num_nodetypes, num_nodeattributes, max_depth):
        super().__init__()
        self.max_depth = max_depth
        self.type_encoder      = torch.nn.Embedding(num_nodetypes,      emb_dim)
        self.attribute_encoder = torch.nn.Embedding(num_nodeattributes, emb_dim)
        self.depth_encoder     = torch.nn.Embedding(max_depth + 1,      emb_dim)

    def forward(self, x, depth):
        depth = depth.clone()
        depth[depth > self.max_depth] = self.max_depth
        return self.type_encoder(x[:, 0]) + self.attribute_encoder(x[:, 1]) + self.depth_encoder(depth)

def get_vocab_mapping(seq_list, num_vocab):
    vocab_cnt, vocab_list = {}, []
    for seq in seq_list:
        for w in seq:
            if w in vocab_cnt: vocab_cnt[w] += 1
            else: vocab_cnt[w] = 1; vocab_list.append(w)

    cnt_list = np.array([vocab_cnt[w] for w in vocab_list])
    topvocab = np.argsort(-cnt_list, kind='stable')[:num_vocab]
    print(f'Vocab coverage: {float(np.sum(cnt_list[topvocab]))/np.sum(cnt_list):.4f}')

    vocab2idx = {vocab_list[i]: idx for idx, i in enumerate(topvocab)}
    idx2vocab = [vocab_list[i] for i in topvocab]
    vocab2idx['__UNK__'] = num_vocab;   idx2vocab.append('__UNK__')
    vocab2idx['__EOS__'] = num_vocab+1; idx2vocab.append('__EOS__')
    return vocab2idx, idx2vocab

def encode_y_to_arr(data, vocab2idx, max_seq_len):
    seq = data.y
    augmented = seq[:max_seq_len] + ['__EOS__'] * max(0, max_seq_len - len(seq))
    data.y_arr = torch.tensor(
        [[vocab2idx.get(w, vocab2idx['__UNK__']) for w in augmented]], dtype=torch.long
    )
    return data

def decode_arr_to_seq(arr, idx2vocab):
    # arr: 파이썬 리스트(정수). 텐서 연산(torch.nonzero/.item()) 대신 순수 파이썬으로 처리해
    # 샘플당 호출되는 이 함수의 텐서 dispatch 오버헤드를 없앤다.
    eos_idx = len(idx2vocab) - 1
    try:
        arr = arr[:arr.index(eos_idx)]
    except ValueError:
        pass
    return [idx2vocab[i] for i in arr]

def augment_edge(data):
    ei = data.edge_index
    ea = torch.zeros(ei.size(1), 2)
    ei_inv = torch.stack([ei[1], ei[0]], dim=0)
    ea_inv = torch.cat([torch.zeros(ei_inv.size(1),1), torch.ones(ei_inv.size(1),1)], dim=1)

    data.edge_index = torch.cat([ei, ei_inv], dim=1)
    data.edge_attr  = torch.cat([ea, ea_inv], dim=0)
    return data

def augment_edge_with_leaf_edge(data):
    ei = data.edge_index
    ea = torch.zeros(ei.size(1), 2)
    ei_inv = torch.stack([ei[1], ei[0]], dim=0)
    ea_inv = torch.cat([torch.zeros(ei_inv.size(1),1), torch.ones(ei_inv.size(1),1)], dim=1)

    attr_nodes = torch.where(data.node_is_attributed.view(-1) == 1)[0]
    if len(attr_nodes) > 1:
        ei_next = torch.stack([attr_nodes[:-1], attr_nodes[1:]], dim=0)
        ea_next = torch.cat([torch.ones(ei_next.size(1),1), torch.zeros(ei_next.size(1),1)], dim=1)
        ei_next_inv = torch.stack([ei_next[1], ei_next[0]], dim=0)
        ea_next_inv = torch.ones(ei_next.size(1), 2)
        data.edge_index = torch.cat([ei, ei_inv, ei_next, ei_next_inv], dim=1)
        data.edge_attr  = torch.cat([ea, ea_inv, ea_next, ea_next_inv], dim=0)
    else:
        data.edge_index = torch.cat([ei, ei_inv], dim=1)
        data.edge_attr  = torch.cat([ea, ea_inv], dim=0)
    return data
from torch_geometric.data import InMemoryDataset

def centroid(tree_data):
    n = len(tree_data)
    edge_data = []

    for i in range(n):
        edge_data.append([i,tree_data[i]])

    tree_ans = get_centroid_tree(n, edge_data)
    return tree_ans

def data_put_edge(data, tree):
    col, row = [], []

    for i in tree: col.append(i[0]); row.append(i[1])

    col = torch.tensor(col, dtype=torch.long)
    row = torch.tensor(row, dtype=torch.long)

    edge_attr = torch.ones(len(col),2)
    edge_index = torch.stack([col, row], dim=0)
    edge_attr_inv = torch.ones(len(col),2)
    edge_index_inv = torch.stack([row, col], dim=0)
    data.edge_index = torch.cat([data.edge_index,edge_index_inv,edge_index], dim=1)
    data.edge_attr = torch.cat([data.edge_attr,edge_attr_inv,edge_attr], dim=0)

    return data


def _traverse_ast_from_edges(edges, num_nodes):
    ret = list(0 for _ in range(num_nodes))

    visited = [False] * num_nodes
    stack = [(0, 0)]
    adj = [[] for _ in range(num_nodes)]

    for i in range(edges.shape[1]):
        src = edges[0, i]
        dst = edges[1, i]
        adj[src].append(dst)

    while stack:
        curr, prev = stack.pop()
        visited[curr] = True

        ret[curr] = prev

        for neighbor in reversed(adj[curr]):
            if not visited[neighbor]:
                stack.append((neighbor, curr))

    return ret


def traverse_ast(data):
    # [수정 구간] 텐서를 넘파이 배열로 변환 후, C-level 속도로 루프 수행
    edges = data.edge_index.cpu().numpy()
    return _traverse_ast_from_edges(edges, data.num_nodes)


class CustomEasyDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__(None)
        # collate 함수를 통해 리스트를 PyG 데이터셋 포맷으로 압축합니다.
        self.data, self.slices = self.collate(data_list)

def _compute_centroid_tree(payload):
    """워커 프로세스에서 실행. numpy 배열/파이썬 정수만 주고받아서 torch 텐서가 프로세스
    경계를 넘지 않게 한다 — torch의 공유메모리 텐서 리듀서(fd 고갈, file_system 방식의
    디스크 I/O 오버헤드)를 아예 타지 않으므로 더 빠르고 안전하다."""
    edge_index_np, num_nodes = payload
    ret = _traverse_ast_from_edges(edge_index_np, num_nodes)
    return centroid(ret)


def convert_into_easy(dataset, num_workers, chunksize=16):
    raw_list = [dataset[idx] for idx in range(len(dataset))]
    # 워커에는 계산에 필요한 edge_index(numpy)와 num_nodes만 보낸다.
    light_iter = ((d.edge_index.cpu().numpy(), d.num_nodes) for d in raw_list)

    if num_workers <= 1:
        trees = [_compute_centroid_tree(payload) for payload in light_iter]
    else:
        with mp.Pool(processes=num_workers) as pool:
            trees = list(pool.imap(_compute_centroid_tree, light_iter, chunksize=chunksize))

    # 텐서 재조립(data_put_edge)은 메인 프로세스에서 — 단순 concat이라 가볍다.
    new_data_list = [data_put_edge(raw_list[i].clone(), trees[i]) for i in range(len(raw_list))]
    return CustomEasyDataset(new_data_list)