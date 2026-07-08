import numpy as np
import torch
import scipy.sparse as sp
from numpy.linalg import inv
import pickle

from torch_geometric.datasets import *

import torch
import numpy as np
from torch_sparse.matmul import matmul
from torch_sparse import SparseTensor
from torch_geometric.data import InMemoryDataset

from utils.utils_centroid import get_centroid_tree


c = 0.15
k = 5


def adj_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def get_intimacy_matrix(edges,n):
    edges= np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n,n),
                        dtype=np.float32)
    print('normalize')
    adj_norm = adj_normalize(adj)
    print('inverse')
    eigen_adj = c * inv((sp.eye(adj.shape[0]) - (1 - c) * adj_norm).toarray())

    return eigen_adj


def adj_normalize_sparse(mx):
    mx=mx.to(device)
    rowsum = mx.sum(1)
    r_inv =rowsum.pow(-0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = SparseTensor(row = torch.arange(n).to(device),col=torch.arange(n).to(device),value=r_inv, sparse_sizes=(n,n))
    nr_mx = matmul(matmul(r_mat_inv,mx),r_mat_inv)
    return nr_mx

def get_intimacy_matrix_sparse(edges,n):
    adj = SparseTensor(row=edges[0], col=edges[1], value=torch.ones(edges.shape[1]), sparse_sizes=(n, n))
    adj_norm = adj_normalize_sparse(adj)
    return adj_norm

def get_svd_dense(mx,q=3):
    mx = mx.float()
    u,s,v = torch.svd_lowrank(mx,q=q)
    s=torch.diag(s)
    pu = u@s.pow(0.5)
    pv = v@s.pow(0.5)
    return pu,pv


def unweighted_adj_normalize_dense_batch(adj):
    adj = (adj+adj.transpose(-1,-2)).bool().float()
    adj = adj.float()
    rowsum = adj.sum(-1)
    r_inv = rowsum.pow(-0.5)
    r_mat_inv = torch.diag_embed(r_inv)
    nr_adj = torch.matmul(torch.matmul(r_mat_inv,adj),r_mat_inv)
    return nr_adj


def get_eig_dense(adj):
    adj = adj.float()
    rowsum = adj.sum(1)
    r_inv =rowsum.pow(-0.5)
    r_mat_inv = torch.diag(r_inv)
    nr_adj = torch.matmul(torch.matmul(r_mat_inv,adj),r_mat_inv)
    graph_laplacian = torch.eye(adj.shape[0])-nr_adj
    L,V = torch.eig(graph_laplacian,eigenvectors=True)
    return L.T[0],V



def check_checkpoints(output_dir):
    import os
    import shutil
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        for file in files:
            if 'checkpoint' in file:

                return True
        print('remove ',output_dir)
        shutil.rmtree(output_dir)
    return False


# ============================================
# AST Node Encoding and Graph Manipulation
# ============================================

class ASTNodeEncoder(torch.nn.Module):
    """
    AST 노드 인코딩 모듈
    
    Args:
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
    """
    시퀀스 리스트에서 어휘 매핑 생성
    
    Args:
        seq_list: 시퀀스 리스트
        num_vocab: 어휘 크기
        
    Returns:
        vocab2idx: 단어 -> 인덱스 매핑
        idx2vocab: 인덱스 -> 단어 리스트
    """
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
    """
    시퀀스를 텐서 배열로 인코딩
    
    Args:
        data: PyG Data 객체
        vocab2idx: 단어 -> 인덱스 매핑
        max_seq_len: 최대 시퀀스 길이
        
    Returns:
        data: y_arr이 추가된 Data 객체
    """
    seq = data.y
    augmented = seq[:max_seq_len] + ['__EOS__'] * max(0, max_seq_len - len(seq))
    data.y_arr = torch.tensor(
        [[vocab2idx.get(w, vocab2idx['__UNK__']) for w in augmented]], dtype=torch.long
    )
    return data


def decode_arr_to_seq(arr, idx2vocab):
    """
    인덱스 배열을 시퀀스로 디코딩
    
    Args:
        arr: 인덱스 텐서 배열
        idx2vocab: 인덱스 -> 단어 리스트
        
    Returns:
        시퀀스 (단어 리스트)
    """
    eos_pos = torch.nonzero(arr == len(idx2vocab)-1, as_tuple=False)
    arr = arr[:torch.min(eos_pos).item()] if len(eos_pos) > 0 else arr
    return [idx2vocab[i.item()] for i in arr]


def augment_edge(data):
    """
    양방향 엣지로 그래프 증강
    
    Args:
        data: PyG Data 객체
        
    Returns:
        data: 양방향 엣지가 추가된 Data 객체
    """
    ei = data.edge_index
    ea = torch.zeros(ei.size(1), 2)
    ei_inv = torch.stack([ei[1], ei[0]], dim=0)
    ea_inv = torch.cat([torch.zeros(ei_inv.size(1),1), torch.ones(ei_inv.size(1),1)], dim=1)

    data.edge_index = torch.cat([ei, ei_inv], dim=1)
    data.edge_attr  = torch.cat([ea, ea_inv], dim=0)
    return data


def augment_edge_with_leaf_edge(data):
    """
    속성이 있는 노드 간 엣지 추가로 그래프 증강
    
    Args:
        data: PyG Data 객체
        
    Returns:
        data: 리프 엣지가 추가된 Data 객체
    """
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


def traverse_ast(data):
    """
    AST를 DFS로 순회하여 각 노드의 부모를 반환
    
    Args:
        data: PyG Data 객체 (AST)
        
    Returns:
        ret: 각 노드의 부모 노드 인덱스 배열
    """
    ret = list(0 for _ in range(data.num_nodes))
    
    visited = [False] * data.num_nodes
    stack = [(0, 0)] 
    adj = [[] for _ in range(data.num_nodes)]
    
    # 텐서를 넘파이 배열로 변환 후, C-level 속도로 루프 수행
    edges = data.edge_index.cpu().numpy()
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


def centroid(tree_data):
    """
    트리를 centroid tree로 변환
    
    Args:
        tree_data: 각 노드의 부모 배열
        
    Returns:
        centroid tree 구조
    """
    n = len(tree_data)
    edge_data = []

    for i in range(n):
        edge_data.append([i, tree_data[i]])

    tree_ans = get_centroid_tree(n, edge_data)
    return tree_ans


def data_put_edge(data, tree):
    """
    centroid tree 엣지를 PyG Data 객체에 추가
    
    Args:
        data: PyG Data 객체
        tree: centroid tree 구조
        
    Returns:
        data: centroid tree 엣지가 추가된 Data 객체
    """
    col, row = [], []

    for i in tree: 
        col.append(i[0])
        row.append(i[1])

    col = torch.tensor(col, dtype=torch.long)
    row = torch.tensor(row, dtype=torch.long)

    edge_attr = torch.ones(len(col), 2)
    edge_index = torch.stack([col, row], dim=0)
    edge_attr_inv = torch.ones(len(col), 2)
    edge_index_inv = torch.stack([row, col], dim=0)
    
    data.edge_index = torch.cat([data.edge_index, edge_index_inv, edge_index], dim=1)
    data.edge_attr = torch.cat([data.edge_attr, edge_attr_inv, edge_attr], dim=0)

    return data


class CustomEasyDataset(InMemoryDataset):
    """
    Centroid tree 엣지가 추가된 커스텀 PyG 데이터셋
    """
    def __init__(self, data_list):
        super().__init__(None)
        # collate 함수를 통해 리스트를 PyG 데이터셋 포맷으로 압축합니다.
        self.data, self.slices = self.collate(data_list)


def convert_into_easy(dataset):
    """
    데이터셋의 모든 그래프에 centroid tree 엣지 추가
    
    Args:
        dataset: PyG 데이터셋
        
    Returns:
        CustomEasyDataset: centroid tree 엣지가 추가된 커스텀 데이터셋
    """
    new_data_list = []

    for idx in range(len(dataset)):
        tmp_data = dataset[idx]
        data = tmp_data.clone()
        
        ret = traverse_ast(data)
        tree = centroid(ret)
        new_data_list.append(data_put_edge(data, tree))

    return CustomEasyDataset(new_data_list)


if __name__=='__main__':
    #just test

    device = torch.device('cuda',0)

    data = Flickr('dataset/flickr')

    edges= data.data.edge_index
    n=data.data.x.shape[0]


    adj = SparseTensor(row=edges[0], col=edges[1], value=torch.ones(edges.shape[1]), sparse_sizes=(n, n))
    nr_adj = adj_normalize_sparse(adj)

    pu,pv= get_svd_dense(nr_adj.to_torch_sparse_coo_tensor(),q=10)


    adj= (torch.randn(10,10)>0).float()
    L,V = get_eig_dense(adj)
