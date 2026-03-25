import torch
import torch.nn as nn
from torch_geometric.data import Data

x = torch.tensor([[-1], [0], [1]], dtype = torch.float)

edge_index = torch.tensor([[0, 1],[1, 0],[1, 2],[2, 1]], dtype = torch.int)

data = Data(x=x, edge_index = edge_index.t().contiguous() )

n_nodes, n_features = data.x.shape

from torch_geometric.utils import add_self_loops,degree
from torch_geometric.nn import GCNConv

convA = GCNConv(1,3)
conv1 = convA(data.x, data.edge_index)

convB = GCNConv(3,2)
conv2 = convB(conv1, data.edge_index)

lin = nn.Linear(1, 3);
print(conv2)

def GCNConv2():
    x = data.x
    edge_index = data.edge_index

    # A -> A~
    edge_index, _= add_self_loops(edge_index, num_nodes=x.size(0))

    out_node, in_node = edge_index

    deg = degree(in_node, x.size(0), dtype=x.dtype)

    # D -> D^-0.5
    deg_inv_sqrt = deg.pow(-0.5)
    norm = deg_inv_sqrt[out_node] * deg_inv_sqrt[in_node]

    message = lin(x[in_node])
    message *= norm.view(-1, 1)
    node_0 = message[out_node == 0].sum(0)
    print(node_0)
    
    node_1 = message[out_node == 1].sum(0)
    print(node_1)

    node_2 = message[out_node == 2].sum(0)
    print(node_2)

GCNConv2()