import torch
import numpy as np
import pandas as pd
import math
import random
import torch_geometric
import torch_geometric.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.data import HeteroData
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HGTConv, Linear
from sklearn.metrics import precision_recall_curve, auc
from torch_geometric.explain import Explainer, CaptumExplainer
from captum.attr import IntegratedGradients
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch_geometric.data import HeteroData
from captum.attr import IntegratedGradients

def seed_everything(seed = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set as {seed}")
seed_everything(42)

device = 'cuda'
feature_index = 0

config = dict()
config["lr"] = 0.001
config["weight_decay"] = 5e-3
config["epochs"] = 1
config['train_ratio'] = 0.8
config['val_ratio'] = 0.1
config['test_ratio'] = 0.1
config['hidden_channels'] = 64
config['num_heads'] = 4
config['num_layers'] = 2

#Creating Graph out of EHR dataset
# 1. Load the nodes, edges, and labels
patient_features = torch.tensor(np.load('data/Processed/patient_features.npy')).to(torch.float32)
procedure_features = torch.tensor(np.load('data/Processed/procedure_features.npy')).to(torch.float32)
medication_features = torch.tensor(np.load('data/Processed/medication_features.npy')).to(torch.float32)
lab_features = torch.tensor(np.load('data/Processed/lab_features.npy')).to(torch.float32)
patient_edges = torch.tensor(np.load('data/Processed/patient_edges.npy'))
procedure_edges = torch.tensor(np.load('data/Processed/procedures_edges.npy'))
medication_edges = torch.tensor(np.load('data/Processed/medication_edges.npy'))
lab_edges = torch.tensor(np.load('data/Processed/lab_edges.npy'))
labels = torch.tensor(np.load('data/Processed/MIMIC_y.npy')[:, feature_index]).to(torch.float32)
# print(len(patient_features), len(procedure_features), len(medication_features), len(lab_features), len(labels))
print(patient_features.shape, procedure_features.shape, medication_features.shape, lab_features.shape, labels.shape)
# print(len(patient_edges), len(procedure_edges), len(medication_edges), len(lab_edges))
print(patient_edges.shape, procedure_edges.shape, medication_edges.shape, lab_edges.shape)

num_patient_nodes = len(patient_features)
num_train_nodes = int(config['train_ratio'] * num_patient_nodes)
num_val_nodes = int(config['val_ratio'] * num_patient_nodes)
num_test_nodes = num_patient_nodes - num_train_nodes - num_val_nodes

# patient_nodes_indices = torch.arange(num_patient_nodes)
# shuffled_indices = torch.randperm(num_patient_nodes)

patient_nodes_indices = np.arange(num_patient_nodes)
np.random.shuffle(patient_nodes_indices)
shuffled_indices = torch.tensor(patient_nodes_indices)

train_indices = shuffled_indices[:num_train_nodes]
val_indices = shuffled_indices[num_train_nodes:num_train_nodes + num_val_nodes]
test_indices = shuffled_indices[num_train_nodes + num_val_nodes:]

train_mask = torch.zeros(num_patient_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_patient_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_patient_nodes, dtype=torch.bool)

train_mask[patient_nodes_indices[train_indices]] = True
val_mask[patient_nodes_indices[val_indices]] = True
test_mask[patient_nodes_indices[test_indices]] = True

# print(len(train_mask), train_mask)
# print(len(val_mask), val_mask)
# print(len(test_mask), test_mask)

# data = HeteroData()
# data['patient'].x = patient_features
# data['patient'].y = labels
# data['patient'].train_mask = train_mask
# data['patient'].val_mask = val_mask
# data['patient'].test_mask = test_mask
# data['procedure'].x = procedure_features
# data['medication'].x = medication_features
# data['lab'].x = lab_features
# data['patient', 'same_patient', 'patient'].edge_index = patient_edges
# data['patient', 'proc_values', 'procedure'].edge_index = procedure_edges
# data['patient', 'med_values', 'medication'].edge_index = medication_edges
# data['patient', 'lab_values', 'lab'].edge_index = lab_edges

# data = T.ToUndirected()(data)
# # data = T.AddSelfLoops()(data)
# # data = T.NormalizeFeatures()(data)
# data = data.to(device)

# print(data.has_isolated_nodes(), data.has_self_loops(), data.is_undirected())
# print(len(patient_edges))
# print(len(patient_features))
# print(len(procedure_features))
# print(len(medication_features))
# print(len(lab_features))

#Visualising the Heterogeneous graph
# 1. Create a sample HeteroData object
data = HeteroData()

# Example
# data['paper'].x = torch.randn(5, 16)   # 5 papers (5 nodes of papers), 16 features
# data['author'].x = torch.randn(3, 8)   # 3 authors (3 nodes of authors), 8 features
# data['author', 'writes', 'paper'].edge_index = torch.tensor([[0, 1, 2], [0, 2, 4]]) # 2, number of edges = 3
# data['paper', 'cites', 'paper'].edge_index = torch.tensor([[0, 1], [1, 3]])         # 2, number of edges = 2

# # Example: Two node types: 'paper' and 'author'
# num_papers = 5
# num_authors = 5
# num_paper_features = 64
# num_author_features = 32
# data['paper'].x = torch.randn(num_papers, num_paper_features)
# data['author'].x = torch.randn(num_authors, num_author_features)

# # Example: Two edge types: 'cites' (paper->paper) and 'writes' (author->paper)
# # 'cites' and
# num_cites_edges = 5
# source_paper_indices = torch.randint(0, num_papers, (num_cites_edges,)) # 200 numbers between 0 to 100(number of paper nodes)
# target_paper_indices = torch.randint(3, num_papers, (num_cites_edges,)) # 200 numbers between 0 to 100(number of paper nodes)
# data['paper', 'cites', 'paper'].edge_index = torch.stack([source_paper_indices, target_paper_indices], dim=0)
# # 'writes' edges
# num_writes_edges = 5
# author_indices = torch.randint(0, num_authors, (num_writes_edges,))
# paper_indices = torch.randint(0, num_papers, (num_writes_edges,))
# data['author', 'writes', 'paper'].edge_index = torch.stack([author_indices, paper_indices], dim=0)

# # Example: Adding features to 'cites' edges
# num_cites_edge_features = 16
# data['paper', 'cites', 'paper'].edge_attr = torch.randn(num_cites_edges, num_cites_edge_features)

source_indices = patient_edges[0][:5]
target_indices = patient_edges[1][:5]
data['patient', 'same_patient', 'patient'].edge_index = torch.stack([source_indices, target_indices], dim=0)
source_indices = procedure_edges[0][:5]
target_indices = procedure_edges[1][:5]
data['patient', 'proc_values', 'procedure'].edge_index = torch.stack([source_indices, target_indices], dim=0)
source_indices = medication_edges[0][:5]
target_indices = medication_edges[1][:5]
data['patient', 'med_values', 'medication'].edge_index = torch.stack([source_indices, target_indices], dim=0)
source_indices = lab_edges[0][:5]
target_indices = lab_edges[1][:5]
data['patient', 'lab_values', 'lab'].edge_index = torch.stack([source_indices, target_indices], dim=0)

# 2. Prepare for NetworkX
G = nx.DiGraph()
node_colors = {}
node_shapes = {}
edge_colors = {}

# Add nodes and edges
for edge_type in data.edge_types:
    src_type, rel_type, dst_type = edge_type
    edge_index = data[edge_type].edge_index
    for i in range(edge_index.size(1)):
        src_node_idx = edge_index[0, i].item()
        dst_node_idx = edge_index[1, i].item()
        print(src_node_idx, dst_node_idx)
        src_node_id = f"{src_type}_{src_node_idx}"
        dst_node_id = f"{dst_type}_{dst_node_idx}"
        G.add_edge(src_node_id, dst_node_id, type=rel_type)
        if rel_type == 'same_patient':
            edge_colors[(src_node_id, dst_node_id)] = 'blue'
        elif rel_type == 'proc_values':
            edge_colors[(src_node_id, dst_node_id)] = 'green'
        elif rel_type == 'med_values':
            edge_colors[(src_node_id, dst_node_id)] = 'yellow'
        elif rel_type == 'lab_values':
            edge_colors[(src_node_id, dst_node_id)] = 'brown'
# # Add edges
# for edge_type in data.edge_types:
#     src_type, rel_type, dst_type = edge_type
#     edge_index = data[edge_type].edge_index
#     # print(edge_index.size(1))
#     for i in range(edge_index.size(1)):
#         src_node_idx = edge_index[0, i].item()
#         dst_node_idx = edge_index[1, i].item()
#         print(src_node_idx, dst_node_idx)
#         src_node_id = f"{src_type}_{src_node_idx}"
#         dst_node_id = f"{dst_type}_{dst_node_idx}"
#         G.add_edge(src_node_id, dst_node_id, type=rel_type)
#         if rel_type == 'writes':
#             edge_colors[(src_node_id, dst_node_id)] = 'blue'
#         elif rel_type == 'cites':
#             edge_colors[(src_node_id, dst_node_id)] = 'green'
for node in G.nodes():
    print(node)
for edge in G.edges():
    print(edge)
    
# Add nodes
for node in G.nodes():
    if 'patient' in node:
        node_colors[node] = 'grey'
        node_shapes[node] = 'o' # Circle
    elif 'procedure' in node:
        node_colors[node] = 'purple'
        node_shapes[node] = 's' # Square
    elif 'medication' in node:
        node_colors[node] = 'lightgreen'
        node_shapes[node] = 't' # Triangle
    elif 'lab' in node:
        node_colors[node] = 'lightcoral'
        node_shapes[node] = 'a' # Star
# # # Add nodes
# # for node_type in data.node_types:
# #     for i in range(data[node_type].num_nodes):
# #         node_id = f"{node_type}_{i}"
# #         G.add_node(node_id, type=node_type)
# #         if node_type == 'paper':
# #             node_colors[node_id] = 'skyblue'
# #             node_shapes[node_id] = 'o' # Circle
# #         elif node_type == 'author':
# #             node_colors[node_id] = 'lightcoral'
# #             node_shapes[node_id] = 's' # Square
# for node in G.nodes():
#     print(node)

# 3. Visualize with Matplotlib
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42) # Choose a layout algorithm

# Draw nodes
nx.draw_networkx_nodes(G, pos, 
                       node_color=[node_colors[node] for node in G.nodes()],
                       node_shape='o', 
                       cmap=plt.cm.get_cmap('Pastel1'), 
                       node_size=1000) # Use a single shape for simplicity here, or iterate for different shapes

# Draw edges
nx.draw_networkx_edges(G, pos, edge_color=[edge_colors.get((u, v), 'gray') for u, v in G.edges()], arrowsize=20)

# Draw labels
node_labels = {node: node.split('_')[0] for node in G.nodes()} # Show only node type
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
edge_labels = {(u, v): G[u][v]['type'] for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.3)

# data = T.ToUndirected()(data)
# # data = T.AddSelfLoops()(data)
# # data = T.NormalizeFeatures()(data)
# data = data.to(device)

plt.title("Heterogeneous Graph Visualization")
plt.axis('off')
plt.savefig('heterogeneous_graph_data_visualisation.png')

# class HGT(torch.nn.Module):
#     def __init__(self, hidden_channels, out_channels, num_heads, num_layers, num_pat, num_proc, num_med, num_lab):
#         super().__init__()
#         self.lin_dict = torch.nn.ModuleDict()
#         for node_type in data.node_types:
#             self.lin_dict[node_type] = Linear(-1, hidden_channels)
            
#         # self.lin_dict["patient"] = Linear(num_pat, hidden_channels)
#         # # self.lin_dict["patient_64"] = Linear(64, hidden_channels)

#         # self.lin_dict["procedure"] = Linear(num_proc, hidden_channels)
#         # # self.lin_dict["procedure_64"] = Linear(64, hidden_channels)

#         # self.lin_dict["medication"] = Linear(num_med, hidden_channels)
#         # # self.lin_dict["medication_64"] = Linear(64, hidden_channels)

#         # self.lin_dict["lab"] = Linear(num_lab, hidden_channels)
#         # self.lin_dict["lab_64"] = Linear(64, hidden_channels)

#         self.convs = torch.nn.ModuleList()
#         for _ in range(num_layers):
#             # conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads, group='sum'): gives the following error: TypeError: MessagePassing.__init__() got an unexpected keyword argument 'group'
#             conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads)
#             self.convs.append(conv)

#         self.lin = Linear(hidden_channels, out_channels)

#     def forward(self, x_dict, edge_index_dict):
#         for node_type, x in x_dict.items():
#             print(x.shape)
#             # if x.shape[1] == 64 and node_type == "patient":
#             #   x_dict[node_type] = self.lin_dict["patient_64"](x).relu_()
#             # elif x.shape[1] == 64 and node_type == "procedure":
#             #   x_dict[node_type] = self.lin_dict["procedure_64"](x).relu_()
#             # elif x.shape[1] == 64 and node_type == "medication":
#             #   x_dict[node_type] = self.lin_dict["medication_64"](x).relu_()
#             # elif x.shape[1] == 64 and node_type == "lab":
#             #   x_dict[node_type] = self.lin_dict["lab_64"](x).relu_()
#             # else:
#             x_dict[node_type] = self.lin_dict[node_type](x).relu_()

#         for conv in self.convs:
#             x_dict = conv(x_dict, edge_index_dict)

#         out = self.lin(x_dict['patient'])
#         out = F.sigmoid(out)
#         return out

# model = HGT(hidden_channels=64,
#             out_channels=1,
#             num_heads=2,
#             num_layers=2,
#             num_pat=3,
#             num_proc=len(procedure_features),
#             num_med=len(medication_features),
#             num_lab=len(lab_features)).to(device)
# # for param in model.lin_dict["patient_64"].parameters():
# #     param.requires_grad = False
# # for param in model.lin_dict["procedure_64"].parameters():
# #     param.requires_grad = False
# # for param in model.lin_dict["medication_64"].parameters():
# #     param.requires_grad = False
# # for param in model.lin_dict["lab_64"].parameters():
# #     param.requires_grad = False
# # for name, param in model.named_parameters():
# #     print(name, param.requires_grad)
# print(model)

# losses = []
# auprc_val = []
# auprc_test = []
# auprc_train = []
# acc_val = []
# acc_test = []
# acc_train = []

# with torch.no_grad():
#     out = model(data.x_dict, data.edge_index_dict)
# optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x_dict, data.edge_index_dict)
#     mask = data['patient'].train_mask
# #     print(out[mask].squeeze()[:2], data['patient'].y[mask][:2])
#     for index, value in enumerate(data['patient'].y[mask].long()):
#         if value not in [0, 1]:
#             # print(index, value)
#             continue
#     criterion = torch.nn.BCELoss()
#     loss = criterion(out[mask].squeeze(), data['patient'].y[mask])
#     loss.backward()
#     optimizer.step()
#     train_acc = ((out[mask].detach().cpu().numpy().squeeze()>0.5).astype(int) == data['patient'].y[mask].detach().cpu().numpy()).sum()/len(out[mask])
#     precision, recall, thresholds = precision_recall_curve(data["patient"].y[mask].cpu().numpy(), out[mask].squeeze().detach().cpu().numpy())
#     train_auprc = auc(recall, precision)
#     return train_acc, train_auprc, float(loss)

# @torch.no_grad()
# def test():
#     model.eval()
#     out = model(data.x_dict, data.edge_index_dict)
#     val_mask = data['patient'].val_mask
#     precision, recall, thresholds = precision_recall_curve(data["patient"].y[val_mask].cpu().numpy(), out[val_mask].squeeze().detach().cpu().numpy())
#     val_auprc = auc(recall, precision)
#     test_mask = data['patient'].test_mask
#     precision, recall, thresholds = precision_recall_curve(data["patient"].y[test_mask].cpu().numpy(), out[test_mask].squeeze().detach().cpu().numpy())
#     test_auprc = auc(recall, precision)
#     val_acc = ((out[val_mask].detach().cpu().numpy().squeeze()>0.5).astype(int) == data['patient'].y[val_mask].detach().cpu().numpy()).sum()/len(out[val_mask])
#     test_acc = ((out[test_mask].detach().cpu().numpy().squeeze()>0.5).astype(int) == data['patient'].y[test_mask].detach().cpu().numpy()).sum()/len(out[test_mask])
#     return (val_auprc, test_auprc, val_acc, test_acc)

# for epoch in range(1, config["epochs"]):
#     train_acc, train_auprc, loss = train()
#     val_auprc, test_auprc, val_acc, test_acc = test()
#     losses.append(loss)
#     auprc_train.append(train_auprc)
#     acc_train.append(train_acc)
#     auprc_val.append(val_auprc)
#     acc_val.append(val_acc)
#     auprc_test.append(test_auprc)
#     acc_test.append(test_acc)
#     print(f'{epoch:03d}, Loss: {loss:.3f}, TrPRC: {train_auprc:.3f}, TrAcc: {train_acc:.3f}, VaPRC: {val_auprc:.3f}, VaAcc: {val_acc:.3f}, TePRC: {test_auprc:.3f}, TeAcc: {test_acc:.3f}')

# # print(help(CaptumExplainer))
# #  |  * :class:`captum.attr.IntegratedGradients`
# #  |  * :class:`captum.attr.Saliency`
# #  |  * :class:`captum.attr.InputXGradient`
# #  |  * :class:`captum.attr.Deconvolution`
# #  |  * :class:`captum.attr.ShapleyValueSampling`
# #  |  * :class:`captum.attr.GuidedBackprop`

# # Custom subclass that allows unused gradients
# explainer = Explainer(
#     model,  #It is assumed that model outputs a single tensor.
#     algorithm = CaptumExplainer('IntegratedGradients'),
#     explanation_type = 'model',
#     node_mask_type = 'attributes',
#     edge_mask_type = 'object',
#     model_config = dict(
#         mode = 'multiclass_classification',
#         task_level = 'node',
#         return_type = 'probs', #Model returns probabilities.
#     ),
# )
# # Generate batch-wise heterogeneous explanations for
# # the nodes at index `1` and `3`:
# hetero_explanation = explainer(
#                               data.x_dict,
#                               data.edge_index_dict,
#                               index=torch.tensor([1, 2]),
#                               )

# print(hetero_explanation.edge_mask_dict)
# print(hetero_explanation.node_mask_dict)
# print(data.edge_index_dict.keys())
# print(data.edge_index_dict[('patient', 'same_patient', 'patient')])
# print(data.edge_index_dict[('patient', 'same_patient', 'patient')].shape)

# # ------------------
# # 0) Toy hetero model for node classification on target node type
# # ------------------
# class HeteroGAT(torch.nn.Module):
#     def __init__(self, metadata, hidden_channels, out_channels, target_ntype):
#         super().__init__()
#         self.target_ntype = target_ntype
#         # First layer: per-relation GAT
#         self.conv1 = HeteroConv({
#             rel: GATConv((-1, -1), hidden_channels, add_self_loops=False)
#             for rel in metadata[1]  # list of (src, rel, dst)
#         }, aggr='sum')

#         # Second layer: per-relation GAT
#         self.conv2 = HeteroConv({
#             rel: GATConv((-1, -1), hidden_channels, add_self_loops=False)
#             for rel in metadata[1]
#         }, aggr='sum')

#         # Linear heads per node type (just doing one for target is fine)
#         self.heads = torch.nn.ModuleDict()
#         for ntype in metadata[0]:  # list of node types
#             self.heads[ntype] = Linear(hidden_channels, out_channels if ntype == target_ntype else hidden_channels)

#     def forward(self, x_dict, edge_index_dict):
#         # x_dict: {ntype: [N_ntype, F_ntype]}
#         h = self.conv1(x_dict, edge_index_dict)
#         h = {k: torch.relu(v) for k, v in h.items()}
#         h = self.conv2(h, edge_index_dict)
#         h = {k: torch.relu(v) for k, v in h.items()}
#         out_dict = {}
#         for ntype, x in h.items():
#             out_dict[ntype] = self.heads[ntype](x)
#         return out_dict  # e.g., out_dict[target_ntype] -> [N_target, C]

# # ------------------
# # 1) Assume you already have a HeteroData graph (x per node type, edge_index per relation)
# # ------------------
# # Example placeholders (replace with your real data):
# # data = HeteroData()
# # data['paper'].x = torch.randn(P, Fp)
# # data['author'].x = torch.randn(A, Fa)
# # data[('author','writes','paper')].edge_index = ...
# # data[('paper','cites','paper')].edge_index = ...
# # etc.

# target_ntype = 'patient'         # node type you are explaining
# num_classes = 4                  # set to your real num classes
# hidden_channels = 64

# metadata = (list(data.node_types), list(data.edge_types))
# model = HeteroGAT(metadata, hidden_channels, num_classes, target_ntype)
# model.eval()

# # ------------------
# # 2) Build inputs for IG: tuple of tensors (one per node type), all require_grad=True
# # ------------------
# # We’ll keep a consistent order of node types:
# ntypes = list(data.node_types)

# x_list = []
# for ntype in ntypes:
#     x = data[ntype].x
#     assert x is not None, f"Missing x for node type {ntype}"
#     x = x.clone().detach().requires_grad_(True)
#     x_list.append(x)
# inputs = tuple(x_list)  # Captum accepts Tuple[Tensor, ...]

# # Baselines (same shapes, e.g. zeros)
# baselines = tuple(torch.zeros_like(x) for x in inputs)

# # Edge indices etc. are passed as non-differentiable args:
# print(data.edge_index_dict.items())
# edge_index_dict = {k: v for k, v in data.edge_index_dict.items()}

# # ------------------
# # 3) Define a scalar-returning forward() for IG
# # ------------------
# # We want the logit for (node_idx, class_id) on the target node type.
# def forward_for_ig(*feat_tuple, node_idx, class_id):
#     x_dict = {ntype: feat for ntype, feat in zip(ntypes, feat_tuple)} # Rebuild x_dict by the same ntype order:
#     out = model(x_dict, edge_index_dict)  # dict: {ntype: [N, C]}
#     logits = out[target_ntype]            # [N_target, C]
#     return logits[node_idx, class_id]     # Return a scalar tensor for IG:

# # ------------------
# # 4) Run IG for a specific node and class
# # ------------------
# node_idx = 12          # <- the specific target node you want to explain (0..N_target-1)
# class_id = 3           # <- the class index you want attribution for
# ig = IntegratedGradients(forward_for_ig)
# attributions = ig.attribute(
#     inputs=inputs,
#     baselines=baselines,
#     additional_forward_args=(node_idx, class_id),
#     n_steps=64,  # typical 32–256; higher = smoother but slower
#     method='gausslegendre'  # good default for smoothness
# )
# # 'attributions' is a tuple aligned with 'inputs': one attribution tensor per node type.
# # Each tensor has the same shape as its input features.

# # ------------------
# # 5) Post-process: per-feature importance (e.g., L1 along feature dim)
# # ------------------
# attr_by_ntype = {}
# for ntype, attr in zip(ntypes, attributions):
#     # attr shape: [N_ntype, F_ntype]; take |.| and sum over features for each node as a quick view
#     per_node_score = attr.abs().sum(dim=-1)          # [N_ntype]
#     per_feature_score = attr.abs().sum(dim=0)        # [F_ntype]
#     attr_by_ntype[ntype] = {
#         'raw': attr,                       # full attribution tensor
#         'per_node_score': per_node_score,  # influence of each node of this type
#         'per_feature_score': per_feature_score  # influence of each feature of this type
#     }

# # Example: the top-10 most influential features for the target node type:
# topk = 10
# v, i = torch.topk(attr_by_ntype[target_ntype]['per_feature_score'], k=min(topk, attr_by_ntype[target_ntype]['per_feature_score'].numel()))
# print("Top features (indices) for", target_ntype, ":", i.tolist())
# print("Scores:", v.tolist())