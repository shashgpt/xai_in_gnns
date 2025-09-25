import torch
from torch import nn
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch_geometric.data import HeteroData
from captum.attr import IntegratedGradients

# ------------------
# 0) Toy hetero model for node classification on target node type
# ------------------
class HeteroGAT(nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, target_ntype):
        super().__init__()
        self.target_ntype = target_ntype
        # First layer: per-relation GAT
        self.conv1 = HeteroConv({
            rel: GATConv((-1, -1), hidden_channels, add_self_loops=False)
            for rel in metadata[1]  # list of (src, rel, dst)
        }, aggr='sum')

        # Second layer: per-relation GAT
        self.conv2 = HeteroConv({
            rel: GATConv((-1, -1), hidden_channels, add_self_loops=False)
            for rel in metadata[1]
        }, aggr='sum')

        # Linear heads per node type (just doing one for target is fine)
        self.heads = nn.ModuleDict()
        for ntype in metadata[0]:  # list of node types
            self.heads[ntype] = Linear(hidden_channels, out_channels if ntype == target_ntype else hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        # x_dict: {ntype: [N_ntype, F_ntype]}
        h = self.conv1(x_dict, edge_index_dict)
        h = {k: torch.relu(v) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict)
        h = {k: torch.relu(v) for k, v in h.items()}
        out_dict = {}
        for ntype, x in h.items():
            out_dict[ntype] = self.heads[ntype](x)
        return out_dict  # e.g., out_dict[target_ntype] -> [N_target, C]
    

# ------------------
# 1) Assume you already have a HeteroData graph (x per node type, edge_index per relation)
# ------------------
# Example placeholders (replace with your real data):
data = HeteroData()
# data['paper'].x = torch.randn(P, Fp)
# data['author'].x = torch.randn(A, Fa)
# data[('author','writes','paper')].edge_index = ...
# data[('paper','cites','paper')].edge_index = ...
# etc.

target_ntype = 'patient'         # node type you are explaining
num_classes = 5                  # set to your real num classes
hidden_channels = 64

metadata = (list(data.node_types), list(data.edge_types))
model = HeteroGAT(metadata, hidden_channels, num_classes, target_ntype)
model.eval()

# (Load trained weights here)
# model.load_state_dict(torch.load('...'))

# ------------------
# 2) Build inputs for IG: tuple of tensors (one per node type), all require_grad=True
# ------------------
# We’ll keep a consistent order of node types:
ntypes = list(data.node_types)

x_list = []
for ntype in ntypes:
    x = data[ntype].x
    assert x is not None, f"Missing x for node type {ntype}"
    x = x.clone().detach().requires_grad_(True)
    x_list.append(x)
inputs = tuple(x_list)  # Captum accepts Tuple[Tensor, ...]

# Baselines (same shapes, e.g. zeros)
baselines = tuple(torch.zeros_like(x) for x in inputs)

# Edge indices etc. are passed as non-differentiable args:
edge_index_dict = {k: v for k, v in data.edge_index_dict.items()}

# ------------------
# 3) Define a scalar-returning forward() for IG
# ------------------
# We want the logit for (node_idx, class_id) on the target node type.
def forward_for_ig(*feat_tuple, node_idx, class_id):
    # Rebuild x_dict by the same ntype order:
    x_dict = {ntype: feat for ntype, feat in zip(ntypes, feat_tuple)}
    out = model(x_dict, edge_index_dict)  # dict: {ntype: [N, C]}
    logits = out[target_ntype]            # [N_target, C]
    # Return a scalar tensor for IG:
    return logits[node_idx, class_id]

# ------------------
# 4) Run IG for a specific node and class
# ------------------
ig = IntegratedGradients(forward_for_ig)

node_idx = 12          # <- the specific target node you want to explain (0..N_target-1)
class_id = 3           # <- the class index you want attribution for

attributions = ig.attribute(
    inputs=inputs,
    baselines=baselines,
    additional_forward_args=(node_idx, class_id),
    n_steps=64,  # typical 32–256; higher = smoother but slower
    method='gausslegendre'  # good default for smoothness
)
# 'attributions' is a tuple aligned with 'inputs': one attribution tensor per node type.
# Each tensor has the same shape as its input features.

# ------------------
# 5) Post-process: per-feature importance (e.g., L1 along feature dim)
# ------------------
attr_by_ntype = {}
for ntype, attr in zip(ntypes, attributions):
    # attr shape: [N_ntype, F_ntype]; take |.| and sum over features for each node as a quick view
    per_node_score = attr.abs().sum(dim=-1)          # [N_ntype]
    per_feature_score = attr.abs().sum(dim=0)        # [F_ntype]
    attr_by_ntype[ntype] = {
        'raw': attr,                       # full attribution tensor
        'per_node_score': per_node_score,  # influence of each node of this type
        'per_feature_score': per_feature_score  # influence of each feature of this type
    }

# Example: the top-10 most influential features for the target node type:
topk = 10
v, i = torch.topk(attr_by_ntype[target_ntype]['per_feature_score'], k=min(topk, attr_by_ntype[target_ntype]['per_feature_score'].numel()))
print("Top features (indices) for", target_ntype, ":", i.tolist())
print("Scores:", v.tolist())