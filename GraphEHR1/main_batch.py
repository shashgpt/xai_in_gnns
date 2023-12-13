import argparse
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch_geometric.loader import NeighborLoader
from collections import Counter
import pickle
from tqdm import tqdm
from datetime import datetime
from model import *
import os
import logging
from sklearn.metrics import precision_recall_curve, auc

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

from dataset import *
from model import *

data = EHRData().to(device)
model = HGT(data, 64, 1, 4, 2).to(device)
train_loader = NeighborLoader(
    data,
    # Sample 15 neighbors for each node and each edge type for 2 iterations:
    num_neighbors={key: [5] * 2 for key in data.edge_types},
    # Use a batch size of 128 for sampling training nodes of type "paper":
    batch_size=5,
    input_nodes=('patient', data.train_mask),
)

criterion = torch.nn.BCELoss()
# pos_weight = torch.ones(1).float().to(device) * (ratio[True] / ratio[False])
# criterion = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight)      
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #  weight_decay=args.wd

epochs = 100
for epoch in range(epochs):
    model.train()
    loss_total = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict)
    #     print(out[data.train_mask].shape)
    #     print(data["patient"].y[data.train_mask])
        loss = criterion(out.squeeze(-1), batch["patient"].y)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    print(loss_total)
    
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        precision, recall, thresholds = precision_recall_curve(data["patient"].y[data.val_mask].cpu().numpy(), out[data.val_mask].squeeze(-1).detach().cpu().numpy())
        val_auprc = auc(recall, precision)

        precision, recall, thresholds = precision_recall_curve(data["patient"].y[data.test_mask].cpu().numpy(), out[data.test_mask].squeeze(-1).detach().cpu().numpy())
        test_auprc = auc(recall, precision)

        print('AUPRC-Valid:%f; AUPRC-Test: %f' % (val_auprc, test_auprc))

