import argparse
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from collections import Counter
import pickle
from tqdm import tqdm
from datetime import datetime
from model import *
import os
import logging
import time
from sklearn.metrics import precision_recall_curve, auc

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

from dataset import *
from model import *

import argparse

parser = argparse.ArgumentParser(description='configuraitons')
parser.add_argument('--embedding_size', type=int, default=64, help='embedding size')
parser.add_argument('--num_layers', type=int, default=2, help='number of graph layers')
parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=50, help='num epochs')
parser.add_argument('--trial', type=int, default=0, help='trial indexing')
parser.add_argument('--label', type=int, default=0, help='label indexing')

args = parser.parse_args()

data = EHRData(args.label)
model = HGT(data, args.embedding_size, 1, args. num_heads, args.num_layers).to(device)
data = data.to(device)

criterion = torch.nn.BCELoss()
# pos_weight = torch.ones(1).float().to(device) * (ratio[True] / ratio[False])
# criterion = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight)      
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) #  weight_decay=args.wd

dirname = "result/%d/%d_%d_%d_%.4f/%d/" % (args.label, args.embedding_size, args.num_heads, args.num_layers, args.lr, args.trial)
if os.path.isdir(dirname) is False:
    os.makedirs(dirname)
if os.path.isfile(dirname + "log.txt"):
    os.remove(dirname + "log.txt")

epochs = args.epochs
max_valid_auprc = 0
max_test_auprc = 0
start_time = time.time()
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
#     print(out[data.train_mask].shape)
#     print(data["patient"].y[data.train_mask])
    loss = criterion(out[data.train_mask].squeeze(-1), data["patient"].y[data.train_mask])
    loss.backward()
    optimizer.step()
    print(loss.item())
    with open(dirname + "log.txt", "+a") as f:
        f.write("Train Loss: "+ str(loss.item()) + "\n")
    
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        precision, recall, thresholds = precision_recall_curve(data["patient"].y[data.val_mask].cpu().numpy(), out[data.val_mask].squeeze(-1).detach().cpu().numpy())
        val_auprc = auc(recall, precision)

        precision, recall, thresholds = precision_recall_curve(data["patient"].y[data.test_mask].cpu().numpy(), out[data.test_mask].squeeze(-1).detach().cpu().numpy())
        test_auprc = auc(recall, precision)

        print('AUPRC-Valid:%f; AUPRC-Test: %f' % (val_auprc, test_auprc))
        if max_valid_auprc < val_auprc:
            max_valid_auprc = val_auprc
            max_test_auprc = test_auprc
        with open(dirname + "log.txt", "+a") as f:
            f.write("AUPRC-Valid: " + str(val_auprc) +"\tAUPRC-Test: " + str(test_auprc) + "\n")
    print(max_valid_auprc, max_test_auprc)
end_time = time.time()

with open(dirname + "logtime.txt", "w") as f:
    f.write(str(end_time - start_time) + " second\n")