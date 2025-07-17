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
from model import VariationalGNN
from utils import train, evaluate, EHRData, collate_fn
import os
import logging
import sys
import time

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

# os.PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python train.py

def main():
    parser = argparse.ArgumentParser(description='configuraitons')
    parser.add_argument('--result_path', type=str, default='./result/', help='output path of model checkpoints')
    parser.add_argument('--data_path', type=str, default='./data/mimic/', help='input path of processed dataset')
    parser.add_argument('--embedding_size', type=int, default=256, help='embedding size')
    parser.add_argument('--num_of_layers', type=int, default=2, help='number of graph layers')
    parser.add_argument('--num_of_heads', type=int, default=1, help='number of attention heads')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--label', type=int, default=0, help='label')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout')
    parser.add_argument('--reg', type=str, default="True", help='regularization')
    parser.add_argument('--lbd', type=int, default=1.0, help='regularization')
    parser.add_argument('--trial', type=int, default=0, help='trial indexing')

    args = parser.parse_args()
    result_path = args.result_path + str(args.label)
    data_path = args.data_path + "/" + str(args.label) + "/"
    in_feature = args.embedding_size
    out_feature = args.embedding_size
    n_layers = args.num_of_layers - 1
    lr = args.lr
    args.reg = (args.reg == "True")
    n_heads = args.num_of_heads
    dropout = args.dropout
    alpha = 0.1
    BATCH_SIZE = args.batch_size
    number_of_epochs = 50
    eval_freq = 1000

    # Load data
    train_x, train_y = pickle.load(open(data_path + 'train_csr.pkl', 'rb'))
    val_x, val_y = pickle.load(open(data_path + 'validation_csr.pkl', 'rb'))
    test_x, test_y = pickle.load(open(data_path + 'test_csr.pkl', 'rb'))
    train_upsampling = np.concatenate((np.arange(len(train_y)), np.repeat(np.where(train_y == 1)[0], 1)))
    train_x = train_x[train_upsampling]
    train_y = train_y[train_upsampling]

    # Create result root
    s = datetime.now().strftime('%Y%m%d%H%M%S')
    result_root = '%s/lr_%s-input_%s-output_%s-dropout_%s/%d'%(result_path, lr, in_feature, out_feature, dropout, args.trial)
    if not os.path.exists(result_root):
        os.makedirs(result_root)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename='%s/train.log' % result_root, format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info("Time:%s" %(s))

    # initialize models
    num_of_nodes = train_x.shape[1] + 1
    device_ids = range(torch.cuda.device_count())
    # eICU has 1 feature on previous readmission that we didn't include in the graph
    model = VariationalGNN(in_feature, out_feature, num_of_nodes, n_heads, n_layers, dropout=dropout, alpha=alpha, variational=args.reg, none_graph_features=0).to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    val_loader = DataLoader(dataset=EHRData(val_x, val_y), batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=torch.cuda.device_count(), shuffle=False)
    test_loader = DataLoader(dataset=EHRData(test_x, test_y), batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=torch.cuda.device_count(), shuffle=False)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train models
    max_valid_auprc = 0
    max_test_auprc = 0
    start_time = time.time()
    for epoch in range(number_of_epochs):
        print("Learning rate:{}".format(optimizer.param_groups[0]['lr']))
        ratio = Counter(train_y)
        train_loader = DataLoader(dataset=EHRData(train_x, train_y), batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=torch.cuda.device_count(), shuffle=True)
        pos_weight = torch.ones(1).float().to(device) * (ratio[True] / ratio[False])
        criterion = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight)
        t = tqdm(iter(train_loader), leave=False, total=len(train_loader))
        model.train()
        total_loss = np.zeros(3)
        for idx, batch_data in enumerate(t):
            loss, kld, bce = train(batch_data, model, optimizer, criterion, args.lbd, 5)
            total_loss += np.array([loss, bce, kld])
#         if idx % eval_freq == 0 and idx > 0:
#         torch.save(model.state_dict(), "{}/parameter{}_{}".format(result_root, epoch, idx))
        val_auprc, _ = evaluate(model, val_loader, len(val_y))
        logging.info('epoch:%d AUPRC:%f; loss: %.4f, bce: %.4f, kld: %.4f' % (epoch + 1, val_auprc, total_loss[0]/idx, total_loss[1]/idx, total_loss[2]/idx))
        
        test_auprc, _ = evaluate(model, test_loader, len(val_y))
        if max_valid_auprc < val_auprc:
            max_valid_auprc = val_auprc
            max_test_auprc = test_auprc
        logging.info('[END] AUPRC-Valid:%f; AUPRC-Test: %f' % (val_auprc, test_auprc))
        
        print('epoch:%d AUPRC:%f; loss: %.4f, bce: %.4f, kld: %.4f' % (epoch + 1, val_auprc, total_loss[0]/idx, total_loss[1]/idx, total_loss[2]/idx))
#         if idx % 50 == 0 and idx > 0:
        t.set_description('[epoch:%d] loss: %.4f, bce: %.4f, kld: %.4f' %
                          (epoch + 1, total_loss[0]/idx, total_loss[1]/idx, total_loss[2]/idx))
        t.refresh()
        scheduler.step()
        
    end_time = time.time()
    with open(result_root + "/time.txt", "w") as f:
        f.write(str(end_time - start_time) + " second\n")
    
    print("End Training")
    val_auprc, _ = evaluate(model, val_loader, len(val_y))
    print(max_valid_auprc, max_test_auprc)
    test_auprc, _ = evaluate(model, test_loader, len(val_y))
    logging.info('[END] AUPRC-Valid:%f; AUPRC-Test: %f' % (val_auprc, test_auprc))
    print('[END] AUPRC-Valid:%f; AUPRC-Test: %f' % (val_auprc, test_auprc))

if __name__ == '__main__':
    main()
