#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File    :   pgnn_redict_gap.py
@Time    :   2022/04/06 20:30:46
@Author  :   Daniel W 
@Version :   1.0
@Contact :   willhelmd@tamu.edu
'''

#%% 
import numpy as np 
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
# from matgnn.data.datasets import get_dgl_dataset_QM9
from matgnn.data.datasets import get_matminer_dielectric_constant
from matgnn.data.datasets import get_matminer_castelli_perovskites

## load dataset
# dataset = get_dgl_dataset_QM9(label_keys=['gap'])
tdf = dgl.data.GINDataset('PROTEINS', self_loop=True)
dataset = get_matminer_dielectric_constant() 

# get a subsample for testing
# dataset = dataset.sample(n=100)

## get structures and target labels
structures = dataset.structure
targets    = dataset.n

#%% 
# convert targets to tensor()
targets = [torch.FloatTensor(np.array([i])) for i in targets.tolist()]

# convert structures into graphs via CGCNN 
from deepchem.feat.material_featurizers import CGCNNFeaturizer
fzr = CGCNNFeaturizer(radius=8.0)
graphs = [ i.to_dgl_graph() for i in fzr.featurize(structures.tolist())] 

data = tuple(zip(graphs, targets))

#%% 
## Define a data loader
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
num_examples = len(data)
num_train = int(num_examples * 0.70)
num_val   = int(num_examples * 0.20)
num_test  = int(num_examples * 0.10)

idx_train, idx_test = train_test_split(np.arange(len(data)), train_size=0.75,random_state=123)
idx_train, idx_val = train_test_split(idx_train, train_size=0.75,random_state=123)

train_sampler = SubsetRandomSampler(torch.LongTensor(idx_train))
val_sampler   = SubsetRandomSampler(torch.LongTensor(idx_val))
test_sampler  = SubsetRandomSampler(torch.LongTensor(idx_test))

#%% 
train_dataloader = GraphDataLoader(
    data, sampler=train_sampler, batch_size=10, drop_last=False)

val_dataloader   = GraphDataLoader(
    data, sampler=train_sampler, batch_size=10, drop_last=False)

test_dataloader  = GraphDataLoader(
    data, sampler=test_sampler, batch_size=10, drop_last=False)

#%% 
## Define a GNN model
from dgl.nn import GraphConv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

## Create the model with given dimensions
n_epoch = 100
dim_nfeats = 92
model = GCN(dim_nfeats, 16, 1)
model.to(device)
print(f'{device = }\n{n_epoch = }')

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

hist = dict()
hist['epoch'] = list()
hist['epoch_loss'] = list()

for epoch in range(n_epoch):
    acc_loss = []
    for b, (batched_graph, labels) in enumerate(train_dataloader):
        pred = model(batched_graph, batched_graph.ndata['x'].float())
        loss = F.mse_loss(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc_loss.append(loss.item())
    epoch_loss = sum(acc_loss)/len(acc_loss)
    print(f'Epoch: {epoch} Loss: {epoch_loss}')
    hist['epoch'].append(epoch)
    hist['epoch_loss'].append(sum(acc_loss)/len(acc_loss))


abserr = []  

for batched_graph, labels in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata['x'].float())
    er = (labels - pred)
    abserr = abserr + list(er)

hist = pd.DataFrame(hist)
mae = sum(abserr)/len(abserr)
print(f'MAE = {mae}')

#%% 
import matplotlib.pyplot as plt
import pandas as pd 


# plot loss 
fig, ax = plt.subplots()
ax.plot(hist.epoch, hist.epoch_loss)
ax.grid(True,alpha=0.2)
ax.set_xlabel('Epoch')
ax.set_xlabel('MSE Loss')
fig.tight_layout() 
plt.savefig('fig-LossPlot.pdf')
# %%
plt.close() 


