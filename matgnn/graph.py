#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File    :   graph.py
@Time    :   2022/04/06 21:05:53
@Author  :   Daniel W 
@Version :   1.0
@Contact :   willhelmd@tamu.edu
'''
import numpy as np
import torch

from dgl.convert import graph as dgl_graph
from dgl.transform import to_bidirected

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def connect_within_cutoff(self, coords, cutoff, k=None, strict=None):
    coords = torch.tensor(coords, dtype=torch.float32)
    dist = torch.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    adj = (dist <= cutoff).long() - torch.eye(len(coords))
    if k is not None:
        adj = self.choose_k_nearest_CWC(adj, dist, k, strict)
    adj = adj + adj.T
    adj = adj.bool().to_sparse()
    u, v = adj.indices()
    graph = dgl_graph((u, v))
    graph = to_bidirected(graph)
    graph = graph.to(device)