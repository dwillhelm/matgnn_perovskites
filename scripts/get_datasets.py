#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File    :   get_datasets.py
@Time    :   2022/03/15 09:31:02
@Author  :   Daniel W 
@Version :   1.0
@Contact :   willhelmd@tamu.edu
'''

#%%
import os 
from pathlib import Path 

# get some working paths
basedir = Path(os.path.dirname(os.path.abspath(__file__)))
datadir = Path(basedir.joinpath('../data/'))

def dgl_dataset_qm9(label_keys=None): 
    if label_keys is None: 
        print('')

# get QM9 dataset for graph regression
from dgl.data import QM9Dataset
label_keys = ['gap','U0']
dataset = QM9Dataset(label_keys=label_keys,
                     raw_dir=datadir.joinpath('dgl_datasets/'))
dataset



#%%