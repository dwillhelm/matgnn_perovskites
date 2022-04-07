#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File    :   util.py
@Time    :   2022/03/15 09:26:50
@Author  :   Daniel W 
@Version :   1.0
@Contact :   willhelmd@tamu.edu

Util script for data related tasks

'''
import os
from pathlib import Path 

from dgl.data import QM9Dataset
from matminer.datasets import convenience_loaders
from matminer.datasets.dataset_retrieval import load_dataset
from torch import load

# get some working paths
basedir = Path(os.path.dirname(os.path.abspath(__file__)))
topdir  = basedir.joinpath('../../')

# script level variables
MATMINER_DATA_HOME = topdir.joinpath('data/matminer_datasets')


def get_dgl_dataset_QM9(label_keys=['gap','U0'], cutoff=5.0,force_reload=False):
    dataset = QM9Dataset(label_keys=label_keys, cutoff=cutoff,
                         force_reload=force_reload, 
                         raw_dir=topdir.joinpath('data/dgl_datasets')) 
    return dataset

def get_matbench_dielectric():
    dataset = load_dataset('matbench_dielectric',data_home=MATMINER_DATA_HOME)
    return dataset
    
    

def get_matminer_castelli_perovskites(): 
    dataset = convenience_loaders.load_castelli_perovskites(data_home=MATMINER_DATA_HOME, download_if_missing=True)
    return dataset

def get_matminer_dielectric_constant(): 
    dataset = convenience_loaders.load_dielectric_constant(data_home=MATMINER_DATA_HOME, download_if_missing=True)
    return dataset 