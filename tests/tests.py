#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File    :   tests.py
@Time    :   2022/04/06 14:58:20
@Author  :   Daniel W 
@Version :   1.0
@Contact :   willhelmd@tamu.edu
'''



# test important modules

def test_pkgs(): 
    print('-Checking Packages')
    import torch; print('\tloading torch')
    import dgl; print('\tloading dgl')
    import torch_geometric; print('\tloading torch_geometric')
    print('\tno pkg erros found')


if __name__ == '__main__':
    
    print('\nRunning Tests')
    test_pkgs() 




# def test_cuda():
#     # test gpu support for DGL
#     print('\n\nTesting CPU <-> GPU passing')
#     import dgl
#     import torch as th
    
#     u, v = th.tensor([0, 1, 2]), th.tensor([2, 3, 4])
#     g = dgl.graph((u, v))
#     print(g)
#     g.ndata['x'] = th.randn(5, 3)  # original feature is on CPU
#     print(f'\tCurrent device = {g.device.type}')
#     print('\tAttempting to pass to GPU')
#     try:
#         cuda_g = g.to('cuda:0')
#         if cuda_g.device != 'gpu':
#             print('\tCannot pass to GPU')
#     except:
#         print('\tPassing Error')