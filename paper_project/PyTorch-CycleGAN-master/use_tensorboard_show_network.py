"""
-*- codeing = utf-8 -*-
作者：ZHG
日期：2021年10月28日
"""
import models
import tensorboardX as tx
import numpy as np
import torch
writer = tx.SummaryWriter(logdir="logs") #创建
inputs = torch.from_numpy(np.random.rand(1,3,28,28)).type(torch.FloatTensor) # B D W H

writer.add_graph(models.Discriminator(3),(inputs,))



