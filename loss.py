#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/10/27 下午8:31
# @Author  : chuyu zhang
# @File    : loss.py
# @Software: PyCharm

import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch


class FocalLoss(nn.Module):
    def __init__(self, class_num=5, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        if P.dim() > 2:
            P = P.view(P.size(0), P.size(1), -1)
            P = P.permute(0, 2, 1).contiguous()
            P = P.view(-1, P.size(-1))
        # print(P.shape)
        # print(targets.shape)
        ids = targets.view(-1, 1)
        class_mask = inputs.data.new(ids.size(0), C).fill_(0)
        class_mask = Variable(class_mask)

        # print(ids.shape)
        class_mask = class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)
        if class_mask.device != P.device:
            class_mask = class_mask.to(P.device)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss