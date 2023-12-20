"""
损失函数类
"""
from torch import optim

import args
from model import LightGCN


class MSELoss:
    def __init__(self, model: LightGCN):
        self.model = model
        self.weight_decay = args.DECAY
        self.lr = args.LR
        self.opt = optim.Adam(model.parameters(), lr=self.lr)

    def stageOne(self, users: list[int], items: list[int], actual_score):
        loss, reg_loss = self.model.mse_loss(users, items, actual_score)
        reg_loss = reg_loss * self.weight_decay
        loss = loss ** 0.5
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


class BPRLoss:
    def __init__(self, model: LightGCN):
        self.model = model
        self.weight_decay = args.DECAY
        self.lr = args.LR
        self.opt = optim.Adam(model.parameters(), lr=self.lr)

    def stageOne(self, users: list[int], pos: list[int], neg: list[int]):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()
