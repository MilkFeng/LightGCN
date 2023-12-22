"""
损失函数类
"""
from abc import ABC, abstractmethod

import torch
from torch import optim, nn

import args
from model import LightGCN


class BasicLossModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, users, pos_items, neg_items, actual_score):
        """
        计算损失
        """
        pass


class EnhancedBPRLossModule(BasicLossModule):
    def __init__(self, model: LightGCN):
        super(EnhancedBPRLossModule, self).__init__()
        self.model = model

        print('Enhanced BPR Loss loaded')

    def get_bpr_loss(self, users, pos, neg):
        """
        计算 BPR 损失
        """

        all_users, all_items = self.model.propagate()

        # 获取最终的嵌入向量
        users_emb_ego = self.model.user_embeddings(users)
        pos_emb_ego = self.model.item_embeddings(pos)
        neg_emb_ego = self.model.item_embeddings(neg)

        users_emb = all_users[users]
        pos_emb = all_items[pos]
        neg_emb = all_items[neg]

        # 计算正则化损失
        reg_loss = 0.5 * (users_emb_ego.norm(2).pow(2) + pos_emb_ego.norm(2).pow(2) + neg_emb_ego.norm(2).pow(2)) / float(len(users))

        # 计算评分
        # U * DIM, U * DIM -> U
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        if args.SIGMOID:
            pos_scores = torch.sigmoid(pos_scores)
            neg_scores = torch.sigmoid(neg_scores)

        # 计算损失
        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return bpr_loss, reg_loss

    def get_mse_loss(self, users, items, actual_score):
        """
        计算 MSE 损失
        """

        # 获取 LGC 的嵌入向量
        all_users, all_items = self.model.propagate()

        # 获取最终的嵌入向量
        users_emb = all_users[users]
        items_emb = all_items[items]
        users_emb_ego = self.model.user_embeddings(users)
        items_emb_ego = self.model.item_embeddings(items)

        # 计算正则化损失
        reg_loss = 0.5 * (users_emb_ego.norm(2).pow(2) + items_emb_ego.norm(2).pow(2) / float(len(users)))

        # 计算评分
        # U * DIM, U * DIM -> U
        gamma = torch.sum(users_emb * items_emb, dim=1)
        if args.SIGMOID:
            gamma = torch.sigmoid(gamma)

        # 计算损失
        mse_loss = nn.MSELoss()
        loss = mse_loss(gamma, actual_score)

        return loss, reg_loss

    def forward(self, users, pos, neg, actual_scores):
        # 计算BPR损失和评分损失
        bpr_loss, reg_bpr_loss = self.get_bpr_loss(users, pos, neg)
        mse_loss, reg_mse_loss = self.get_mse_loss(users, pos, actual_scores)

        # 添加L2正则化项
        bpr_loss = bpr_loss ** 0.5 + args.DECAY * reg_bpr_loss
        mse_loss = mse_loss ** 0.5 + args.DECAY * reg_mse_loss

        # 最终损失为BPR损失和评分损失的加权和
        total_loss = (bpr_loss + mse_loss) / 2

        return total_loss
