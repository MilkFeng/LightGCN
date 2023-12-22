import torch
from torch import nn

from dataset import BasicDataset

import args


class LightGCN(nn.Module):
    def __init__(self, dataset: BasicDataset):
        """
        初始化 LightGCN 模型
        :param dataset: 数据集
        """
        super().__init__()

        # 训练集数据
        self.dataset = dataset
        self.train_data = self.dataset.train_data

        # 用户和物品的嵌入
        self.user_embeddings = nn.Embedding(
            self.dataset.user_num,
            embedding_dim=args.REC_DIM,
            device=args.DEVICE,
        )
        self.item_embeddings = nn.Embedding(
            self.dataset.item_num,
            embedding_dim=args.REC_DIM,
            device=args.DEVICE,
        )

        # 初始化 embedding
        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)

        print(f'model loaded')

    def propagate(self):
        """
        图卷积操作，用于更新用户和物品的嵌入向量。传播 K 层，计算所有的 e_U^k 和 e_I^k
        :return: (e_U^0, ..., e_U^K)^T, (e_I^0, ..., e_I^K)^T
        """

        # 归一化的邻接矩阵
        graph = self.train_data.graph

        # 用户和物品的嵌入
        # [(e_U^0, e_I^0), ..., (e_U^K, e_I^K)]
        all_embeddings_list = [torch.cat([self.user_embeddings.weight, self.item_embeddings.weight]).to(args.DEVICE)]

        # 传播 K 层
        for _ in range(args.LAYER):
            # 获取 e_U^k 和 e_I^k
            all_embeddings = all_embeddings_list[-1]

            # 计算 e_U^(k+1) 和 e_I^(k+1)
            new_all_embeddings = torch.sparse.mm(graph, all_embeddings).to(args.DEVICE)

            # 保存 e_U^(k+1) 和 e_I^(k+1)
            all_embeddings_list.append(new_all_embeddings)

        # 将所有的 e_U^k 和 e_I^k 拼接起来
        embeddings = torch.stack(all_embeddings_list, dim=1)

        # 求平均
        embeddings = torch.mean(embeddings, dim=1)
        # embeddings = torch.cat([embeddings[:, i, :] for i in range(args.LAYER + 1)], dim=1)

        # 分割
        users, items = torch.split(embeddings, [self.dataset.user_num, self.dataset.item_num])

        return users.to(args.DEVICE), items.to(args.DEVICE)

    def get_ratings(self, users: list[int]):
        """
        获取指定用户对所有物品分数
        """

        # 获取 LGC 的嵌入向量
        all_users, all_items = self.propagate()

        users_emb = all_users[users]
        items_emb = all_items

        # U * DIM, I * DIM -> U * I
        ratings = torch.matmul(users_emb, items_emb.t())
        if args.SIGMOID:
            ratings = torch.sigmoid(ratings)
        if args.CLIP:
            ratings = torch.clip(ratings, 0., 1.)
        return ratings

    def forward(self, users: list[int], items: list[int]):
        """
        前向传播
        """

        # 获取 LGC 的嵌入向量
        all_users, all_items = self.propagate()

        # 获取最终的嵌入向量
        users_emb = all_users[users]
        items_emb = all_items[items]

        # 计算评分
        # U * DIM, U * DIM -> U
        gamma = torch.sum(users_emb * items_emb, dim=1)
        if args.SIGMOID:
            gamma = torch.sigmoid(gamma)
        return gamma
