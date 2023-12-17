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

        # Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

        print(f"""model loaded
        """)

    def propagate(self):
        """
        图卷积操作，用于更新用户和物品的嵌入向量。传播 K 层，计算所有的 e_U^k 和 e_I^k
        :return: (e_U^0, ..., e_U^K)^T, (e_I^0, ..., e_I^K)^T
        """

        # 邻接矩阵
        graph = self.train_data.graph

        # 用户和物品的嵌入
        # [(e_U^0, e_I^0), ..., (e_U^K, e_I^K)]
        all_embeddings_list = [torch.cat([self.user_embeddings.weight, self.item_embeddings.weight]).to(args.DEVICE)]

        # 传播 K 层
        for k in range(args.LAYER):
            # 获取 e_U^k 和 e_I^k
            all_embeddings = all_embeddings_list[-1]

            # 计算 e_U^(k+1) 和 e_I^(k+1)
            new_all_embeddings = torch.sparse.mm(graph, all_embeddings)

            # 保存 e_U^(k+1) 和 e_I^(k+1)
            all_embeddings_list.append(new_all_embeddings)

        # 将所有的 e_U^k 和 e_I^k 拼接起来
        embeddings = torch.stack(all_embeddings_list, dim=1)

        # 求平均
        embeddings = torch.mean(embeddings, dim=1)

        # 分割
        users, items = torch.split(embeddings, [self.dataset.user_num, self.dataset.item_num])

        return users.to(args.DEVICE), items.to(args.DEVICE)

    def mse_loss(self, users: list[int], items: list[int], actual_score):
        """
        计算 MSE 损失
        """

        # 获取 LGC 的嵌入向量
        all_users, all_items = self.propagate()

        # 获取最终的嵌入向量
        users_emb = all_users[users]
        items_emb = all_items[items]
        users_emb_ego = self.user_embeddings(users)
        items_emb_ego = self.item_embeddings(items)

        # 计算正则化损失
        reg_loss = 0.5 * (users_emb_ego.norm(2).pow(2) + items_emb_ego.norm(2).pow(2) / float(len(users)))

        # 计算评分
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        gamma = self.sigmoid(gamma)

        # 计算损失
        mse_loss = nn.MSELoss()
        loss = mse_loss(gamma, actual_score)

        return loss, reg_loss

    def get_rating(self, users: list[int]):
        """
        获取分数
        """

        # 获取 LGC 的嵌入向量
        all_users, all_items = self.propagate()

        users_emb = all_users[users]
        items_emb = all_items

        ratings = torch.matmul(users_emb, items_emb.t())
        ratings = self.sigmoid(ratings)
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
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        gamma = self.sigmoid(gamma)
        return gamma
