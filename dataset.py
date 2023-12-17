"""
数据集，参考官方代码，BasicDataset 是基类，如果想要添加新的数据集的话需要继承
BasicDataset 并实现方法
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import args
from data import Data, Edge


class BasicDataset(ABC):
    @property
    @abstractmethod
    def train_data(self) -> Data:
        """
        训练数据
        """
        pass

    @property
    @abstractmethod
    def test_data(self) -> Data:
        """
        测试数据
        """
        pass

    @property
    @abstractmethod
    def item_num(self) -> int:
        """
        物品数量
        """
        pass

    @property
    @abstractmethod
    def user_num(self) -> int:
        """
        用户数量
        """
        pass

class MoveLens(BasicDataset):
    def __init__(self, path: str = 'data/modcloth/', filename: str = 'ratings.csv', max_rating: float = 5.0):

        self.max_rating = max_rating

        # 读取数据并去除空值
        dataset = pd.read_csv(path + filename)
        dataset = dataset.dropna()

        # 生成用户和物品的编码
        le = LabelEncoder()
        dataset['userId'] = le.fit_transform(dataset['userId'])
        dataset['movieId'] = le.fit_transform(dataset['movieId'])

        self.__user_num = max(dataset['userId']) + 1
        self.__item_num = max(dataset['movieId']) + 1

        # 按照用户划分数据集
        users = list(set(dataset['userId']))
        train_users, test_users = train_test_split(users, test_size=0.2)
        train_data = dataset[dataset['userId'].isin(train_users)]
        test_data = dataset[dataset['userId'].isin(test_users)]

        # 保存数据集
        train_data.to_csv(path + 'train.csv')
        test_data.to_csv(path + 'test.csv')

        # 解析数据
        self.__train_data = self.__decode_data(train_data).to(args.DEVICE)
        self.__test_data = self.__decode_data(test_data).to(args.DEVICE)

        print(f"MoveLens Dataset loaded\n"
              f"path: {path}\n"
              f"filename: {filename}\n"
              f"max_rating: {max_rating}\n"
              f"user_num: {self.user_num}\n"
              f"item_num: {self.item_num}\n"
              f"train_user_num: {len(train_users)}\n"
              f"test_user_num: {len(test_users)}\n")

    def __decode_data(self, data: pd.DataFrame) -> Data:
        users: list[int] = data['userId'].values
        items: list[int] = data['movieId'].values
        ratings = data['rating'].values

        # 除以最大的评分
        ratings: list[float] = [rating / self.max_rating for rating in ratings]

        # 生成邻接矩阵
        graph = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
        graph = graph.tolil()

        R = sp.dok_matrix((self.user_num, self.item_num), dtype=np.float32)
        for u, i, r in zip(users, items, ratings):
            R[u, i] = r
        R = R.tolil()

        graph[:self.user_num, self.user_num:] = R
        graph[self.user_num:, :self.user_num] = R.T

        # 生成度矩阵
        rowsum = np.array(graph.sum(axis=1))
        degree_inv = np.power(rowsum, -0.5).flatten()
        degree_inv[np.isinf(degree_inv)] = 0.
        degree_mat = sp.diags(degree_inv)

        # 归一化邻接矩阵
        graph = degree_mat.dot(graph).dot(degree_mat).tocsr()

        # 转化为 tensor
        coo = graph.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.Tensor(coo.data)
        graph = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

        # 所有边
        edges = [Edge(u, i, r) for u, i, r in zip(users, items, ratings)]

        return Data(self.__user_num, self.__item_num, edges, graph).to(args.DEVICE)

    @property
    def train_data(self) -> Data:
        return self.__train_data

    @property
    def test_data(self) -> Data:
        return self.__test_data

    @property
    def item_num(self) -> int:
        return self.__item_num

    @property
    def user_num(self) -> int:
        return self.__user_num

    def __str__(self):
        return f"MoveLens(train_data={self.train_data}, test_data={self.test_data})"
