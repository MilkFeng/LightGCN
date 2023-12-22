"""
一些数据类
"""
import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor


class Edge(object):
    """
    边
    """

    def __init__(self, user: int, item: int, rating: float):
        self.user = user
        self.item = item
        self.rating = rating

    def __str__(self):
        return f"Edge(user={self.user}, item={self.item}, rating={self.rating})"

    def __getitem__(self, item):
        if item == 0:
            return self.user
        elif item == 1:
            return self.item
        elif item == 2:
            return self.rating
        else:
            raise IndexError("Edge index out of range")


class Data(object):
    """
    数据，存储所有边和邻接矩阵
    """
    def __init__(
            self,
            user_num: int,
            item_num: int,
            edges: list[Edge],
    ):
        """
        :param user_num: 所用用户数量
        :param item_num: 所有物品数量
        :param edges: 所有边
        """
        self.edges = edges
        self.edge_num = len(self.edges)

        self.users: list[int] = list(set([edge.user for edge in edges]))

        self.edges_of_users = {user: set() for user in range(user_num)}
        for edge in edges:
            self.edges_of_users[edge.user].add(edge)

        self.graph = self.__get_graph(user_num, item_num)

        self.rating = torch.zeros(size=[user_num, item_num], dtype=torch.float)

        for edge in edges:
            self.rating[edge.user, edge.item] = edge.rating

        pass

    def to(self, device) -> 'Data':
        self.graph = self.graph.to(device)
        self.rating = self.rating.to(device)
        return self

    def get_rating(self, users: list[int]) -> Tensor:
        """
        得到评分
        :param users: 用户列表
        :return:
        """
        return self.rating[users]

    def get_liked_items(self, user: int) -> set[int]:
        """
        获得喜欢的物品
        """
        return set([edge.item for edge in self.edges_of_users[user]])

    def get_liked_items_of_users(self, users: list[int]) -> list[set[int]]:
        """
        获得喜欢的物品
        """
        return [self.get_liked_items(user) for user in users]

    def get_liked_items_with_rating(self, user: int) -> set[tuple[int, float]]:
        """
        获得喜欢的物品和评分
        """
        return set([(edge.item, edge.rating) for edge in self.edges_of_users[user]])

    def get_liked_items_with_rating_of_users(self, users: list[int]) -> list[set[tuple[int, float]]]:
        """
        获得喜欢的物品和评分
        """
        return [self.get_liked_items_with_rating(user) for user in users]

    def __get_graph(self, user_num: int, item_num: int) -> torch.Tensor:
        """
        生成归一化的邻接矩阵
        """

        graph = sp.dok_matrix((user_num + item_num, user_num + item_num), dtype=np.float32)
        graph = graph.tolil()

        R = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        for u, i, r in self.edges:
            R[u, i] = r
        R = R.tolil()

        # |0,   R|
        # |R^T, 0|
        graph[:user_num, user_num:] = R
        graph[user_num:, :user_num] = R.T

        # 生成度矩阵
        row_sum = np.array(graph.sum(axis=1))
        degree_inv = np.power(row_sum, -0.5).flatten()
        degree_inv[np.isinf(degree_inv)] = 0.
        degree_inv_mat = sp.diags(degree_inv)

        # 归一化邻接矩阵
        graph = (degree_inv_mat @ graph @ degree_inv_mat).tocsr()

        # 转化为 tensor
        coo = graph.tocoo().astype(np.float32)
        index = torch.stack([torch.Tensor(coo.row).long(), torch.Tensor(coo.col).long()])
        data = torch.Tensor(coo.data).float()
        graph = torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

        return graph

    def __str__(self):
        return f"Data(edges={self.edges}, graph={self.graph})"
