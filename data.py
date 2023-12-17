"""
一些数据类
"""
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

class Data(object):
    """
    数据，存储所有边和邻接矩阵
    """
    def __init__(
            self,
            user_num: int,
            item_num: int,
            edges: list[Edge],
            graph: torch.IntTensor,
    ):
        """
        :param user_num: 所用用户数量
        :param item_num: 所有物品数量
        :param edges: 所有边
        :param graph: 邻接矩阵
        """
        self.__user_num = user_num
        self.__item_num = item_num

        self.edges = edges
        self.edge_num = len(self.edges)

        self.users: list[int] = list(set([edge.user for edge in edges]))

        # |I,   R|
        # |R^T, I|
        self.graph = graph

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
        :param user: 用户
        :return:
        """
        return set([edge.item for edge in self.edges if edge.user == user])

    def get_liked_items_of_users(self, users: list[int]) -> list[set[int]]:
        """
        获得喜欢的物品
        :param users: 用户列表
        :return:
        """
        return [self.get_liked_items(user) for user in users]

    def __str__(self):
        return f"Data(edges={self.edges}, graph={self.graph})"
