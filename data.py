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
            graph: torch.IntTensor,
    ):
        """
        :param user_num: 所用用户数量
        :param item_num: 所有物品数量
        :param edges: 所有边
        :param graph: 邻接矩阵
        """
        self.edges = edges
        self.edge_num = len(self.edges)

        self.users: list[int] = list(set([edge.user for edge in edges]))

        self.edges_of_users = {user: set() for user in range(user_num)}
        for edge in edges:
            self.edges_of_users[edge.user].add(edge)

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

    def __str__(self):
        return f"Data(edges={self.edges}, graph={self.graph})"
