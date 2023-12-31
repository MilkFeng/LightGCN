"""
数据集，参考官方代码，BasicDataset 是基类，如果想要添加新的数据集的话需要继承
BasicDataset 并实现方法
"""

from abc import ABC, abstractmethod

import pandas as pd
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


class MovieLens(BasicDataset):
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

        # 解析数据
        self.__train_data = self.__decode_data(train_data).to(args.DEVICE)
        self.__test_data = self.__decode_data(test_data).to(args.DEVICE)

        print(f"MovieLens Dataset loaded\n"
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

        # 所有边
        edges = [Edge(u, i, r) for u, i, r in zip(users, items, ratings)]

        return Data(self.__user_num, self.__item_num, edges).to(args.DEVICE)

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


def load_dataset(name: str) -> BasicDataset:
    all_dataset = ['indonesia_tourism', 'ml-latest-small', 'modcloth']

    if name == 'indonesia_tourism':
        return MovieLens(
            path='data/indonesia_tourism/',
            filename='tourism_rating.csv',
            max_rating=5.0
        )
    elif name == 'ml-latest-small':
        return MovieLens(
            path='data/ml-latest-small/',
            filename='ratings.csv',
            max_rating=5.0
        )
    elif name == 'modcloth':
        return MovieLens(
            path='data/modcloth/',
            filename='ratings.csv',
            max_rating=5.0
        )
    else:
        raise NotImplementedError(f"Haven't supported {name} yet!, try {all_dataset}")
