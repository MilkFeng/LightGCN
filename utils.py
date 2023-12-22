"""
一些工具函数
"""
import numpy as np
import torch

import args
from dataset import BasicDataset

def sample(dataset: BasicDataset) -> np.ndarray:
    """
    BPR 负采样
    :param dataset: 数据集
    :return: 采样结果，[[user, pos_item, neg_item, pos_item_rating], ...]
    """

    data = dataset.train_data

    # 采样用户
    users = data.users

    # 获取用户喜欢的物品
    liked_items = data.get_liked_items_of_users(users)

    samples = []
    for i, user in enumerate(users):
        # 获取用户喜欢的物品
        pos_items = list(liked_items[i])

        if len(pos_items) == 0:
            continue

        # 采样正样本
        pos_item = np.random.choice(pos_items)

        # 获取正样本评分
        pos_item_rating = data.get_rating([user])[0, pos_item].cpu().item()

        # 采样负样本
        neg_item = np.random.randint(dataset.item_num)
        while neg_item in pos_items:
            neg_item = np.random.randint(dataset.item_num)

        samples.append([user, pos_item, neg_item, pos_item_rating])

    return np.array(samples)


def shuffle(*arrays, **kwargs):
    """
    打乱
    """

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def minibatch(*tensors, **kwargs):
    """
    生成 mini-batch
    """

    batch_size = kwargs.get('batch_size', args.BATCH_SIZE)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


set_seed(args.SEED)
