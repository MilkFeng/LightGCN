"""
一些工具函数
"""
import numpy as np
import torch

import args
from dataset import BasicDataset

def sample(dataset: BasicDataset):
    """
    加权负采样
    :param dataset: 数据集
    :return: 采样结果
    """

    data = dataset.train_data

    items_dict = {}
    for e in data.edges:
        if e.user not in items_dict.keys():
            items_dict[e.user] = []
        items_dict[e.user].append((e.item, e.rating))

    samples = []
    for user in data.users:
        items = items_dict.get(user, [])
        if len(items) == 0:
            continue

        # 取评分低的当做负样本
        ratings = [item[1] for item in items]

        rating_inv_sum = sum([2. - item[1] for item in items])
        weights_inv = [(2. - item[1]) / rating_inv_sum for item in items]

        rating_sum = sum([item[1] for item in items])
        weights = [item[1] / rating_sum for item in items]

        items = [item[0] for item in items]
        items_index = range(len(items))

        neg_item_index = np.random.choice(items_index, p=weights_inv)
        pos_item_index = np.random.choice(items_index, p=weights)

        samples.append([user, items[pos_item_index], ratings[pos_item_index]])
        if neg_item_index != pos_item_index:
            samples.append([user, items[neg_item_index], ratings[neg_item_index]])

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

    batch_size = kwargs.get('batch_size', args.BPR_BATCH_SIZE)

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
