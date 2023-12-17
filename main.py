"""
程序入口
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import mse_loss

import args
from dataset import BasicDataset, MoveLens
from loss import MSELoss
from model import LightGCN
from utils import sample, shuffle, minibatch


def train(dataset: BasicDataset, model: LightGCN, epoch: int) -> float:
    """
    训练 LightGCN 模型
    :param dataset: 数据集
    :param model: 模型
    :param epoch: 当前 epoch
    :return: loss
    """

    model.train()
    mse = MSELoss(model)

    # 采样
    samples = sample(dataset)
    users = torch.Tensor(samples[:, 0]).long().to(args.DEVICE)
    items = torch.Tensor(samples[:, 1]).long().to(args.DEVICE)
    ratings = torch.Tensor(samples[:, 2]).float().to(args.DEVICE)

    # 打乱
    users, items, ratings = shuffle(users, items, ratings)

    # 训练
    total_batch = 0
    aver_loss = 0.
    for (batch_i, (batch_users, batch_items, batch_ratings)) in enumerate(minibatch(users, items, ratings, batch_size=args.BPR_BATCH_SIZE)):
        total_batch += 1
        cri = mse.stageOne(batch_users, batch_items, batch_ratings)
        aver_loss += cri

    aver_loss = aver_loss / total_batch
    return aver_loss


def test(dataset: BasicDataset, model: LightGCN, epoch: int) -> tuple[float, float]:
    """
    测试
    :param dataset: 数据集
    :param model: 模型
    :param epoch: 当前 epoch
    :return: RMSE 和 Recall@K
    """

    model.eval()

    aver_rmse = 0.
    aver_recall = 0.

    with torch.no_grad():
        data = dataset.test_data
        users = data.users

        batch_num = 0

        for (batch_i, batch_users) in enumerate(minibatch(users, batch_size=args.TEST_BATCH_SIZE)):
            K = args.TOPKS[batch_i] if (batch_i < len(args.TOPKS)) else args.TOPKS[len(args.TOPKS) - 1]

            batch_num += 1

            # 计算预测分数
            predict_ratings = model.get_rating(batch_users)

            # 实际分数
            actual_ratings = dataset.test_data.get_rating(batch_users)

            # 使用 RMSE 进行评估
            mse = mse_loss(predict_ratings, actual_ratings)
            rmse = torch.sqrt(mse)

            aver_rmse += rmse

            # 获取前 K 个预测
            _, top_K_items = torch.topk(predict_ratings, k=K)

            # 前 K 个预测的正确性
            user_like_items = data.get_liked_items_of_users(batch_users)
            r = []
            for i in range(len(batch_users)):
                label = [int(item) in user_like_items[i] for item in top_K_items[i]]
                r.append(label)
            r = torch.Tensor(np.array(r).astype('float')).to(args.DEVICE)

            # 使用 Recall@K 进行评估
            correct_num = torch.sum(r, dim=-1).to(args.DEVICE)

            user_like_num = torch.Tensor([len(user_like_items[i]) for i in range(len(batch_users))]).to(args.DEVICE)
            recall = torch.mean(correct_num / user_like_num)
            aver_recall += recall

        aver_rmse /= batch_num
        aver_recall /= batch_num

    return float(aver_rmse), float(aver_recall)


def main():
    args.cprint("[LOAD DATASET AND MODEL]")
    dataset = MoveLens()
    model = LightGCN(dataset)
    model.to(args.DEVICE)
    print()

    args.cprint("[START TRAIN]")

    train_epochs = []
    train_losses = []

    test_epochs = []
    test_rmses = []
    test_recalls = []

    for epoch in range(args.EPOCHS):
        loss = train(dataset, model, epoch)

        train_epochs.append(epoch)
        train_losses.append(loss)

        print(f"EPOCH[{epoch}/{args.EPOCHS}] loss: {loss}")
        if epoch % 10 == 0:
            print()
            args.cprint("[TEST]")

            test_epochs.append(epoch)

            rmse, recall = test(dataset, model, epoch)
            print(f"TEST rmse: {rmse} recall: {recall}")
            print()

            test_rmses.append(rmse)
            test_recalls.append(recall)

    plt.plot(train_epochs, train_losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("train loss curves")
    plt.show()

    plt.plot(test_epochs, test_rmses)
    plt.xlabel("epoch")
    plt.ylabel("rmse")
    plt.title("RMSE curves")
    plt.show()

    plt.plot(test_epochs, test_recalls)
    plt.xlabel("epoch")
    plt.ylabel("recall")
    plt.title("Recall@K curves")
    plt.show()


if __name__ == '__main__':
    main()
