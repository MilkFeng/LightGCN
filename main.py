"""
程序入口
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.nn.functional import mse_loss

import args
from dataset import BasicDataset, load_dataset
from loss import EnhancedBPRLossModule, BasicLossModule
from model import LightGCN
from utils import sample, shuffle, minibatch

def train(dataset: BasicDataset, model: LightGCN, loss: BasicLossModule, epoch: int) -> float:
    """
    使用 Enhanced BPR Loss 训练 LightGCN 模型
    :param dataset: 数据集
    :param model: 模型
    :param loss: 损失
    :param epoch: 当前 epoch
    :return: loss
    """

    model.train()
    opt = optim.Adam(loss.parameters(), lr=args.LR)

    # 采样
    samples = sample(dataset)
    users = torch.Tensor(samples[:, 0]).long().to(args.DEVICE)
    pos_items = torch.Tensor(samples[:, 1]).long().to(args.DEVICE)
    neg_items = torch.Tensor(samples[:, 2]).long().to(args.DEVICE)
    actual_scores = torch.Tensor(samples[:, 3]).float().to(args.DEVICE)

    # 打乱
    users, pos_items, neg_items, actual_scores = shuffle(users, pos_items, neg_items, actual_scores)

    # 训练
    total_batch = 0
    aver_loss = 0.

    for (batch_i, (batch_users, batch_pos, batch_neg, batch_actual_scores)) in enumerate(minibatch(users, pos_items, neg_items, actual_scores, batch_size=args.BATCH_SIZE)):
        total_batch += 1

        # 获取损失
        cri = loss(batch_users, batch_pos, batch_neg, batch_actual_scores)
        aver_loss += cri

        opt.zero_grad()
        cri.backward()
        opt.step()

    aver_loss = aver_loss / total_batch
    return float(aver_loss)


def get_rmse(predict_ratings, actual_ratings):
    """
    计算 RMSE
    :param predict_ratings: 预测分数
    :param actual_ratings: 实际分数
    :return: RMSE
    """
    mse = mse_loss(predict_ratings, actual_ratings)
    rmse = torch.sqrt(mse)
    return float(rmse)


def get_recall_and_precision(predict_ratings: torch.Tensor, user_like_items: list[set[int]], k: int):
    """
    计算 Recall@K 和 Precision@K
    :param predict_ratings: 预测分数
    :param user_like_items: 用户喜欢的物品
    :param k: K
    :return: Recall@K, Precision@K
    """
    user_num = predict_ratings.shape[0]

    _, top_K_items = torch.topk(predict_ratings, k=k)

    # 前 K 个预测的正确性
    r = []
    for i in range(user_num):
        label = [int(item) in user_like_items[i] for item in top_K_items[i]]
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float')).to(args.DEVICE)

    # 正确的数量
    correct_num = torch.sum(r, dim=-1).to(args.DEVICE)

    # 用户喜欢的物品数量
    user_like_num = torch.Tensor([len(user_like_items[i]) for i in range(user_num)]).to(args.DEVICE)

    # 计算 Recall@K
    recall = torch.mean(correct_num / user_like_num)

    # 计算 Precision@K
    precision = torch.mean(correct_num) / k
    return float(recall), float(precision)


def test(dataset: BasicDataset, model: LightGCN, epoch: int):
    """
    测试
    :param dataset: 数据集
    :param model: 模型
    :param epoch: 当前 epoch
    :return: RMSE, Recall@K 和 Precision@K
    """

    model.eval()

    aver_rmse = 0.
    aver_recall = 0.
    aver_precision = 0.

    with torch.no_grad():
        data = dataset.test_data
        users = data.users

        batch_num = 0

        for (batch_i, batch_users) in enumerate(minibatch(users, batch_size=args.TEST_BATCH_SIZE)):
            K = args.TOPK

            batch_num += 1

            # 计算预测分数
            predict_ratings = model.get_ratings(batch_users)

            # 实际分数
            actual_ratings = dataset.test_data.get_rating(batch_users)

            # 使用 RMSE 进行评估
            rmse = get_rmse(predict_ratings, actual_ratings)
            aver_rmse += rmse

            # 用户喜欢的物品
            user_like_items = dataset.test_data.get_liked_items_of_users(batch_users)

            # 使用 Recall@K 和 Precision@K 进行评估
            recall, precision = get_recall_and_precision(predict_ratings, user_like_items, K)
            aver_recall += recall
            aver_precision += precision

        aver_rmse /= batch_num
        aver_recall /= batch_num
        aver_precision /= batch_num

    return aver_rmse, aver_recall, aver_precision


def main():
    args.cprint("[LOAD DATASET AND MODEL]")
    dataset = load_dataset(args.DATASET_NAME)

    model = LightGCN(dataset)
    model.to(args.DEVICE)

    loss = EnhancedBPRLossModule(model)
    loss.to(args.DEVICE)

    print()

    args.cprint("[START TRAIN]")

    train_epochs = []
    train_losses = []

    test_epochs = []
    test_rmses = []
    test_recalls = []
    test_precisions = []

    for epoch in range(args.EPOCHS + 1):
        # 训练
        if epoch > 0:
            cri = train(dataset, model, loss, epoch)

            train_epochs.append(epoch)
            train_losses.append(cri)

            print(f"EPOCH[{epoch}/{args.EPOCHS}] loss: {cri}")

        # 每 10 个 epoch 测试一次
        if epoch % 10 == 0:
            print()
            args.cprint("[TEST]")

            test_epochs.append(epoch)

            rmse, recall, precision = test(dataset, model, epoch)
            print(f"TEST rmse: {rmse} recall@{args.TOPK}: {recall} precision@{args.TOPK}: {precision}")
            print()

            test_rmses.append(rmse)
            test_recalls.append(recall)
            test_precisions.append(precision)

    # 保存模型参数
    torch.save(model.state_dict(), os.path.join(args.CURRENT_FILE_PATH, f"checkpoint.pth"))

    # 画图并保存
    print()
    args.cprint("[START PLOT]")

    plt.plot(train_epochs, train_losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"train {'MSE' if args.MSE else 'BPR'} Loss curves")
    plt.savefig(os.path.join(args.CURRENT_FILE_PATH, "train_loss.png"))
    plt.show()

    plt.plot(test_epochs, test_rmses)
    plt.xlabel("epoch")
    plt.ylabel("rmse")
    plt.title("RMSE curves")
    plt.savefig(os.path.join(args.CURRENT_FILE_PATH, "rmse.png"))
    plt.show()

    plt.plot(test_epochs, test_recalls, label='recall')
    plt.plot(test_epochs, test_precisions, label='precision')
    plt.legend()
    plt.xlabel("epoch")
    plt.title(f"Recall@{args.TOPK} and Precision@{args.TOPK} curves")
    plt.savefig(os.path.join(args.CURRENT_FILE_PATH, "recall_precision.png"))
    plt.show()


if __name__ == '__main__':
    main()
