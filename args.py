"""
一些全局参数，parse_args 是从官方代码里面抄过来的，有一些没用的参数
"""
import argparse

import torch


def parse_args():
    """
    解析命令行参数
    :return: parser
    """
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int, default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int, default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int, default=50,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str, default='gowalla',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str, default="lgn")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lmse', help='rec-model, support [mf, lgn, lmse]')
    parser.add_argument('--mse', type=int, default=1, help='Use MSELoss or not, affects mf only, lgn uses bpr, lmse must use MSELoss')
    parser.add_argument('--sigmoid', type=int, default=1, help='whether we use sigmoid activation, support [mf(with mse=1), lmse]')
    parser.add_argument('--clip', type=int, default=0, help='whether we clip output between 0-5, support [mf(with mse=1), lmse]')

    return parser.parse_args()


def cprint(msg):
    print(f"\033[0;30;43m{msg}\033[0m")


__args = parse_args()

REC_DIM: int = __args.recdim
BPR_BATCH_SIZE: int = __args.bpr_batch
LR: float = __args.lr
TEST_BATCH_SIZE: int = __args.testbatch
LAYER: int = __args.layer
EPOCHS: int = __args.epochs
DECAY: float = __args.decay
TOPKS: list[int] = eval(__args.topks)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cprint("[LOAD ARGUMENTS]")
print(f"""REC_DIM: {REC_DIM}
BPR_BATCH_SIZE: {BPR_BATCH_SIZE}
LR: {LR}
TEST_BATCH_SIZE: {TEST_BATCH_SIZE}
LAYER: {LAYER}
EPOCHS: {EPOCHS}
DECAY: {DECAY}
TOPKS: {TOPKS}
""")
print()
