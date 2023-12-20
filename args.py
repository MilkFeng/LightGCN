"""
一些全局参数，parse_args 是从官方代码里面抄过来的，有一些没用的参数
"""
import argparse
import os.path
import time

import torch


def parse_args():
    """
    解析命令行参数
    :return: parser
    """
    parser = argparse.ArgumentParser(description="Go lightGCN")

    parser.add_argument('--batch', type=int, default=2048, help="the batch size for training procedure")
    parser.add_argument('--recdim', type=int, default=64, help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int, default=3, help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4, help="the weight decay for l2 normalizaton")

    # parser.add_argument('--dropout', type=int, default=0, help="using the dropout or not")
    # parser.add_argument('--keepprob', type=float, default=0.6, help="the batch size for bpr loss training procedure")
    # parser.add_argument('--a_fold', type=int, default=100, help="the fold num used to split large adj matrix, like gowalla")

    parser.add_argument('--testbatch', type=int, default=50, help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str, default='ml-latest-small', help="available datasets: [indonesia_tourism, ml-latest-small, modcloth]")

    # parser.add_argument('--path', type=str, default="checkpoints", help="path to save weights")

    parser.add_argument('--topk', type=int, default="20", help="@K")

    # parser.add_argument('--tensorboard', type=int, default=1, help="enable tensorboard")
    # parser.add_argument('--comment', type=str, default="lgn")
    # parser.add_argument('--load', type=int, default=0)

    parser.add_argument('--epochs', type=int, default=100)

    # parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    # parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')

    parser.add_argument('--seed', type=int, default=2023, help='random seed')

    # parser.add_argument('--model', type=str, default='lmse', help='rec-model, support [mf, lgn, lmse]')

    parser.add_argument('--mse', type=int, default=0, help='Use MSELoss or not')
    parser.add_argument('--sigmoid', type=int, default=1, help='whether we use sigmoid activation')
    parser.add_argument('--clip', type=int, default=0, help='whether we clip output between 0-1')

    return parser.parse_args()


def cprint(msg):
    print(f"\033[0;30;43m{msg}\033[0m")


cprint("[LOAD ARGUMENTS]")

TIMESTAMP = time.strftime("%m-%d-%Hh%Mm%Ss")

__args = parse_args()

DATASET_NAME: str = str(__args.dataset)

REC_DIM: int = int(__args.recdim)
BATCH_SIZE: int = int(__args.batch)
LR: float = float(__args.lr)
TEST_BATCH_SIZE: int = int(__args.testbatch)
LAYER: int = int(__args.layer)
EPOCHS: int = int(__args.epochs)
DECAY: float = float(__args.decay)
TOPK: int = int(__args.topk)
SEED: int = int(__args.seed)

MSE: bool = bool(__args.mse)
SIGMOID: bool = bool(__args.sigmoid)
CLIP: bool = bool(__args.clip)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__root_path = os.path.dirname(__file__)
ROOT_FILE_PATH = os.path.join(__root_path, 'runs')
CURRENT_FILE_PATH = os.path.join(ROOT_FILE_PATH, f'{DATASET_NAME}-{TIMESTAMP}')

if not os.path.exists(ROOT_FILE_PATH):
    os.makedirs(ROOT_FILE_PATH, exist_ok=True)
if not os.path.exists(CURRENT_FILE_PATH):
    os.makedirs(CURRENT_FILE_PATH, exist_ok=True)

print(f"""TIMESTAMP: {TIMESTAMP}
DATASET_NAME: {DATASET_NAME}
REC_DIM: {REC_DIM}
BPR_BATCH_SIZE: {BATCH_SIZE}
LR: {LR}
TEST_BATCH_SIZE: {TEST_BATCH_SIZE}
LAYER: {LAYER}
EPOCHS: {EPOCHS}
DECAY: {DECAY}
TOPK: {TOPK}
SEED: {SEED}
SIGMOID: {SIGMOID}
MSE: {MSE}
CLIP: {CLIP}
DEVICE: {DEVICE}

CURRENT_FILE_PATH: {CURRENT_FILE_PATH}
""")
print()
