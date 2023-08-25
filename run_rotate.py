# -*- encoding: utf-8 -*-
"""
@ Author  ：
@ File    : run_single.py
@ Time    ：
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import os
from config import config
from utils.standardrange import *

from model import EncoderForecaster
from utils.earlyStopping import EarlyStopping
from utils.loss import WeightedMseMae
from utils.dataset import MultiDataset, SingleDataset
from utils.rotate import rotate_sequence
from utils.assesment import *


LR, STEP_SIZE, GAMMA = config.pre_parameters


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    # cudnn.benchmark = False
    # cudnn.enabled = False

setup_seed(2019)

def _init_fn(worker_id):
    random.seed(10 + worker_id)
    np.random.seed(10 + worker_id)
    torch.manual_seed(10 + worker_id)
    torch.cuda.manual_seed(10 + worker_id)
    torch.cuda.manual_seed_all(10 + worker_id)


import logging


logging.basicConfig(filename='logger.log', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)  #
logger.setLevel(level=logging.INFO)  #

pathlog = "log_train.txt"
filehandler = logging.FileHandler(pathlog)  
filehandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
filehandler.setFormatter(formatter)

console = logging.StreamHandler()  
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(filehandler)
logger.addHandler(console)

logger.info("Start log")

logger.info("Parametes:　LR: {}, STEP_SIZE:{}, GAMMA: {}".format(LR, STEP_SIZE, GAMMA))


def main(path="/second", case=1, n_train=1, n_test=1, dt=1, st=0, ste=0, rotate=0):
    """
    """
    logger.info("Run in file: {}".format(path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    case_str = "_n_"

    # create dataloader
    print("Laoding dataset to torch.")
    path_fa = path + "_half_"  
    trainDataset = SingleDataset(path_fa, n_train, dt, case_str, "train", st) 
    trainDataloader = DataLoader(
        dataset=trainDataset,
        batch_size=config.batch,
        shuffle=True,
        num_workers=config.num_workers,
        worker_init_fn=_init_fn
    )
    testDataset = SingleDataset(path_fa, n_test, dt, case_str, "test", ste)
    testDataloader = DataLoader(
        dataset=testDataset,
        batch_size=config.batch,
        shuffle=False,
        num_workers=config.num_workers
    )
    print("Dataset prepared.")

    torch.cuda.empty_cache() 

    model = EncoderForecaster(config).to(device)

    loss_fn = WeightedMseMae()
    loss_fn.to(device)

    loss_fn_p = precision()
    loss_fn_r = recall()

    optimiser = optim.Adam(params=model.parameters(), lr=LR)  

    # prepare for loss_fn
    sample = np.load(config.sample)["sequence_array"]

    print("Start Train.")
    num_epochs = config.num_epochs
    total_step = len(trainDataloader)
    loss_List = []
    loss_test_list = []
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=STEP_SIZE, gamma=GAMMA)  # StepLR

    # save_path for model in different case
    path_m = path + "_n_conv2d.pth"

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=config.patience, verbose=False, path=path_m)

    if os.path.exists(path_m):
        print(path_m)
        model.load_state_dict(torch.load(path_m)["state_dict"])

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        totalLoss = 0  
        for i, (images, target) in enumerate(trainDataloader):
            sam = sample.copy()
            if rotate:
                # print("start rotate!")
                sz = images.shape
                images = images.reshape((-1, sz[3], sz[4]))
                target = target.reshape((-1, sz[3], sz[4]))
                x = np.append(images, target, axis=0)
                x = np.append(x, sam, axis=0)
                x = rotate_sequence(x)

                images, target, sam = x[:sz[0]*sz[1]], x[sz[0]*sz[1]: -1], x[-1]
                images = images.reshape(*sz)
                target = target.reshape((-1, 1, sz[-2], sz[-1]))
                sam = sam.reshape(-1, sz[-2], sz[-1])

                images = torch.from_numpy(images)
                target = torch.from_numpy(target)
                # print("rotation is finished")
            images = images.to(device)
            target = target.to(device)
  
            pred = model(images)
            loss = loss_fn(pred, target, sam)

            optimiser.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=50.0) 
            optimiser.step()

            lr_now = optimiser.state_dict()['param_groups'][0]['lr']

            lr_scheduler.step()

            totalLoss = totalLoss + loss.item()

            if i % 30 == 0:
                tem = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print("Time {}, Epoch [{}/{}], Step [{}/{}], loss: {:.8f}, lr: {}".format(tem, epoch+1, num_epochs, i+1, total_step, totalLoss/(i+1), lr_now))
        loss_List.append(totalLoss/(i+1))
        logger.info("Time {}, Epoch [{}/{}], Step [{}/{}], loss: {:.8f}, lr: {}".format(tem, epoch+1, num_epochs, i+1, total_step, totalLoss/(i+1), lr_now))

        model.eval()
        with torch.no_grad():
            loss_t = 0
            for j, (images, target) in enumerate(testDataloader):
                images, target = images.to(device), target.to(device)
                preds = model(images)
                loss_test = loss_fn(preds, target)
                loss_t += loss_test.item()
                loss_prec = loss_fn_p(preds, target, threshold=100).item()
                loss_reca = loss_fn_r(preds, target, threshold=100).item()
        logger.info("Loss in Test dataset: {}, precision: {}, recall: {}".format(loss_t / (j + 1), loss_prec, loss_reca))

        checkpoint = {
            "state_dict": model.state_dict(),
            "opt_state_dict": optimiser.state_dict(),
            "epoch": epoch
        }
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping.checkpoint = checkpoint
        early_stopping(loss_t/(j+1), model)

        loss_test_list.append(loss_t/(j+1))
        print("_"*10)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    df = pd.DataFrame(data=np.array(loss_List), columns=["loss_train"])
    df["loss_test"] = np.array(loss_test_list)

    if case == 0:
        pathdf = path + "_n_conv2d_loss.xlsx"
    elif case == 1:
        pathdf = path + "_r_conv2d_loss.xlsx"
    elif case == 2:
        pathdf = path + "_r3_conv2d_loss.xlsx"
    else:
        pathdf = path + "_r4_conv2d_loss.xlsx"

    df.to_excel(pathdf)

    plt.figure(figsize=(8, 8))
    plt.plot(loss_List, color='red', linewidth=1.5, linestyle='-', label="loss_train")
    plt.plot(loss_test_list, color='black', linewidth=1.5, linestyle='-', label="loss_test")
    plt.legend(loc="upper right")
    if case == 0:
        pathpic = path + "_n_conv2d_loss.jpg"
    elif case == 1:
        pathpic = path + "_r_conv2d_loss.jpg"
    elif case == 2:
        pathpic = path + "_r3_conv2d_loss.jpg"
    else:
        pathpic = path + "_r4_conv2d_loss.jpg"
    plt.savefig(pathpic, dpi=100)
    plt.close()


if __name__ == "__main__":
    filename = "p915"
    path = "./data/" + filename + "/second"
    torch.cuda.empty_cache()
    case = 0 
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    main(path, case, n_train=1, n_test=1, dt=1, st=7878, ste=805, rotate=True)  # a example
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
  
