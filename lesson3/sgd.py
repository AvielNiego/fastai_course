# from fastai.vision.all import *
from pathlib import Path

import torch
from PIL import Image
from fastai.data.external import untar_data, URLs
from torch import tensor
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader


def predict_roller_coaster_speed():
    time = torch.arange(0, 20).float()
    speed = torch.randn(20) * 3 + 0.75 * (time - 9.5) ** 2 + 1

    lr = 1e-5

    params = torch.randn(3).requires_grad_()

    for i in range(5):
        preds = f(time, params)
        loss = mse(preds, speed)
        show_res(time, preds, speed)
        print(loss.data)
        loss.backward()
        params.data -= params.grad.data * lr
        params.grad = None

    preds = f(time, params)
    show_res(time, preds, speed)


def show_res(time, preds, target):
    plt.scatter(time, target)
    plt.scatter(time, preds.data, color='red')
    plt.show()


def f(t, params):
    a, b, c = params
    return a * (t ** 2) + (b * t) + c


def mse(preds, targets):
    return ((preds - targets) ** 2).mean()


if __name__ == '__main__':
    predict_roller_coaster_speed()
