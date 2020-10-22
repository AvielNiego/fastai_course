from fastai.optimizer import SGD
import torch
from fastai.learner import Learner

from lesson3.mnist_loader import read_data_loaders


def predict_mnist():
    dls = read_data_loaders()
    lr = 1.

    learn = Learner(dls=dls,
                    model=torch.nn.Linear(28 * 28, 1),
                    opt_func=SGD,
                    loss_func=mnist_loss,
                    metrics=batch_accuracy)
    learn.fit(10, lr=lr)


def batch_accuracy(preds, y):
    preds = preds.sigmoid()
    correct = (preds > 0.5) == y
    return correct.float().mean()


def mnist_loss(preds: torch.Tensor, truth):
    preds = preds.sigmoid()
    return torch.where(truth == 1, 1 - preds, preds).mean()


if __name__ == '__main__':
    predict_mnist()
