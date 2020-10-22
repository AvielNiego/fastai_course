import torch
from torch import tensor

from lesson3.mnist_loader import read_data_loaders


def predict_mnist():
    dls = read_data_loaders()

    weights = init_params((28 * 28, 1))
    bias = init_params(1)
    params = weights, bias
    lr = 1.

    for i in range(40):
        train_epoch(linear1, dls.train, params, lr)
        accuracy = validate_epoch(linear1, params, dls.valid)
        print(i, 'accuracy', accuracy)


def validate_epoch(model, params, valid_dl):
    accs = [batch_accuracy(model(x, params), y) for x, y in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)


def batch_accuracy(preds, y):
    preds = preds.sigmoid()
    correct = (preds > 0.5) == y
    return correct.float().mean()


def train_epoch(model, dl, params, lr):
    for x, y in dl:
        calc_grad(model, params, x, y)

        for p in params:
            p.data -= p.grad * lr
            p.grad.zero_()


def calc_grad(model, params, x, y):
    preds = model(x, params)
    loss = mnist_loss(preds, y)
    loss.backward()


def mnist_loss(preds: torch.Tensor, truth):
    preds = preds.sigmoid()
    return torch.where(truth == 1, 1 - preds, preds).mean()


def linear1(xb, params) -> tensor:
    weights, bias = params
    return xb @ weights + bias


def init_params(size, var=1.0):
    return (torch.randn(size) * var).requires_grad_()


if __name__ == '__main__':
    predict_mnist()
