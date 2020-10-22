import torch
from torch import tensor

from lesson3.mnist_loader import read_data_loaders


def predict_mnist():
    dls = read_data_loaders()

    w1 = init_params((28 * 28, 30))
    bias1 = init_params(30)
    w2 = init_params((30, 1))
    bias2 = init_params(1)
    params = w1, bias1, w2, bias2
    lr = 1

    for i in range(40):
        train_epoch(simple_net, dls.train, params, lr)
        accuracy = validate_epoch(simple_net, params, dls.valid)
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


def simple_net(xb, params) -> tensor:
    w1, b1, w2, b2 = params
    res = xb @ w1 + b1
    res = res.max(tensor(0.0))
    res = res @ w2 + b2
    return res


def init_params(size, var=1.0):
    return (torch.randn(size) * var).requires_grad_()


if __name__ == '__main__':
    predict_mnist()
