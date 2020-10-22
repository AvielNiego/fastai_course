from pathlib import Path

import torch
from PIL import Image
from fastai.data.core import DataLoaders
from fastai.data.external import untar_data, URLs
from torch.utils.data import DataLoader


def read_data_loaders():
    three_stacked, seven_stacked, threes_stacked_v, seven_stacked_v = read_mnist()

    train_x, train_y, train_dest = create_mnist_dset(three_stacked, seven_stacked)
    valid_x, valid_y, valid_dset = create_mnist_dset(threes_stacked_v, seven_stacked_v)

    dl = DataLoader(train_dest, batch_size=256, shuffle=True)
    valid_dl = DataLoader(valid_dset, batch_size=256, shuffle=True)
    return DataLoaders(dl, valid_dl)


def create_mnist_dset(three_stacked, seven_stacked):
    train_x = torch.cat([three_stacked, seven_stacked]).view(-1, 28 * 28)
    train_y = torch.cat([torch.ones(three_stacked.shape[0]), torch.zeros(seven_stacked.shape[0])]).unsqueeze(1)
    return train_x, train_y, list(zip(train_x, train_y))


def read_mnist():
    path = untar_data(URLs.MNIST_SAMPLE)

    threes_t = load_lazy('/tmp/mnist_sample_stacked3.pt', (path / 'train' / '3').ls().sorted())
    seven_t = load_lazy('/tmp/mnist_sample_stacked7.pt', (path / 'train' / '7').ls().sorted())

    threes_t_v = load_lazy('/tmp/mnist_sample_stacked3_valid.pt', (path / 'valid' / '3').ls().sorted())
    seven_t_v = load_lazy('/tmp/mnist_sample_stacked7_valid.pt', (path / 'valid' / '7').ls().sorted())

    return threes_t, seven_t, threes_t_v, seven_t_v


def load_lazy(pt_file, files):
    if Path(pt_file).exists():
        return torch.load(pt_file)
    stacked = torch.stack([torch.tensor(Image.open(t).getdata()) for t in files]).float() / 255
    torch.save(stacked, pt_file)
    return stacked
