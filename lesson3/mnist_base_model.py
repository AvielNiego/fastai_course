from fastai.vision.all import *

path = untar_data(URLs.MNIST_SAMPLE)


def main():
    threes = (path/'train'/'3').ls().sorted()
    sevens = (path/'train'/'7').ls().sorted()

    threes_t = torch.stack([tensor(Image.open(t)) for t in threes]).float() / 255
    seven_t = torch.stack([tensor(Image.open(t)) for t in sevens]).float() / 255

    three_mean = threes_t.mean(dim=0)
    seven_mean = seven_t.mean(dim=0)

    # plt.imshow(three_mean, cmap='gray')
    # plt.show()

    valid_threes = (path / 'valid' / '3').ls().sorted()
    valid_sevens = (path / 'valid' / '7').ls().sorted()

    three_right_pred = [1 for t in valid_threes if is_three(tensor(Image.open(t)).float() / 255, three_mean, seven_mean)]
    seven_right_pred = [1 for t in valid_sevens if not is_three(tensor(Image.open(t)).float() / 255, three_mean, seven_mean)]
    hi = 4


def is_three(x, three_mean, seven_mean):
    return (three_mean - x).abs().mean() < (seven_mean - x).abs().mean()



if __name__ == '__main__':
    main()
