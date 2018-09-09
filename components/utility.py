import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# set the backend
plt.switch_backend('agg')


def rolling_mean(X, window_size, pad=None):
    # pad in the front
    if pad is None:
        front = np.full((window_size,), X[0]).tolist()
    else:
        front = np.full((window_size,), pad).tolist()
    padded = front + X
    mean = np.convolve(padded, np.ones((window_size,))/window_size, "valid")
    return mean


def plot_learning_curve(file, title, series, value="Loss"):
    fig = plt.figure()
    sns.set(style="darkgrid")
    sns.set_context("paper")
    plt.title(title, fontsize=10)
    sns.tsplot(data=series, time="Iteration", unit="run", condition="strategy", value=value)
    plt.legend(loc="upper right", fontsize=10)
    plt.ylabel(value, fontsize=10)
    plt.xlabel("Iteration", fontsize=10)
    plt.savefig(file)


def print_accuracy(model, data_loader, device):
    # overall accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data

            # send it to appropriate device
            images, labels = to_device(images, device), to_device(labels, device)

            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (100 * correct / total)

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

    return accuracy


def to_device(tensor, device):
    if device == "cpu":
        return tensor
    else:
        return tensor.cuda(device)


def init_multiproc():
    from torch.multiprocessing import set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass


class MinMaxCustomScaler:
    def __init__(self, min=0, max=1):
        self.sample_min = +np.inf
        self.sample_max = -np.inf

        self.to_min, self.to_max = min, max

    def fit_transform(self, value):

        # update the sample min and max if necessary
        self.sample_min = min(self.sample_min, value)
        self.sample_max = max(self.sample_max, value)

        # scale the value
        if self.sample_max == self.sample_min:
            value_std = 0.0
        else:
            value_std = (value - self.sample_min) / (self.sample_max - self.sample_min)

        # transform to range
        value_scaled = value_std * (self.to_max - self.to_min) + self.to_min
        return value_scaled


class MinMaxCustomScalerArray:
    def __init__(self, dim, min=0, max=1):
        self.dim = dim
        self.sample_min = +np.inf * np.ones(dim)
        self.sample_max = -np.inf * np.ones(dim)

        self.to_min, self.to_max = min * np.ones(dim), max * np.ones(dim)

    def fit_transform(self, value):

        # update the sample min and max if necessary
        self.sample_min = np.minimum(self.sample_min, value)
        self.sample_max = np.maximum(self.sample_max, value)

        # scale the value
        if (self.sample_max == self.sample_min).all():
            value_std = np.zeros(value.shape[0])
        else:
            value_std = (value - self.sample_min) / (self.sample_max - self.sample_min)
            value_std = np.nan_to_num(value_std)

        # transform to range
        value_scaled = value_std * (self.to_max - self.to_min) + self.to_min
        return value_scaled







