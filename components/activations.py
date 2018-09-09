import numpy as np
from scipy.special import expit


class HyperbolicTangent(object):
    def __call__(self, x):
        try:
            return np.tanh(x)
        except BaseException as e:
            print("Exception: "+str(e))


class HyperbolicTangent2(object):
    def __call__(self, x):
        exp = np.exp(-2 * x)
        return (1-exp)/(1+exp)


class Softmin(object):
    def __call__(self, x):
        result = np.zeros(x.shape)
        argmin = np.argmin(x)
        result[argmin] = x[argmin]
        return result


class LogisticFunction(object):
    def __init__(self, beta=1.0):
        self.beta = beta

    def __call__(self, x):
        return expit(self.beta * x)


class ReLU(object):
    def __call__(self, x):
        return np.maximum(x, np.zeros(x.shape))


class Linear(object):
    def __call__(self, x):
        return x


class SoftMax(object):
    def __call__(self, x):
        exp = np.exp(x - np.max(x))
        sum = np.sum(exp)
        softmax = exp / sum
        return softmax