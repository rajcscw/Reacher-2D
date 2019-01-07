import torch


class Optimizer():
    def __init__(self, learning_rate, A=100, alpha=0.101):
        self.learning_rate = learning_rate
        self.A = A
        self.alpha = alpha
        self.t = 1

    def get_learning_rate(self):
        # SPSA type decay
        lr_t = self.learning_rate / (self.t + 1 + self.A)**self.alpha
        return lr_t

    def step_t(self):
        self.t = self.t + 1


class SGD(Optimizer):
    def __init__(self, learning_rate):

        Optimizer.__init__(self, learning_rate=learning_rate)

    def step(self, parameter_name, parameter_value, gradient):
        """
        :param parameter_name: parameter that has to be updated
        :param parameter_value: current value of the parameter
        :param gradient: gradient
        :return: updated parameter
        """

        # gradient descent
        parameter_value = parameter_value - (self.get_learning_rate() * gradient)

        return parameter_value


class RMSProp(Optimizer):
    def __init__(self, learning_rate, A, alpha, decay_rate=0.9):

        Optimizer.__init__(self, learning_rate=learning_rate, A=A, alpha=alpha)

        # set up the gradient cache
        self.gradient_cache = dict()

        # set up other parameters
        self.decay_rate = decay_rate

    def step(self, parameter_name, parameter_value, gradient):
        """
        :param parameter_name: parameter that has to be updated
        :param parameter_value: current value of the parameter
        :param gradient: gradient
        :return: updated parameter
        """

        if parameter_name not in self.gradient_cache.keys():
            self.gradient_cache[parameter_name] = torch.zeros(parameter_value.shape)

        # update the cache
        self.gradient_cache[parameter_name] = self.decay_rate * self.gradient_cache[parameter_name] + \
                                           (1 - self.decay_rate) * (gradient ** 2)

        # calculate the step
        step = (gradient / (torch.sqrt(self.gradient_cache[parameter_name]) + 1e-8))

        # gradient ascent
        parameter_value = parameter_value + (self.get_learning_rate() * step)

        return parameter_value