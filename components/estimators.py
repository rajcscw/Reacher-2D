import numpy as np
from torch.multiprocessing import Pool
import torch
from functools import reduce


class PerturbRunner:
    def __init__(self, loss_function):
        self.loss_function = loss_function

        # temp variables (to avoid passing to a function because of map())
        self.c_t = None
        self.current_estimate = None
        self.current_parameter = None

    def run_for_perturb(self, i):
        # get the random perturbation vector from bernoulli distribution
        # it has to be symmetric around zero
        # But normal distribution does not work which makes the perturbations close to zero
        # Also, uniform distribution should not be used since they are not around zero
        delta = torch.randint(0,2, self.current_estimate.shape) * 2 - 1
        scale = self.c_t

        # param_plus and minus
        param_plus = self.current_estimate + delta * scale
        param_minus = self.current_estimate - delta * scale

        # measure the loss function at perturbations
        self.loss_function.parameter_name = self.current_parameter
        loss_plus, _ = self.loss_function(param_plus)
        loss_minus, _ = self.loss_function(param_minus)

        return (loss_plus, loss_minus), delta * scale


class SPSA:
    """
    An optimizer class that implements Simultaneous Perturbation Stochastic Approximation (SPSA)
    """
    def __init__(self, a, c, A, alpha, gamma, k, param_decay, loss_function, model, device, obj_scaler, use_parallel_gpu=True, parallel_workers=8):
        # Initialize gain parameters and decay factors
        self.a = a
        self.c = c
        self.A = A
        self.k = k
        self.alpha = alpha
        self.gamma = gamma
        self.param_decay = param_decay
        self.loss_function = loss_function
        self.use_parallel_gpu = use_parallel_gpu

        # model
        self.model = model

        # device
        self.device = device

        # counters
        self.t = 0

        # generate a pool of processes
        self.p = Pool(processes=parallel_workers)

        # perturb runner
        self.runner = PerturbRunner(loss_function)

        # objective scale
        self.obj_scaler = obj_scaler

    def compute_gradient(self, obj_plus, obj_minus, perturb):

        # total objective - weighted sum of objective and novelty
        total_obj_plus = obj_plus
        total_obj_minus = obj_minus

        # compute the estimate of the gradient
        if total_obj_plus == total_obj_minus:
            return None
        else:
            g_t = (total_obj_plus - total_obj_minus) / (2.0 * perturb)
            g_t = g_t.type(torch.float32)
            return g_t

    def gradients_from_objectives(self, current_estimate, obj_values):
        non_zero_count = 0
        total_gradients = torch.zeros(current_estimate.shape)
        for obj in obj_values:

            # separate the bahavior values and objective values
            obj_plus, obj_minus = obj[0]
            scale = obj[1]

            # scale the objectives
            if self.obj_scaler is not None:
                obj_plus, obj_minus = self.obj_scaler.fit_transform(obj_plus), self.obj_scaler.fit_transform(obj_minus)

            # compute gradient
            gradient = self.compute_gradient(obj_plus, obj_minus, scale)

            if gradient is not None:
                total_gradients += gradient
                non_zero_count += 1

        # average the gradients
        if non_zero_count > 0:
            g_t = total_gradients / non_zero_count
        else:
            g_t = total_gradients

        return g_t

    def run_parellely(self, current_estimate, current_parameter, c_t):
        # run parallely for all k mirrored candidates
        self.runner.c_t = c_t
        self.runner.current_estimate = current_estimate
        self.runner.current_parameter = current_parameter
        obj_values = self.p.map(self.runner.run_for_perturb, range(self.k))
        gradient = self.gradients_from_objectives(current_estimate, obj_values)
        return gradient

    def run_sequentially(self, current_estimate, current_parameter, c_t):
        obj_values = []
        for i in range(self.k):
            self.runner.c_t = c_t
            self.runner.current_estimate = current_estimate
            self.runner.current_parameter = current_parameter
            obj = self.runner.run_for_perturb(i)
            obj_values.append(obj)
        gradient = self.gradients_from_objectives(current_estimate, obj_values)
        return gradient

    def estimate_gradient(self, current_parameter, current_estimate):
        """
        :param current_estimate: This is the current estimate of the parameter vector
        :return: returns the estimate of the gradient at that point
        """

        # get the current values for gain sequences
        a_t = self.a / (self.t + 1 + self.A)**self.alpha
        c_t = self.c / (self.t + 1)**self.gamma

        # run parallely or sequentially based on the device
        if self.device == "cpu":
            g_t = self.run_parellely(current_estimate, current_parameter, c_t)
        else: # in gpu mode
            if self.use_parallel_gpu:
                g_t = self.run_parellely(current_estimate, current_parameter, c_t)
            else:
                g_t = self.run_sequentially(current_estimate, current_parameter, c_t)

        return g_t

    def step_t(self):
        self.t = self.t + 1

    def reset_t(self):
        self.t = 0

    def get_perturb_scale(self):
        return self.c / (self.t + 1)**self.gamma





