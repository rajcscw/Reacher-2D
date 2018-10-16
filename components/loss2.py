import torch
import numpy as np
import gym
gym.logger.set_level(40)
from components.manipulator import controller_2j, manipulator_2d_get_angles, controller, manipulator_2d_get_arms
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from components.utility import to_device

# set the backend
plt.switch_backend('agg')


class LinearKinematics:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        # Set up the model
        self.model = model

    def plot(self, l1, l2, l3, initial=False, final=False):
        """
        plot the arms
        :param l1: arm segment 1
        :param l2: arm segment 2
        :param l3: arm segment 3
        :param initial:
        :return:
        """
        x = [0]
        y = [0]
        res = l1
        plt.scatter(l1[0], l1[1], marker=".", c="black")
        x.append(res[0])
        y.append(res[1])
        res = l1 + l2
        plt.scatter(res[0], res[1], marker=".", c="black")
        x.append(res[0])
        y.append(res[1])
        res = l1 + l2 + l3
        plt.scatter(res[0], res[1], marker=".", c="black")
        x.append(res[0])
        y.append(res[1])
        if initial:
            plt.plot(x,y, "--", c="black")
        elif final:
            plt.plot(x,y, "--", c="red")
        else:
            plt.plot(x,y, c="black")

    def run_episode(self, model, params, eval, filename):

        # get the parameters
        l1, l2, l3, target = np.copy(params[0]), np.copy(params[1]), np.copy(params[2]), np.copy(params[3])

        # Initialize figure
        if eval:
            fig = plt.figure(num=1, facecolor="white")
            gs = gridspec.GridSpec(nrows=2, ncols=2)
            ax = plt.subplot(gs[0])
            plt.gca().set_aspect('equal', adjustable='box')
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.fill([0, 0, 5, 10], [0, 5, 5, 0], "b")
            ax.fill([0, 5, 5, 0], [0, 0, -5, -5], "cyan")
            ax.fill([0, 0, -5, -5], [0, -5, -5, 0], "b")
            ax.fill([0, -5, -5, 0], [0, 0, 5, 10], "cyan")
            self.plot(l1, l2, l3, True)
            circle1 = plt.Circle((target[0], target[1]), 0.2, color='g')
            ax.add_artist(circle1)

        # pass the model to the device
        self.model.net = to_device(self.model.net, self.device)

        to_target = 0
        episode_length = self.config["model"]["episode_length"]
        for i in range(episode_length):

            # difference to the target
            if self.config["model"]["diff"]:
                diff = (target - l1 + l2 + l3) # difference between end-effector to target, just like in reacher environment
            else:
                diff = target #target position

            # in this case, l1, l2, l3, r are the inputs
            # do a forward pass
            input = np.array([l1[0], l1[1], l2[0], l2[1], l3[0], l3[1], diff[0], diff[1]]).reshape((1,-1))
            input = torch.from_numpy(input).type(torch.float32)

            # send input to device
            input = to_device(input, self.device)

            if self.config["model"]["is_rec"]:
                output, hidden = model.net.forward(input, hidden)
            else:
                output = model.net.forward(input)

            output = output.cpu().data.numpy().reshape((1,-1))[0]
            l1 = np.array([output[0], output[1]])
            l2 = np.array([output[2], output[3]])
            l3 = np.array([output[4], output[5]])

            # plot
            if eval:
                if i == episode_length - 1:
                    self.plot(l1, l2, l3, final=True)
                else:
                    self.plot(l1, l2, l3)

            # finally compare the position with the target position
            error = (l1 + l2 + l3 - target)**2
            to_target = error

        # loss is based on the final position
        loss = np.sum(to_target)

        # save plots
        if eval:
            plt.savefig(filename)

        return loss

    def __call__(self, parameter_value, eval=False, file_name=""):

        # build a model with this parameter
        if parameter_value is not None:
            self.model.set_layer_value_by_name(self.parameter_name, parameter_value)

        # then run for 20 episodes and get the average loss
        loss = 0
        times = self.config["SPSA"]["n_evals"]
        for i in range(times):
            # randomly generate a target
            l1 = np.random.randint(-3,3,2)
            l2 = np.random.randint(-3,3,2)
            l3 = np.random.randint(-3,3,2)
            target = np.random.randint(-5, 5, 2)
            params = (l1, l2, l3, target)
            loss += self.run_episode(self.model, params, eval, file_name)
        loss = loss/times
        return loss


class NonLinearKinematics:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        # Set up the model
        self.model = model

    def plot(self, l1, l2, l3, initial=False, final=False):
        """
        plot the arms
        :param l1: arm segment 1
        :param l2: arm segment 2
        :param l3: arm segment 3
        :param initial:
        :return:
        """
        x = [0]
        y = [0]
        res = l1
        plt.scatter(l1[0], l1[1], marker=".", c="black")
        x.append(res[0])
        y.append(res[1])
        res = l1 + l2
        plt.scatter(res[0], res[1], marker=".", c="black")
        x.append(res[0])
        y.append(res[1])
        res = l1 + l2 + l3
        plt.scatter(res[0], res[1], marker=".", c="black")
        x.append(res[0])
        y.append(res[1])
        if initial:
            plt.plot(x,y, "--", c="black")
        elif final:
            plt.plot(x,y, "--", c="red")
        else:
            plt.plot(x,y, c="black")

    def run_episode(self, model, params, eval, filename):

        # get the parameters
        l1, l2, l3, target = np.copy(params[0]), np.copy(params[1]), np.copy(params[2]), np.copy(params[3])

        # Initialize figure
        if eval:
            fig = plt.figure(num=1, facecolor="white")
            gs = gridspec.GridSpec(nrows=2, ncols=2)
            ax = plt.subplot(gs[0])
            plt.gca().set_aspect('equal', adjustable='box')
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.fill([0, 0, 5, 10], [0, 5, 5, 0], "b")
            ax.fill([0, 5, 5, 0], [0, 0, -5, -5], "cyan")
            ax.fill([0, 0, -5, -5], [0, -5, -5, 0], "b")
            ax.fill([0, -5, -5, 0], [0, 0, 5, 10], "cyan")
            self.plot(l1, l2, l3, True)
            circle1 = plt.Circle((target[0], target[1]), 0.2, color='g')
            ax.add_artist(circle1)

        # pass the model to the device
        self.model.net = to_device(self.model.net, self.device)

        # L1, L2, L3 - in non-linear model, these original lengths are maintained
        L1 = np.linalg.norm(l1)
        L2 = np.linalg.norm(l2)
        L3 = np.linalg.norm(l3)

        to_target = 0
        episode_length = self.config["model"]["episode_length"]
        for i in range(episode_length):

            # difference to the target
            if self.config["model"]["diff"]:
                diff = (target - l1 + l2 + l3) # difference between end-effector to target, just like in reacher environment
            else:
                diff = target #target position

            # in this case, l1, l2, l3, r are the inputs
            # do a forward pass
            input = np.array([l1[0], l1[1], l2[0], l2[1], l3[0], l3[1], diff[0], diff[1]]).reshape((1,-1))
            input = torch.from_numpy(input).type(torch.float32)

            # send input to device
            input = to_device(input, self.device)

            if self.config["model"]["is_rec"]:
                output, hidden = model.net.forward(input, hidden)
            else:
                output = model.net.forward(input)

            output = output.cpu().data.numpy().reshape((1,-1))[0]
            l1_ = np.array([output[0], output[1]])
            l2_ = np.array([output[2], output[3]])
            l3_ = np.array([output[4], output[5]])

            # find angles
            alpha, beta, gamma = manipulator_2d_get_angles(l1_, l2_, l3_)

            # find corresponding l1, l2, l3 to alpha, beta, gamma (note here we ensure that lengths are maintained)
            l1, l2, l3, _, _ = manipulator_2d_get_arms(alpha, beta, gamma, L1, L2, L3)

            # plot
            if eval:
                if i == episode_length - 1:
                    self.plot(l1, l2, l3, final=True)
                else:
                    self.plot(l1, l2, l3)

            # finally compare the position with the target position
            error = (l1 + l2 + l3 - target)**2
            to_target = error

        # loss based on the final position
        loss = np.sum(to_target)

        # save plots
        if eval:
            plt.savefig(filename)

        return loss

    def __call__(self, parameter_value, eval=False, file_name=""):

        # build a model with this parameter
        if parameter_value is not None:
            self.model.set_layer_value_by_name(self.parameter_name, parameter_value)

        # then run for 20 episodes and get the average loss
        loss = 0
        times = self.config["SPSA"]["n_evals"]
        for i in range(times):
            # randomly generate a target
            l1 = np.random.randint(-3,3,2)
            l2 = np.random.randint(-3,3,2)
            l3 = np.random.randint(-3,3,2)
            target = np.random.randint(-5, 5, 2)
            params = (l1, l2, l3, target)
            loss += self.run_episode(self.model, params, eval, file_name)
        loss = loss/times
        return loss


# just coding, but the idea has to be validated and thought through
class NonLinearFullKinematics:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        # Set up the model
        self.model = model

    def plot(self, l1, l2, l3, initial=False, final=False):
        """
        plot the arms
        :param l1: arm segment 1
        :param l2: arm segment 2
        :param l3: arm segment 3
        :param initial:
        :return:
        """
        x = [0]
        y = [0]
        res = l1
        plt.scatter(l1[0], l1[1], marker=".", c="black")
        x.append(res[0])
        y.append(res[1])
        res = l1 + l2
        plt.scatter(res[0], res[1], marker=".", c="black")
        x.append(res[0])
        y.append(res[1])
        res = l1 + l2 + l3
        plt.scatter(res[0], res[1], marker=".", c="black")
        x.append(res[0])
        y.append(res[1])
        if initial:
            plt.plot(x,y, "--", c="black")
        elif final:
            plt.plot(x,y, "--", c="red")
        else:
            plt.plot(x,y, c="black")

    def run_episode(self, model, params, eval, filename):

        # get the parameters
        l1, l2, l3, target = np.copy(params[0]), np.copy(params[1]), np.copy(params[2]), np.copy(params[3])
        d1 = l1 + l2
        d2 = l2 + l3

        # Initialize figure
        if eval:
            fig = plt.figure(num=1, facecolor="white")
            gs = gridspec.GridSpec(nrows=2, ncols=2)
            ax = plt.subplot(gs[0])
            plt.gca().set_aspect('equal', adjustable='box')
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.fill([0, 0, 5, 10], [0, 5, 5, 0], "b")
            ax.fill([0, 5, 5, 0], [0, 0, -5, -5], "cyan")
            ax.fill([0, 0, -5, -5], [0, -5, -5, 0], "b")
            ax.fill([0, -5, -5, 0], [0, 0, 5, 10], "cyan")
            self.plot(l1, l2, l3, True)
            circle1 = plt.Circle((target[0], target[1]), 0.2, color='g')
            ax.add_artist(circle1)

        # pass the model to the device
        self.model.net = to_device(self.model.net, self.device)

        # L1, L2, L3 - in non-linear model, these original lengths are maintained
        L1 = np.linalg.norm(l1)
        L2 = np.linalg.norm(l2)
        L3 = np.linalg.norm(l3)

        to_target = 0
        episode_length = self.config["model"]["episode_length"]
        for i in range(episode_length):

            # difference to the target
            if self.config["model"]["diff"]:
                diff = (l1 + l2 + l3 - target)
            else:
                diff = target

            # in this case, l1, l2, l3, d1, d2, r are the inputs
            # do a forward pass
            input = np.array([l1[0], l1[1], l2[0], l2[1], l3[0], l3[1], d1[0], d1[1], d2[0], d2[1], diff[0], diff[1]]).reshape((1,-1))
            input = torch.from_numpy(input).type(torch.float32)

            # send input to device
            input = to_device(input, self.device)

            if self.config["model"]["is_rec"]:
                output, hidden = model.net.forward(input, hidden)
            else:
                output = model.net.forward(input)

            output = output.cpu().data.numpy().reshape((1,-1))[0]
            l1_ = np.array([output[0], output[1]])
            l2_ = np.array([output[2], output[3]])
            l3_ = np.array([output[4], output[5]])
            d1 = l1_ + l2_
            d2 = l2_ + l3_

            # find angles
            alpha, beta, gamma = manipulator_2d_get_angles(l1_, l2_, l3_)

            # find corresponding l1, l2, l3 to alpha, beta, gamma (note here we ensure that lengths are maintained)
            l1, l2, l3, d1, d2 = manipulator_2d_get_arms(alpha, beta, gamma, L1, L2, L3)

            # plot
            if eval:
                if i == episode_length - 1:
                    self.plot(l1, l2, l3, final=True)
                else:
                    self.plot(l1, l2, l3)

            # finally compare the position with the target position
            error = (l1 + l2 + l3 - target)**2
            to_target += error

        # average the loss
        loss = np.sum(to_target)/episode_length

        # save plots
        if eval:
            plt.savefig(filename)

        return loss

    def __call__(self, parameter_value, eval=False, file_name=""):

        # build a model with this parameter
        if parameter_value is not None:
            self.model.set_layer_value_by_name(self.parameter_name, parameter_value)

        # then run for 20 episodes and get the average loss
        loss = 0
        times = self.config["SPSA"]["n_evals"] if not eval else 1
        for i in range(times):
            # randomly generate a target
            l1 = np.random.randint(-3,3,2)
            l2 = np.random.randint(-3,3,2)
            l3 = np.random.randint(-3,3,2)
            target = np.random.randint(-5, 5, 2)
            params = (l1, l2, l3, target)
            loss += self.run_episode(self.model, params, eval, file_name)
        loss = loss/times
        return loss