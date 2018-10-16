import torch
import numpy as np
import gym
gym.logger.set_level(40)
from components.manipulator import controller_2j, manipulator_2d_get_angles, controller
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import os
import seaborn as sns
from components.utility import to_device
from dm_control import suite
import cv2

# set the backend
plt.switch_backend('agg')


class InverseKinematic:
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

        # check if recurrent
        if self.config["model"]["is_rec"]:
            hidden = self.model.net.init_hidden()
            hidden = to_device(hidden, self.device)

        to_target = 0
        episode_length = self.config["model"]["episode_length"]
        for i in range(episode_length):

            # difference to the target
            if self.config["model"]["diff"]:
                diff = (l1 + l2 + l3 - target)
            else:
                diff = target

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
            d1_ = np.array([output[6], output[7]])
            d2_ = np.array([output[8], output[9]])

            # get alpha, beta and gamma from here
            alpha, beta, gamma = manipulator_2d_get_angles(l1_, l2_, l3_)

            # convert again back to segments
            l1, l2, l3 = controller(l1, l2, l3, alpha, beta, gamma)
            d1 = l1 + l2
            d2 = l2 + l3

            L1, L2, L3 = np.linalg.norm(l1), np.linalg.norm(l2), np.linalg.norm(l3)
            #print(L1, L2, L3)

            # plot
            if eval:
                if i == episode_length - 1:
                    self.plot(l1, l2, l3, final=True)
                else:
                    self.plot(l1, l2, l3)

            # finally compare the position with the target position
            error = (l1 + l2 + l3 - target)**2
            to_target = error

        # average the loss
        #loss = np.sum(to_target)/episode_length
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
            target = np.random.randint(-4, 4, 2)
            params = (l1, l2, l3, target)
            loss += self.run_episode(self.model, params, eval, file_name)
        loss = loss/times
        return loss


class InverseKinematicFull:
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

        # check if recurrent
        if self.config["model"]["is_rec"]:
            hidden = self.model.net.init_hidden()
            hidden = to_device(hidden, self.device)

        # initial pose
        alpha, beta, gamma = manipulator_2d_get_angles(l1, l2, l3)

        to_target = 0
        episode_length = self.config["model"]["episode_length"]
        for i in range(episode_length):

            # difference to the target
            if self.config["model"]["diff"]:
                diff = (l1 + l2 + l3 - target)
            else:
                diff = target

            # do a forward pass
            input = np.array([alpha, beta, gamma, l1[0], l1[1], l2[0], l2[1], l3[0], l3[1], diff[0], diff[1]]).reshape((1,-1))
            input = torch.from_numpy(input).type(torch.float32)

            # send input to device
            input = to_device(input, self.device)

            if self.config["model"]["is_rec"]:
                output, hidden = model.net.forward(input, hidden)
            else:
                output = model.net.forward(input)

            output = output.cpu().data.numpy().reshape((1,-1))[0]

            # get alpha, beta and gamma from here
            alpha, beta, gamma = output[0], output[1], output[2]

            # may have to clip the angles here (TBD)
            alpha = np.clip(alpha, -6.28319, +6.28319)
            beta = np.clip(beta, -6.28319, +6.28319)
            gamma = np.clip(gamma, -6.28319, +6.28319)

            # convert again back to segments
            l1, l2, l3 = controller(l1, l2, l3, alpha, beta, gamma)
            d1 = l1 + l2
            d2 = l2 + l3

            L1, L2, L3 = np.linalg.norm(l1), np.linalg.norm(l2), np.linalg.norm(l3)
            #print(L1, L2, L3)

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
        #loss = np.sum(to_target)

        # save plots
        if eval:
            plt.savefig(filename)

        return loss

    def __call__(self, parameter_value, eval=False, file_name=""):

        # build a model with this parameter
        if parameter_value is not None:
            self.model.set_layer_value(self.parameter_name, parameter_value)

        # then run for 20 episodes and get the average loss
        loss = 0
        times = self.config["SPSA"]["n_evals"]
        for i in range(times):
            # randomly generate a target
            l1 = np.random.randint(-3,3,2)
            l2 = np.random.randint(-3,3,2)
            l3 = np.random.randint(-3,3,2)
            target = np.random.randint(-4, 4, 2)
            params = (l1, l2, l3, target)
            loss += self.run_episode(self.model, params, eval, file_name)
        loss = loss/times
        return loss

