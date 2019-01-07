import torch
import numpy as np
import gym
gym.logger.set_level(40)
from components.manipulator import manipulator_2d_get_angles, controller, manipulator_2d_get_arms
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from components.utility import to_device
import math

# set the backend
plt.switch_backend('agg')

# colors
JOINTS = "#f90000ff"
ARMS = "#ffd39fff"
ARMS_INIT = "#FFA500FF"
ARMS_FINAL = "#B22222ff"
ENDS = "#36f900ff"
GRID_1 = "#A8A8A8ff"
GRID_2 = "#D3D3D3ff"
TARGET = "g"


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
            plt.savefig(filename, bbox_inches="tight")

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

    def plot(self, ax, l1, l2, l3, initial=False, final=False):
        """
        plot the arms
        :param l1: arm segment 1
        :param l2: arm segment 2
        :param l3: arm segment 3
        :param initial:
        :return:
        """
        ax.add_artist(plt.Circle((0, 0), 0.05, color=JOINTS))
        x = [0]
        y = [0]
        res = l1
        ax.add_artist(plt.Circle((l1[0], l1[1]), 0.05, color=JOINTS))
        x.append(res[0])
        y.append(res[1])
        res = l1 + l2
        ax.add_artist(plt.Circle((res[0], res[1]), 0.05, color=JOINTS))
        x.append(res[0])
        y.append(res[1])
        res = l1 + l2 + l3
        ax.add_artist(plt.Circle((res[0], res[1]), 0.05, color=JOINTS))
        x.append(res[0])
        y.append(res[1])
        if initial:
            plt.plot(x,y, c=ARMS_INIT, linewidth=4.0)
        elif final:
            plt.plot(x,y, c=ARMS_FINAL, linewidth=4.0)
        else:
            plt.plot(x,y, c=ARMS, linewidth=5.0)

    def run_episode(self, model, params, eval, filename):

        # get the parameters
        l1, l2, l3, target = np.copy(params[0]), np.copy(params[1]), np.copy(params[2]), np.copy(params[3])

        # Initialize figure
        if eval:
            fig = plt.figure(num=1, facecolor="white")
            gs = gridspec.GridSpec(nrows=1, ncols=1)
            ax = plt.subplot(gs[0])
            plt.gca().set_aspect('equal', adjustable='box')
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.fill([0, 0, 3, 6], [0, 3, 3, 0], GRID_1)
            ax.fill([0, 3, 3, 0], [0, 0, -3, -3], GRID_2)
            ax.fill([0, 0, -3, -3], [0, -3, -3, 0], GRID_1)
            ax.fill([0, -3, -3, 0], [0, 0, 3, 6], GRID_2)
            self.plot(ax, l1, l2, l3, True)
            circle1 = plt.Circle((target[0], target[1]), 0.2, color=TARGET)
            ax.add_artist(circle1)

        # pass the model to the device
        self.model.net = to_device(self.model.net, self.device)

        # L1, L2, L3 - in non-linear model, these original lengths are maintained
        L1 = np.linalg.norm(l1)
        L2 = np.linalg.norm(l2)
        L3 = np.linalg.norm(l3)

        alpha_iter = []
        beta_iter = []
        gamma_iter = []
        reward_at_each_step = []

        to_target = 0
        episode_length = self.config["model"]["episode_length"]
        for i in range(episode_length):

            # difference to the target
            if self.config["model"]["diff"]:
                diff = (target - l1 + l2 + l3) # difference between end-effector to target, just like in reacher environment
            else:
                diff = target # target position

            # in this case, l1, l2, l3, r are the inputs
            # do a forward pass
            input = np.array([l1[0], l1[1], l2[0], l2[1], l3[0], l3[1], diff[0], diff[1]]).reshape((1,-1))
            input = torch.from_numpy(input).type(torch.float32)

            # send input to device
            input = to_device(input, self.device)

            output = model.net.forward(input)

            output = output.cpu().data.numpy().reshape((1,-1))[0]
            l1_ = np.array([output[0], output[1]])
            l2_ = np.array([output[2], output[3]])
            l3_ = np.array([output[4], output[5]])

            # find angles
            alpha, beta, gamma = manipulator_2d_get_angles(l1_, l2_, l3_)
            alpha_iter.append(np.rad2deg(alpha))
            beta_iter.append(np.rad2deg(beta))
            gamma_iter.append(np.rad2deg(gamma))

            # find corresponding l1, l2, l3 to alpha, beta, gamma (note here we ensure that lengths are maintained)
            l1, l2, l3, _, _ = manipulator_2d_get_arms(alpha, beta, gamma, L1, L2, L3)

            # plot
            if eval:
                if i == episode_length - 1:
                    self.plot(ax, l1, l2, l3, final=True)
                else:
                    self.plot(ax, l1, l2, l3)

            # finally compare the position with the target position
            error = -np.linalg.norm(l1 + l2 + l3 - target)
            reward_at_each_step.append(error)
            to_target += error

        # loss based on the final position
        loss = to_target

        # save plots
        if eval:
            plt.savefig(filename, bbox_inches="tight")

        return loss, (alpha_iter, beta_iter, gamma_iter, reward_at_each_step)

    def __call__(self, parameter_value, eval=False, file_name=""):

        # build a model with this parameter
        if parameter_value is not None:
            self.model.set_layer_value(self.parameter_name, parameter_value)

        # then run for 20 episodes and get the average loss
        loss = 0
        times = self.config["SPSA"]["n_evals"]
        for i in range(times):
            arm_lengths_1 = np.linspace(0.2, 1.0).tolist()
            arm_lengths_2 = np.linspace(-1.0, -0.2).tolist()
            arm_lengths = arm_lengths_1 + arm_lengths_2

            l1 = np.random.choice(a=arm_lengths, size=2)
            l1 = l1 / np.linalg.norm(l1)
            l2 = np.random.choice(a=arm_lengths, size=2)
            l2 = l2 / np.linalg.norm(l2)
            l3 = np.random.choice(a=arm_lengths, size=2)
            l3 = l3 / np.linalg.norm(l3)
            target = self.get_random_in_circle(self.config["env"]["radius"])
            params = (l1, l2, l3, target)
            loss_run, angles = self.run_episode(self.model, params, eval, file_name)
            loss += loss_run
        loss = loss/times
        return loss, angles

    def get_random_in_circle(self, radius):

            t = 2 * math.pi * np.random.rand()
            u = np.random.rand() + np.random.rand()

            if u > 1:
                r = 2 - u
            else:
                r = u

            result = radius * r * math.cos(t), radius * r * math.sin(t)
            return np.array(result)


class NonLinearKinematicsByUnrolling:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        # Set up the model
        self.model = model

    def plot(self, ax, l1, l2, l3, initial=False, final=False):
        """
        plot the arms
        :param l1: arm segment 1
        :param l2: arm segment 2
        :param l3: arm segment 3
        :param initial:
        :return:
        """
        ax.add_artist(plt.Circle((0, 0), 0.05, color=JOINTS))
        x = [0]
        y = [0]
        res = l1
        ax.add_artist(plt.Circle((l1[0], l1[1]), 0.05, color=JOINTS))
        x.append(res[0])
        y.append(res[1])
        res = l1 + l2
        ax.add_artist(plt.Circle((res[0], res[1]), 0.05, color=JOINTS))
        x.append(res[0])
        y.append(res[1])
        res = l1 + l2 + l3
        ax.add_artist(plt.Circle((res[0], res[1]), 0.05, color=JOINTS))
        x.append(res[0])
        y.append(res[1])
        if initial:
            plt.plot(x,y, c=ARMS_INIT, linewidth=4.0)
        elif final:
            plt.plot(x,y, c=ARMS_FINAL, linewidth=4.0)
        else:
            plt.plot(x,y, c=ARMS, linewidth=5.0)

    def get_angles(self, model, l1, l2, l3, diff, L1, L2, L3, tau=1):

        l1_, l2_, l3_ = np.copy(l1), np.copy(l2), np.copy(l3)

        # unroll over a period of time and then get the output
        # kind of like imagination where you want to be
        for i in range(tau):
            # in this case, l1, l2, l3, r are the inputs
            # do a forward pass
            input = np.array([l1_[0], l1_[1], l2_[0], l2_[1], l3_[0], l3_[1], diff[0], diff[1]]).reshape((1,-1))
            input = torch.from_numpy(input).type(torch.float32)

            # send input to device
            input = to_device(input, self.device)

            output = model.net.forward(input)

            output = output.cpu().data.numpy().reshape((1,-1))[0]
            l1_ = np.array([output[0], output[1]])
            l2_ = np.array([output[2], output[3]])
            l3_ = np.array([output[4], output[5]])

            # # get the angle
            # alpha, beta, gamma = manipulator_2d_get_angles(l1_, l2_, l3_)
            #
            # l1_, l2_, l3_, _, _ = manipulator_2d_get_arms(alpha, beta, gamma, L1, L2, L3)
        alpha, beta, gamma = manipulator_2d_get_angles(l1_, l2_, l3_)

        return alpha, beta, gamma

    def run_episode(self, model, params, eval, filename):

        # get the parameters
        l1, l2, l3, target = np.copy(params[0]), np.copy(params[1]), np.copy(params[2]), np.copy(params[3])

        # Initialize figure
        if eval:
            fig = plt.figure(num=1, facecolor="white")
            gs = gridspec.GridSpec(nrows=1, ncols=1)
            ax = plt.subplot(gs[0])
            plt.gca().set_aspect('equal', adjustable='box')
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.fill([0, 0, 3, 6], [0, 3, 3, 0], GRID_1)
            ax.fill([0, 3, 3, 0], [0, 0, -3, -3], GRID_2)
            ax.fill([0, 0, -3, -3], [0, -3, -3, 0], GRID_1)
            ax.fill([0, -3, -3, 0], [0, 0, 3, 6], GRID_2)
            self.plot(ax, l1, l2, l3, True)
            circle1 = plt.Circle((target[0], target[1]), 0.2, color=TARGET)
            ax.add_artist(circle1)

        # pass the model to the device
        self.model.net = to_device(self.model.net, self.device)

        # L1, L2, L3 - in non-linear model, these original lengths are maintained
        L1 = np.linalg.norm(l1)
        L2 = np.linalg.norm(l2)
        L3 = np.linalg.norm(l3)

        alpha_iter = []
        beta_iter = []
        gamma_iter = []

        to_target = 0
        episode_length = self.config["model"]["episode_length"]
        for i in range(episode_length):

            # difference to the target
            if self.config["model"]["diff"]:
                diff = (target - l1 + l2 + l3) # difference between end-effector to target, just like in reacher environment
            else:
                diff = target # target position

            # mmc linear net
            alpha, beta, gamma = self.get_angles(model, l1, l2, l3, diff, L1, L2, L3, 2)

            # find angles
            alpha_iter.append(np.rad2deg(alpha))
            beta_iter.append(np.rad2deg(beta))
            gamma_iter.append(np.rad2deg(gamma))

            # find corresponding l1, l2, l3 to alpha, beta, gamma (note here we ensure that lengths are maintained)
            l1, l2, l3, _, _ = manipulator_2d_get_arms(alpha, beta, gamma, L1, L2, L3)

            # plot
            if eval:
                if i == episode_length - 1:
                    self.plot(ax, l1, l2, l3, final=True)
                else:
                    self.plot(ax, l1, l2, l3)

            # finally compare the position with the target position
            error = -np.linalg.norm(l1 + l2 + l3 - target)
            to_target += error

        # loss based on the final position
        loss = to_target

        # save plots
        if eval:
            plt.savefig(filename, bbox_inches="tight")

        return loss, (alpha_iter, beta_iter, gamma_iter)

    def __call__(self, parameter_value, eval=False, file_name=""):

        # build a model with this parameter
        if parameter_value is not None:
            self.model.set_layer_value(self.parameter_name, parameter_value)

        # then run for 20 episodes and get the average loss
        loss = 0
        times = self.config["SPSA"]["n_evals"]
        for i in range(times):
            arm_lengths_1 = np.linspace(0.2, 1.0).tolist()
            arm_lengths_2 = np.linspace(-1.0, -0.2).tolist()
            arm_lengths = arm_lengths_1 + arm_lengths_2

            l1 = np.random.choice(a=arm_lengths, size=2)
            l1 = l1 / np.linalg.norm(l1)
            l2 = np.random.choice(a=arm_lengths, size=2)
            l2 = l2 / np.linalg.norm(l2)
            l3 = np.random.choice(a=arm_lengths, size=2)
            l3 = l3 / np.linalg.norm(l3)
            target = self.get_random_in_circle(self.config["env"]["radius"])
            params = (l1, l2, l3, target)
            loss_run, angles = self.run_episode(self.model, params, eval, file_name)
            loss += loss_run
        loss = loss/times
        return loss, angles

    def get_random_in_circle(self, radius):

        t = 2 * math.pi * np.random.rand()
        u = np.random.rand() + np.random.rand()

        if u > 1:
            r = 2 - u
        else:
            r = u

        result = radius * r * math.cos(t), radius * r * math.sin(t)
        return np.array(result)

class NonLinearFullKinematics:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        # Set up the model
        self.model = model

    def plot(self, ax, l1, l2, l3, initial=False, final=False):
        """
        plot the arms
        :param l1: arm segment 1
        :param l2: arm segment 2
        :param l3: arm segment 3
        :param initial:
        :return:
        """
        ax.add_artist(plt.Circle((0, 0), 0.05, color=JOINTS))
        x = [0]
        y = [0]
        res = l1
        ax.add_artist(plt.Circle((l1[0], l1[1]), 0.05, color=JOINTS))
        x.append(res[0])
        y.append(res[1])
        res = l1 + l2
        ax.add_artist(plt.Circle((res[0], res[1]), 0.05, color=JOINTS))
        x.append(res[0])
        y.append(res[1])
        res = l1 + l2 + l3
        ax.add_artist(plt.Circle((res[0], res[1]), 0.05, color=JOINTS))
        x.append(res[0])
        y.append(res[1])
        if initial:
            plt.plot(x,y, c=ARMS_INIT, linewidth=4.0)
        elif final:
            plt.plot(x,y, c=ARMS_FINAL, linewidth=4.0)
        else:
            plt.plot(x,y, c=ARMS, linewidth=5.0)

    def run_episode(self, model, params, eval, filename):

        # get the parameters
        l1, l2, l3, target = np.copy(params[0]), np.copy(params[1]), np.copy(params[2]), np.copy(params[3])

        # Initialize figure
        if eval:
            fig = plt.figure(num=1, facecolor="white")
            gs = gridspec.GridSpec(nrows=1, ncols=1)
            ax = plt.subplot(gs[0])
            plt.gca().set_aspect('equal', adjustable='box')
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.fill([0, 0, 3, 6], [0, 3, 3, 0], GRID_1)
            ax.fill([0, 3, 3, 0], [0, 0, -3, -3], GRID_2)
            ax.fill([0, 0, -3, -3], [0, -3, -3, 0], GRID_1)
            ax.fill([0, -3, -3, 0], [0, 0, 3, 6], GRID_2)
            self.plot(ax, l1, l2, l3, True)
            circle1 = plt.Circle((target[0], target[1]), 0.2, color=TARGET)
            ax.add_artist(circle1)

        # pass the model to the device
        self.model.net = to_device(self.model.net, self.device)

        # init hidden
        if self.config["model"]["is_rec"]:
            hidden = to_device(self.model.net.init_hidden(), self.device)

        # L1, L2, L3 - in non-linear model, these original lengths are maintained
        L1 = np.linalg.norm(l1)
        L2 = np.linalg.norm(l2)
        L3 = np.linalg.norm(l3)

        alpha_iter = []
        beta_iter = []
        gamma_iter = []

        to_target = 0
        episode_length = self.config["model"]["episode_length"]
        reward_at_each_step = []
        for i in range(episode_length):

            # get the current angles
            alpha, beta, gamma = manipulator_2d_get_angles(l1, l2, l3)

            # difference to the target
            if self.config["model"]["diff"]:
                diff = (target - l1 + l2 + l3) # difference between end-effector to target, just like in reacher environment
            else:
                diff = target # target position

            # in this case, L1, L2, L3, alpha, beta, gamma, R are the inputs
            # do a forward pass
            input = np.array([L1, L2, L3, alpha, beta, gamma, diff[0], diff[1]]).reshape((1,-1))
            input = torch.from_numpy(input).type(torch.float32)

            # send input to device
            input = to_device(input, self.device)

            if self.config["model"]["is_rec"]:
                output, hidden = model.net.forward(input, hidden)
            else:
                output = model.net.forward(input)

            output = output.cpu().data.numpy().reshape((1,-1))[0]
            alpha, beta, gamma = output[0], output[1], output[2]
            #
            # alpha = alpha + alpha_a
            # beta = beta + beta_a
            # gamma = gamma + gamma_a

            # alpha_m, alpha_sd = output[0], output[1]
            # beta_m, beta_sd = output[2], output[3]
            # gamma_m, gamma_sd = output[4], output[5]
            #
            # # action selection based on gaussian policy
            # alpha_action = np.random.normal(loc=alpha_m, scale=np.log(1 + np.exp(alpha_sd)), size=1)[0]
            # beta_action = np.random.normal(loc=beta_m, scale=np.log(1 + np.exp(beta_sd)), size=1)[0]
            # gamma_action = np.random.normal(loc=gamma_m, scale=np.log(1 + np.exp(gamma_sd)), size=1)[0]
            #
            # # apply the action
            # # alpha = alpha + alpha_action
            # # beta = beta + beta_action
            # # gamma = gamma + gamma_action
            #
            # # # # clip the angles here
            # # alpha = np.clip(alpha, -6.29, +6.29)
            # # beta = np.clip(beta, -6.29, +6.29)
            # # gamma = np.clip(gamma, -6.29, +6.29)

            # find corresponding l1, l2, l3 to alpha, beta, gamma (note here we ensure that lengths are maintained)
            l1, l2, l3, _, _ = manipulator_2d_get_arms(alpha, beta, gamma, L1, L2, L3)

            alpha_iter.append(np.rad2deg(alpha))
            beta_iter.append(np.rad2deg(beta))
            gamma_iter.append(np.rad2deg(gamma))

            # plot
            if eval:
                if i == episode_length - 1:
                    self.plot(ax, l1, l2, l3, final=True)
                else:
                    self.plot(ax, l1, l2, l3)

            # finally compare the position with the target position
            error = -np.linalg.norm(l1 + l2 + l3 - target)
            reward_at_each_step.append(error)
            to_target += error

        # loss based on the final position
        loss = to_target

        # save plots
        if eval:
            plt.savefig(filename, bbox_inches="tight")

        return loss, (alpha_iter, beta_iter, gamma_iter, reward_at_each_step)

    def __call__(self, parameter_value, eval=False, file_name=""):

        # build a model with this parameter
        if parameter_value is not None:
            self.model.set_layer_value(self.parameter_name, parameter_value)

        # then run for 20 episodes and get the average loss
        loss = 0
        times = self.config["SPSA"]["n_evals"]
        for i in range(times):
            # randomly generate a target

            arm_lengths_1 = np.linspace(0.2, 1.0).tolist()
            arm_lengths_2 = np.linspace(-1.0, -0.2).tolist()
            arm_lengths = arm_lengths_1 + arm_lengths_2

            l1 = np.random.choice(a=arm_lengths, size=2)
            l1 = l1 / np.linalg.norm(l1)
            l2 = np.random.choice(a=arm_lengths, size=2)
            l2 = l2 / np.linalg.norm(l2)
            l3 = np.random.choice(a=arm_lengths, size=2)
            l3 = l3 / np.linalg.norm(l3)
            target = self.get_random_in_circle(self.config["env"]["radius"])
            params = (l1, l2, l3, target)
            loss_run, angles = self.run_episode(self.model, params, eval, file_name)
            loss += loss_run
        loss = loss/times
        return loss, angles

    def get_random_in_circle(self, radius):

        t = 2 * math.pi * np.random.rand()
        u = np.random.rand() + np.random.rand()

        if u > 1:
            r = 2 - u
        else:
            r = u

        result = radius * r * math.cos(t), radius * r * math.sin(t)
        return np.array(result)