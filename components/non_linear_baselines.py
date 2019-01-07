from components.manipulator import manipulator_2d_inverse_iterate, manipulator_2d_get_angles, apply_rotation, manipulator_2d_get_arms, controller
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import os
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


def plot(ax, l1, l2, l3, initial=False, final=False):
    """
    plot the arms
    :param l1: arm segment 1
    :param l2: arm segment 2
    :param l3: arm segment 3
    :param initial:
    :return:
    """
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
    #ax.add_artist(plt.Circle((l1[0], l1[1]), 0.1, color=JOINTS))
    x.append(res[0])
    y.append(res[1])
    res = l1 + l2
    #ax.add_artist(plt.Circle((res[0], res[1]), 0.1, color=JOINTS))
    x.append(res[0])
    y.append(res[1])
    res = l1 + l2 + l3
    #ax.add_artist(plt.Circle((res[0], res[1]), 0.1, color=ENDS))
    x.append(res[0])
    y.append(res[1])
    if initial:
        plt.plot(x,y, c=ARMS_INIT, linewidth=4.0)
    elif final:
        plt.plot(x,y, c=ARMS_FINAL, linewidth=4.0)
    else:
        plt.plot(x,y, c=ARMS, linewidth=5.0)


def get_random_in_circle(radius):

    t = 2 * math.pi * np.random.rand()
    u = np.random.rand() + np.random.rand()

    if u > 1:
        r = 2 - u
    else:
        r = u

    result = radius * r * math.cos(t), radius * r * math.sin(t)
    return np.array(result)


def run(n_evals, d, episode_length, save_loc=None):
    total_error = 0
    for j in range(n_evals):
        arm_lengths_1 = np.linspace(0.2, 1.0).tolist()
        arm_lengths_2 = np.linspace(-1.0, -0.2).tolist()
        arm_lengths = arm_lengths_1 + arm_lengths_2

        l1 = np.random.choice(a=arm_lengths, size=2)
        l1 = l1 / np.linalg.norm(l1)
        l2 = np.random.choice(a=arm_lengths, size=2)
        l2 = l2 / np.linalg.norm(l2)
        l3 = np.random.choice(a=arm_lengths, size=2)
        l3 = l3 / np.linalg.norm(l3)
        extent = 2.5
        r = get_random_in_circle(extent)

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
        plot(ax, l1, l2, l3, True)
        circle1 = plt.Circle((r[0], r[1]), 0.2, color=TARGET)
        ax.add_artist(circle1)
        alpha_iter = []
        beta_iter = []
        gamma_iter = []

        alpha, beta, gamma = manipulator_2d_get_angles(l1, l2, l3)

        # L1, L2, L3
        L1 = np.linalg.norm(l1)
        L2 = np.linalg.norm(l2)
        L3 = np.linalg.norm(l3)
        to_target = 0
        for i in range(episode_length):

            alpha_iter.append(np.rad2deg(alpha))
            beta_iter.append(np.rad2deg(beta))
            gamma_iter.append(np.rad2deg(gamma))

            # manipulator
            alpha, beta, gamma, l1_, l2_, l3_, _ = manipulator_2d_inverse_iterate(alpha, beta, gamma, L1, L2, L3, r, d)

            # find corresponding l1, l2, l3 to alpha, beta, gamma (note here we ensure that lengths are maintained)
            l1, l2, l3, _, _ = manipulator_2d_get_arms(alpha, beta, gamma, L1, L2, L3)

            # this is a bug I guess
            # # controller - apply the actions and get the new arm vectors
            # # it justs apply rotations (differences between previous values and applies them ?)
            # l1, l2, l3 = controller(l1, l2, l3, alpha, beta, gamma)

            # finally compare the position with the target position ()
            error = -np.linalg.norm(l1 + l2 + l3 - r)
            to_target += error

            plot(ax, l1, l2, l3, final=True if (i == episode_length - 1) else False)

        # total error
        total_error += np.sum(to_target)

        # plt.subplot(gs[1])
        # plt.plot(alpha_iter, c="black")
        # plt.xlabel("iter")
        # plt.ylabel(r'$\alpha$')
        #
        # plt.subplot(gs[2])
        # plt.plot(beta_iter, c="black")
        # plt.xlabel("iter")
        # plt.ylabel(r'$\beta$')
        #
        # plt.subplot(gs[3])
        # plt.plot(gamma_iter, c="black")
        # plt.xlabel("iter")
        # plt.ylabel(r'$\gamma$')
        if save_loc is not None:
            plt.savefig(save_loc + "/2D_manip_"+str(j+1)+".pdf")

    # return the average end-effector error
    average_error = total_error / n_evals
    return average_error