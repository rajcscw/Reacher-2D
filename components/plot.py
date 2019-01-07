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


def plot(ax, l1, l2, l3, initial=False, final=False):
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


def get_random_in_circle(radius):

    a = np.random.random() * 2 * np.pi
    r = radius * np.sqrt(np.random.random())

    x = r * np.cos(a)
    y = r * np.sin(a)

    result = np.array([x, y])
    return result

# randomly generate a target
l1 = np.array([2,1])
l1 = l1 / np.linalg.norm(l1)
l2 = np.array([-0.5,0.5])
l2 = l2 / np.linalg.norm(l2)
l3 = np.array([1,1])
l3 = l3 / np.linalg.norm(l3)
target = np.array([1.7,2])


fig = plt.figure(num=1, facecolor="white")
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax = plt.subplot(gs[0])
plt.gca().set_aspect('equal', adjustable='box')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xticks([])
ax.set_yticks([])
ax.fill([0, 0, 3, 3], [0, 3, 3, 0], GRID_1)
ax.fill([0, 3, 3, 0], [0, 0, -3, -3], GRID_2)
ax.fill([0, 0, -3, -3], [0, -3, -3, 0], GRID_1)
ax.fill([0, -3, -3, 0], [0, 0, 3, 3], GRID_2)
plot(ax, l1, l2, l3)
circle1 = plt.Circle((target[0], target[1]), 0.2, color=TARGET)
ax.add_artist(circle1)
plt.savefig("reacher_env.pdf", bbox_inches="tight")