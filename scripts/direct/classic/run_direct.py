from components.manipulator import manipulator_2d_direct_iterate, manipulator_2d_get_angles
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import os


# cur path
cur_path = os.path.dirname(os.path.realpath(__file__))


def plot(l1, l2, l3, initial=False):
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
    else:
        plt.plot(x,y, c="black")


for j in range(10):
    # configuration of arm segment lengths - they are going to be fixed throughout
    # only their orientations will be adjusted
    l1 = np.array([3,5])
    l2 = np.array([0,2])
    l3 = np.array([0,0])
    d1 = l1 + l2
    d2 = l2 + l3
    d = 3

    # generate a random target
    r = np.random.randint(-5,5, size=2)

    # Initialize figure
    fig = plt.figure(num=1, facecolor="white")
    gs = gridspec.GridSpec(nrows=2, ncols=2)

    # plot the initial arm positions and target
    ax = plt.subplot(gs[0])
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.xlabel("iter")
    ax.fill([0, 0, 10, 10], [0, 10, 10, 0], "b")
    ax.fill([0, 10, 10, 0], [0, 0, -10, -10], "cyan")
    ax.fill([0, 0, -10, -10], [0, -10, -10, 0], "b")
    ax.fill([0, -10, -10, 0], [0, 0, 10, 10], "cyan")
    plot(l1, l2, l3, True)
    circle1 = plt.Circle((r[0], r[1]), 0.5, color='g')
    ax.add_artist(circle1)
    #plt.scatter(r[0], r[1], c="magenta")

    # to store controller angles
    alpha_iter = []
    beta_iter = []
    gamma_iter = []

    # find angles
    alpha, beta, gamma = manipulator_2d_get_angles(l1, l2, l3)
    for i in range(20):

        l1, l2, l3, d1, d2, _ = manipulator_2d_direct_iterate(l1, l2, l3, d1, d2, r, d)
        print("l1: {} l2: {} l3: {}".format(l1, l2, l3))
        plot(l1, l2, l3)

        alpha, beta, gamma = manipulator_2d_get_angles(l1, l2, l3)
        alpha_iter.append(np.rad2deg(alpha))
        beta_iter.append(np.rad2deg(beta))
        gamma_iter.append(np.rad2deg(gamma))
    print("Alpha: {}, Beta: {}, Gamma: {}".format(np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma)))

    plt.subplot(gs[1])
    plt.plot(alpha_iter, c="black")
    plt.xlabel("iter")
    plt.ylabel(r'$\alpha$')

    plt.subplot(gs[2])
    plt.plot(beta_iter, c="black")
    plt.xlabel("iter")
    plt.ylabel(r'$\beta$')

    plt.subplot(gs[3])
    plt.plot(gamma_iter, c="black")
    plt.xlabel("iter")
    plt.ylabel(r'$\gamma$')
    #plt.show()
    plt.savefig(cur_path+"/outputs/2D_manip_"+str(j+1)+".pdf")
