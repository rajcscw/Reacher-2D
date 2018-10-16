from components.manipulator import manipulator_2d_inverse_iterate, manipulator_2d_get_angles, apply_rotation, manipulator_2d_get_arms, controller
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import os

# set the backend
plt.switch_backend('agg')


def plot(l1, l2, l3, initial=False, final=False):
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


def run(n_evals, d, episode_length, save_loc=None):
    total_error = 0
    for j in range(n_evals):
        l1 = np.random.randint(-3,3,2)
        l2 = np.random.randint(-3,3,2)
        l3 = np.random.randint(-3,3,2)
        d1 = l1 + l2
        d2 = l2 + l3
        extent = 5
        r = np.random.randint(-extent, extent, size=2)

        # Initialize figure
        fig = plt.figure(num=1, facecolor="white")
        gs = gridspec.GridSpec(nrows=2, ncols=2)

        # plot the initial arm positions and target
        ax = plt.subplot(gs[0])
        plt.gca().set_aspect('equal', adjustable='box')
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        plt.xlabel("iter")
        ax.fill([0, 0, extent, extent], [0, extent, extent, 0], "b")
        ax.fill([0, extent, extent, 0], [0, 0, -extent, -extent], "cyan")
        ax.fill([0, 0, -extent, -extent], [0, -extent, -extent, 0], "b")
        ax.fill([0, -extent, -extent, 0], [0, 0, extent, extent], "cyan")
        plot(l1, l2, l3, True)
        circle1 = plt.Circle((r[0], r[1]), 0.5, color='g')
        ax.add_artist(circle1)
        alpha_iter = []
        beta_iter = []
        gamma_iter = []

        # find angles
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

            # controller - apply the actions and get the new arm vectors
            # it justs apply rotations (differences between previous values and applies them ?)
            l1, l2, l3 = controller(l1, l2, l3, alpha, beta, gamma)

            # finally compare the position with the target position
            error = (l1 + l2 + l3 - r)**2
            to_target = error

            plot(l1, l2, l3, final=True if (i == episode_length - 1) else False)

        # total error
        total_error += np.sum(to_target)

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
        if save_loc is not None:
            plt.savefig(save_loc + "/2D_manip_"+str(j+1)+".pdf")

    # return the average end-effector error
    average_error = total_error / n_evals
    return average_error