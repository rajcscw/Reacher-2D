import numpy as np


def manipulator_2d_direct_iterate(l1, l2, l3, d1, d2, r, d):
    l1_ = (1 / (4 + d)) * (-2 * l2 + 2 * r - 2 * d2 + 2 * d1 + d * l1)
    l2_ = (1 / (4 + d)) * (-2 * l3 - 2 * l1 + 2 * d2 + 2 * d1 + d * l2)
    l3_ = (1 / (4 + d)) * (-2 * l2 + 2 * r + 2 * d2 - 2 * d1 + d * l3)
    d1_ = (1 / (4 + d)) * (-2 * l3 + 2 * l2 + 2 * l1 + 2 * r + d * d1)
    d2_ = (1 / (4 + d)) * (2 * l3 + 2 * l2 - 2 * l1 + 2 * r + d * d2)
    r_ = (1 / (4 + d)) * (2 * l3 + 2 * l1 + 2 * d2 + 2 * d1 + d * r)
    return l1_, l2_, l3_, d1_, d2_, r_


def manipulator_2d_inverse_iterate(alpha, beta, gamma, L1, L2, L3, r, d):

    # find corresponding l1, l2, l3 to alpha, beta, gamma
    l1_, l2_, l3_, d1_, d2_ = manipulator_2d_get_arms(alpha, beta, gamma, L1, L2, L3)

    # iterate
    l1_, l2_, l3_, _, _, _ = manipulator_2d_direct_iterate(l1_, l2_, l3_, d1_, d2_, r, d)

    # find angles back
    alpha_, beta_, gamma_ = manipulator_2d_get_angles(l1_, l2_, l3_)

    return alpha_, beta_, gamma_, r, l1_, l2_, l3_


def apply_rotation(vector, theta):
    x = vector[0] * np.cos(theta) - vector[1] * np.sin(theta)
    y = vector[0] * np.sin(theta) + vector[1] * np.cos(theta)
    return np.array([x, y])


def find_angle(vector):
    theta = np.arctan2(vector[1], vector[0])
    return theta


def find_vector(l, theta):
    vector_x = l * np.cos(theta)
    vector_y = l * np.sin(theta)
    vector_ = np.array([vector_x, vector_y])
    return vector_


def manipulator_2d_get_angles(l1, l2, l3):

    # alpha
    alpha = find_angle(l1)

    # beta
    # rotate l2 to alpha first
    l2_ = apply_rotation(l2, -alpha)
    beta = find_angle(l2_)

    # gamma
    # rotate l3 to alpha first
    l3_ = apply_rotation(l3, -alpha)
    l3_ = apply_rotation(l3_, -beta)
    gamma = find_angle(l3_)

    return alpha, beta, gamma


def manipulator_2d_get_angles_2j(l1, l2):

    # alpha
    alpha = find_angle(l1)

    # beta
    # rotate l2 to alpha first
    l2_ = apply_rotation(l2, -alpha)
    beta = find_angle(l2_)

    return alpha, beta


def manipulator_2d_get_arms(alpha, beta, gamma, l1, l2, l3):

    # l1
    l1_ = find_vector(l1, alpha)

    # l2
    l2_ = find_vector(l2, beta)
    l2_ = apply_rotation(l2_, alpha)

    # l3
    l3_ = find_vector(l3, gamma)
    l3_ = apply_rotation(l3_, beta)
    l3_ = apply_rotation(l3_, alpha)

    # d1
    d1_ = l1_ + l2_

    # d2
    d2_ = l2_ + l3_

    return l1_, l2_, l3_, d1_, d2_


def controller(l1, l2, l3, alpha_new, beta_new, gamma_new):
    # get current angles
    alpha, beta, gamma = manipulator_2d_get_angles(l1, l2, l3)

    # apply the transformations
    last = alpha_new - alpha
    l1_ = apply_rotation(l1, last)
    last = last + beta_new - beta
    l2_ = apply_rotation(l2, last)
    last = last + gamma_new - gamma
    l3_ = apply_rotation(l3, last)

    return l1_, l2_, l3_


def controller_2j(l1, l2, alpha_new, beta_new):
    # get current angles
    alpha, beta, = manipulator_2d_get_angles_2j(l1, l2)

    # apply the transformations
    last = alpha_new - alpha
    l1_ = apply_rotation(l1, last)
    last = last + beta_new - beta
    l2_ = apply_rotation(l2, last)

    return l1_, l2_