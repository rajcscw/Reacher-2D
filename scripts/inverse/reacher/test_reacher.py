from dm_control import suite
import numpy as np
import cv2
from components.manipulator import manipulator_2d_inverse_iterate, manipulator_2d_get_angles, apply_rotation, manipulator_2d_get_arms, controller
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import os
import seaborn as sns


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

def grabFrame(env):
    # Get RGB rendering of env
    rgbArr = env.physics.render(480, 600, camera_id=0)
    # Convert to BGR for use with OpenCV
    return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)


# cur path
cur_path = os.path.dirname(os.path.realpath(__file__))

# Load task:
env = suite.load(domain_name="reacher", task_name="easy")

# Setup video writer - mp4 at 30 fps
video_name = 'video.mp4'
frame = grabFrame(env)
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))


# First pass - Step through an episode and capture each frame
action_spec = env.action_spec()
time_step = env.reset()
d = 1 # damping factor

# get l1 and l2
data = env.physics.named.data.geom_xpos

# arm, hand and finger
l1 = (data["arm"] - data["root"])[0:2] # just get the x and y
l2 = (data["finger"] - data["arm"])[0:2] # just get the x and y
l3 = np.array([0,0]) # just get the x and y

# get the target
r = data["target"][0:2]

# arm segment lengths
L1 = np.linalg.norm(l1)
L2 = np.linalg.norm(l2)
L3 = np.linalg.norm(l3)

# get the angles made by the arm segments
alpha, beta, gamma = manipulator_2d_get_angles(l1, l2, l3)


# to store controller angles
alpha_iter = []
beta_iter = []
gamma_iter = []

# Initialize figure
fig = plt.figure(num=1, facecolor="white")
gs = gridspec.GridSpec(nrows=2, ncols=2)

# plot the initial arm positions and target
ax = plt.subplot(gs[0])
#ax.set_facecolor("xkcd:sky blue")
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("iter")
plot(l1, l2, l3, True)

plt.scatter(data["arm"][0], data["arm"][1], c="red")
plt.scatter(data["finger"][0], data["finger"][1], c="red")

plt.scatter(r[0], r[1], c="green")

episodic_reward = 0
step_count = 0
speed = 1e-1
while not time_step.last() and step_count < 200:

    # log angles
    alpha_iter.append(np.rad2deg(alpha))
    beta_iter.append(np.rad2deg(beta))
    gamma_iter.append(np.rad2deg(gamma))

    # manipulator - find end effective angles
    alpha_next, beta_next, gamma_next, l1_, l2_, l3_, _ = manipulator_2d_inverse_iterate(alpha, beta, gamma, L1, L2, L3, r, d)

    # convert that to actions
    # basically convert alpha, beta to actuator torques
    action_joint_1 = +1 * speed if (alpha_next - alpha) > 0 else -1 * speed
    action_joint_2 = +1 * speed if (beta_next - beta) > 0 else -1 * speed

    # perform the action
    action = np.array([action_joint_1, action_joint_2])
    next_time_step = env.step(action)

    # Render env output to video
    video.write(grabFrame(env))

    episodic_reward += next_time_step.reward

    # next state
    alpha = alpha_next
    beta = beta_next
    time_step = next_time_step
    step_count += 1

    # controller - apply the actions and get the new arm vectors
    # it justs apply rotations (differences between previous values and applies them ?)
    l1, l2, l3 = controller(l1, l2, l3, alpha, beta, gamma)

    plot(l1, l2, l3)

print("Episodic reward: {}".format(episodic_reward))

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
plt.savefig(cur_path+"/2D_manip_reacher.pdf")

# End render to video file
video.release()

# Exit
cv2.destroyAllWindows()