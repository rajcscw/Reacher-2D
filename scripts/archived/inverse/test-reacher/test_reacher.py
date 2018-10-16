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

# Render env output to video
video.write(grabFrame(env))

# get l1 and l2
data = env.physics.named.data.geom_xpos

# arm, hand and finger
l1 = (data["arm"] - data["root"])[0:2] # just get the x and y
l2 = data["finger"][0:2] - data["arm"][0:2] # just get the x and y
l3 = np.zeros(2) # just get the x and y

# get the target
target = data["target"][0:2]

# get the figure
plot_scale = 0.2
fig = plt.figure(num=1, facecolor="white")
gs = gridspec.GridSpec(nrows=2, ncols=2)
ax = plt.subplot(gs[0])
plt.gca().set_aspect('equal', adjustable='box')
ax.set_xlim(-plot_scale, plot_scale)
ax.set_ylim(-plot_scale, plot_scale)
plt.xlabel("iter")
circle1 = plt.Circle((data["arm"][0], data["arm"][1]), 1e-2, color='g')
ax.add_artist(circle1)
circle1 = plt.Circle((data["finger"][0], data["finger"][1]), 1e-2, color='g')
ax.add_artist(circle1)
ax.fill([0, 0, plot_scale, plot_scale], [0, plot_scale, plot_scale, 0], "b")
ax.fill([0, plot_scale, plot_scale, 0], [0, 0, -plot_scale, -plot_scale], "cyan")
ax.fill([0, 0, -plot_scale, -plot_scale], [0, -plot_scale, -plot_scale, 0], "b")
ax.fill([0, -plot_scale, -plot_scale, 0], [0, 0, plot_scale, plot_scale], "cyan")
plot(l1, l2, l3, True)
circle1 = plt.Circle((target[0], target[1]), 1e-2, color='g')
ax.add_artist(circle1)

step_count = 0
while not time_step.last() and step_count < 50:

    # same action
    action = np.array([0.0, -1.0]) # only on the first action which is what??

    # next step
    time_step = env.step(action)

    # get frame
    # Render env output to video
    video.write(grabFrame(env))


plt.savefig(cur_path+"/2D_manip_reacher.pdf")

# End render to video file
video.release()

# Exit
cv2.destroyAllWindows()