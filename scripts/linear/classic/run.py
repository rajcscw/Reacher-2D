import os
from components.linear_baselines import run_linear

# cur path
cur_path = os.path.dirname(os.path.realpath(__file__))

# save location
save_loc = cur_path + "/outputs"

# run 50 times and get the average error
n_evals = 50
d = 10
episode_length = 20
average_error = run_linear(n_evals, d, episode_length, save_loc)

print("Average Error over {} runs is {}".format(n_evals, average_error))
