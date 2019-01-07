import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.autograd import Variable

from components.SGDOptimizers import RMSProp
from components.estimators import SPSA
from components.loss2 import NonLinearFullKinematics
from components.models import PyTorchModel
from components.models import SamplingStrategy
from components.non_linear_baselines import run
from components.utility import plot_learning_curve, rolling_mean, init_multiproc


# set up the network architecture
class FullController(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(FullController, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_1 = nn.Linear(in_features=self.input_size, out_features=self.hidden_size)
        self.layer_2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.layer_3 = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, input):
        input = input.reshape((1,1,self.input_size))
        input = torch.tanh(self.layer_1(input))
        input = torch.tanh(self.layer_2(input))
        output = self.layer_3(input) # just a linear at the output
        return output


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.rnn = nn.GRU(input_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):

        # may have to deal with batch inputs
        input = input.view(1, 1, -1)

        # batch forward - RNN
        output, hidden = self.rnn(input, hidden)

        # may have to reshape
        output = output.view(-1, self.hidden_size)

        output = self.decoder(output)

        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

if __name__ == "__main__":

    init_multiproc()

    # read the config
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path+"/config.yml", 'r') as ymlfile:
        config = yaml.load(ymlfile)
    print(config)

    # create folder name
    folder_name = os.path.dirname(os.path.realpath(__file__)) + "/outputs/Comparison-all-"+str(datetime.now())
    os.mkdir(folder_name)
    os.mkdir(folder_name+"/eval")

    # diffs config
    diffs_config = [True]

    episodic_target_error_strategy = pd.DataFrame()
    episodic_target_error_strategy_unsmoothened = pd.DataFrame()
    for diff in diffs_config:

        # hack the diff
        config["model"]["diff"] = diff

        print("Running with error difference as {}".format(config["model"]["diff"]))

        for k in range(int(config["log"]["runs"])):
            # check the device (CPU vs GPU)
            use_gpu = False
            use_multi_gpu = False
            device = "cuda:0" if torch.cuda.is_available() and use_gpu else "cpu"

            # here inputs are: L1, L2, L3, alpha, beta, gamma, R (therefore, in total 8)
            # and outputs are: alpha, beta, gamma
            if config["model"]["is_rec"]:
                net = RNN(input_size=8, output_size=3, hidden_size=config["model"]["hidden_size"], n_layers=2)
            else:
                net = FullController(input_size=8, output_size=3, hidden_size=config["model"]["hidden_size"])

            # set up the model
            model = PyTorchModel(net=net, config=config,  device=device, strategy=SamplingStrategy.ALL)

            # the objective function
            objective = NonLinearFullKinematics(model=model,
                                                config=config,
                                                device=device
                                                )

            # optimizer
            optimizer = RMSProp(learning_rate=float(config["SPSA"]["a"]),
                                decay_rate=float(config["SPSA"]["momentum"]),
                                A=float(config["SPSA"]["A"]),
                                alpha=float(config["SPSA"]["alpha"]))

            # estimator
            estimator = SPSA(parallel_workers=int(config["SPSA"]["n_workers"]),
                             a=float(config["SPSA"]["a"]),
                             c=float(config["SPSA"]["c"]),
                             A=float(config["SPSA"]["A"]),
                             k=int(config["SPSA"]["k"]),
                             alpha=float(config["SPSA"]["alpha"]),
                             gamma=float(config["SPSA"]["gamma"]),
                             param_decay=float(config["SPSA"]["decay"]),
                             loss_function=objective,
                             model=model,
                             obj_scaler=None,
                             device=device)

            # the main loop
            episodic_reward = []
            running_reward = 0
            solution_found = False
            max_iter = int(config["log"]["iterations"])
            for i in range(max_iter):
                # get the current parameter name and value
                current_layer_name, current_layer_value = model.sample_layer()

                # estimator
                gradients = estimator.estimate_gradient(current_layer_name, current_layer_value)

                # optimizer
                updated_layer_value = optimizer.step(current_layer_name, current_layer_value, gradients)

                # update the parameter
                model.set_layer_value(current_layer_name, updated_layer_value)

                # evaluate the learning
                objective.parameter_name = current_layer_name
                render = (i % config["log"]["eval_every"] == 0)
                loss = objective(updated_layer_value, render, folder_name+"/eval/eval_{}.pdf".format(i))
                running_reward += loss

                print("\rProcessed iteration {} of {}".format(i, max_iter), end="")
                if i % int(config["log"]["average_every"]) == 0 and i != 0:
                    running_reward = running_reward / int(config["log"]["average_every"])
                    last_running_reward = running_reward
                    print(",Evaluating at iteration: {}, Total Running Reward {}, gamma: {}, lr: {}".format(i, running_reward, estimator.get_perturb_scale(), optimizer.get_learning_rate()))
                    running_reward = 0

                # book keeping stuff
                episodic_reward.append(loss)

                # step optimizer
                optimizer.step_t()
                estimator.step_t()

            # save the final model
            model_ID = "With_target_error" if diff else "with_target_position"
            torch.save(model.net.cpu(), folder_name+"/final_model_{}_run_{}_last_reward_{}".format(model_ID, k, last_running_reward))

            # Compute the running mean
            N = int(config["log"]["average_every"])
            unsmoothened = episodic_reward
            selected = rolling_mean(episodic_reward, 20).tolist()

            # Combine all runs and add them to the strategy dataframe
            df = pd.DataFrame({"Iteration": np.arange(len(selected)), "Episodic Total Reward": selected})
            df["run"] = k
            df["strategy"] = "with_target_diff" if diff else "with_target_pos"
            episodic_target_error_strategy = episodic_target_error_strategy.append(df)

            # Combine all runs and add them to the strategy dataframe
            df = pd.DataFrame({"Iteration": np.arange(len(unsmoothened)), "Episodic Total Reward": unsmoothened})
            df["run"] = k
            df["strategy"] = "with_target_diff" if diff else "with_target_pos"
            episodic_target_error_strategy_unsmoothened = episodic_target_error_strategy_unsmoothened.append(df)

    for k in range(int(config["log"]["runs"])):
        # get the baselines and add them to plot
        baseline = run(config["SPSA"]["n_evals"], 5, config["model"]["episode_length"])
        baseline = np.ones_like(selected) * baseline
        df = pd.DataFrame({"Iteration": np.arange(len(selected)), "Episodic Total Reward": baseline})
        df["run"] = k
        df["strategy"] = "baseline_MMC"
        episodic_target_error_strategy = episodic_target_error_strategy.append(df)

    # Plot the learning curves here
    episodic_target_error_strategy.to_pickle(folder_name + "/learning_curve_df")
    plot_learning_curve(folder_name +"/_Episodic_total_reward.pdf", "Reacher - End to End", episodic_target_error_strategy, "Episodic Total Reward")

    # Plot the learning curves here
    episodic_target_error_strategy_unsmoothened.to_pickle(folder_name + "/learning_curve_unsmoothened_df")
    plot_learning_curve(folder_name +"/_Episodic_total_reward_unsmooth.pdf", "Reacher - End to End", episodic_target_error_strategy_unsmoothened, "Episodic Total Reward")

    # Log all the config parameters
    file = open(folder_name+"/model_config.txt", "w")
    file.write(str(config))
    file.close()