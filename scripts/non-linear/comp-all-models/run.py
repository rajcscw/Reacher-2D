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
from components.loss2 import NonLinearFullKinematics, NonLinearKinematics
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


# set up the network architecture
class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer = nn.Linear(in_features=self.input_size, out_features=self.output_size)

    def forward(self, input):
        input = input.reshape((1,1,self.input_size))
        output = self.layer(input)
        return output

if __name__ == "__main__":

    init_multiproc()

    # read all the config here
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path+"/general_config.yml", 'r') as ymlfile:
        general_config = yaml.load(ymlfile)
    print(general_config)

    # mmc config
    with open(dir_path+"/mmc_config.yml", 'r') as ymlfile:
        mmc_config = yaml.load(ymlfile)
    print(mmc_config)

    # mlp config
    with open(dir_path+"/mlp_config.yml", 'r') as ymlfile:
        mlp_config = yaml.load(ymlfile)
    print(mlp_config)

    # rnn config
    with open(dir_path+"/rnn_config.yml", 'r') as ymlfile:
        rnn_config = yaml.load(ymlfile)
    print(rnn_config)

    # create folder name
    folder_name = os.path.dirname(os.path.realpath(__file__)) + "/outputs/Comparison-all-"+str(datetime.now())
    os.mkdir(folder_name)
    os.mkdir(folder_name+"/eval")

    diff = general_config["env"]["diff"]

    episodic_target_error_strategy = pd.DataFrame()
    episodic_target_error_strategy_unsmoothened = pd.DataFrame()

    # models to compare
    models_to_compare = ["mmc", "mlp", "rnn"]

    for model_ in models_to_compare:

        print("\n Running model {} with difference as {}".format(model_, general_config["env"]["diff"]))

        for k in range(int(general_config["log"]["runs"])):

            print("----Iteration: {}".format(k+1))

            # check the device (CPU vs GPU)
            use_gpu = False
            use_multi_gpu = False
            device = "cuda:0" if torch.cuda.is_available() and use_gpu else "cpu"

            # set up the network based on the model
            if model_ == "mmc":
                net = Linear(input_size=8, output_size=6)
                all_config = {**general_config, **mmc_config}
            elif model_ =="mlp":
                # here inputs are: L1, L2, L3, alpha, beta, gamma, R (therefore, in total 8)
                # and outputs are: alpha, beta, gamma
                net = FullController(input_size=8, output_size=3, hidden_size=mlp_config["model"]["hidden_size"])
                all_config = {**general_config, **mlp_config}
                all_config["model"]["is_rec"] = False
            elif model_ == "rnn":
                # here inputs are: L1, L2, L3, alpha, beta, gamma, R (therefore, in total 8)
                # and outputs are: alpha, beta, gamma
                net = RNN(input_size=8, output_size=3, hidden_size=rnn_config["model"]["hidden_size"], n_layers=rnn_config["model"]["layers"])
                all_config = {**general_config, **rnn_config}
                all_config["model"]["is_rec"] = True

            # set if the difference
            all_config["model"]["diff"] = diff

            # set up the model
            model = PyTorchModel(net=net, config=all_config,  device=device, strategy=SamplingStrategy.ALL)

            # the objective function
            if model_ == "mmc":
                objective = NonLinearKinematics(model=model,
                                                config=all_config,
                                                device=device
                                                )
            else:
                objective = NonLinearFullKinematics(model=model,
                                                config=all_config,
                                                device=device
                                                )

            # optimizer
            optimizer = RMSProp(learning_rate=float(all_config["SPSA"]["a"]),
                                decay_rate=float(all_config["SPSA"]["momentum"]),
                                A=float(all_config["SPSA"]["A"]),
                                alpha=float(all_config["SPSA"]["alpha"]))

            # estimator
            estimator = SPSA(parallel_workers=int(all_config["SPSA"]["n_workers"]),
                             a=float(all_config["SPSA"]["a"]),
                             c=float(all_config["SPSA"]["c"]),
                             A=float(all_config["SPSA"]["A"]),
                             k=int(all_config["SPSA"]["k"]),
                             alpha=float(all_config["SPSA"]["alpha"]),
                             gamma=float(all_config["SPSA"]["gamma"]),
                             param_decay=float(all_config["SPSA"]["decay"]),
                             loss_function=objective,
                             model=model,
                             obj_scaler=None,
                             device=device)

            # the main loop
            episodic_reward = []
            running_reward = 0
            max_iter = int(all_config["log"]["iterations"])
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
                loss, _ = objective(updated_layer_value)
                running_reward += loss

                print("\rProcessed iteration {} of {}".format(i, max_iter), end="")
                if i % int(general_config["log"]["average_every"]) == 0 and i != 0:
                    running_reward = running_reward / int(all_config["log"]["average_every"])
                    last_running_reward = running_reward
                    print(",Evaluating at iteration: {}, Total Running Reward {}, gamma: {}, lr: {}".format(i, running_reward, estimator.get_perturb_scale(), optimizer.get_learning_rate()))
                    running_reward = 0

                # book keeping stuff
                episodic_reward.append(loss)

                # step optimizer
                optimizer.step_t()
                estimator.step_t()

            # save the final model
            torch.save(model.net.cpu(), folder_name+"/final_model_{}_run_{}_last_reward_{}".format(model_, k, last_running_reward))

            # clip the negative values, just to make the plot pretty
            episodic_reward = [-50 if item < -50 else item for item in episodic_reward]

            # Compute the running mean
            N = int(all_config["log"]["average_every"])
            unsmoothened = episodic_reward
            selected = rolling_mean(episodic_reward, 50).tolist()

            # Combine all runs and add them to the strategy dataframe
            df = pd.DataFrame({"Iteration": np.arange(len(selected)), "Episodic Total Reward": selected})
            df["run"] = k
            df["strategy"] = model_.upper()
            episodic_target_error_strategy = episodic_target_error_strategy.append(df)

            # Combine all runs and add them to the strategy dataframe
            df = pd.DataFrame({"Iteration": np.arange(len(unsmoothened)), "Episodic Total Reward": unsmoothened})
            df["run"] = k
            df["strategy"] = model_.upper()
            episodic_target_error_strategy_unsmoothened = episodic_target_error_strategy_unsmoothened.append(df)

    # Plot the learning curves here
    episodic_target_error_strategy.to_pickle(folder_name + "/learning_curve_df")
    plot_learning_curve(folder_name +"/_Episodic_total_reward.pdf", "Reacher", episodic_target_error_strategy, "Episodic Total Reward")

    # Plot the learning curves here
    episodic_target_error_strategy_unsmoothened.to_pickle(folder_name + "/learning_curve_unsmoothened_df")
    plot_learning_curve(folder_name +"/_Episodic_total_reward_unsmooth.pdf", "Reacher", episodic_target_error_strategy_unsmoothened, "Episodic Total Reward")

    # Log all the config parameters
    file = open(folder_name+"/general_config.txt", "w")
    file.write(str(general_config))
    file.close()

    # Log all the config parameters
    file = open(folder_name+"/mmc_config.txt", "w")
    file.write(str(mmc_config))
    file.close()

    # Log all the config parameters
    file = open(folder_name+"/mlp_config.txt", "w")
    file.write(str(mlp_config))
    file.close()
