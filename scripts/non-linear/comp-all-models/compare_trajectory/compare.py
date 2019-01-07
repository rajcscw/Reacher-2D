import torch.nn as nn
from components.models import SamplingStrategy
from components.models import PyTorchModel
from components.loss2 import NonLinearFullKinematics, NonLinearKinematics
from torch.autograd import Variable
import yaml
import os
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


# iterator over diffs
diffs = [False, True]

for diff in diffs:

    diffID = "with_diff" if diff else "without_diff"

    # models to compare
    models_to_compare = ["mmc", "mlp", "rnn"]

    mmc_angles = []
    mlp_angles = []
    rnn_angles = []
    mmc_rewards = []
    mlp_rewards = []
    rnn_rewards = []
    for model_ in models_to_compare:

        # set up the network based on the model
        if model_ == "mmc":
            all_config = {**general_config, **mmc_config}
            all_config["model"]["is_rec"] = False
        elif model_ =="mlp":
            all_config = {**general_config, **mlp_config}
            all_config["model"]["is_rec"] = False
        elif model_ == "rnn":
            all_config = {**general_config, **rnn_config}
            all_config["model"]["is_rec"] = True

        all_config["model"]["diff"] = diff

        # check the device (CPU vs GPU)
        use_gpu = False
        use_multi_gpu = False
        device = "cuda:0" if torch.cuda.is_available() and use_gpu else "cpu"

        # load the model
        if all_config["model"]["diff"]:
            net = torch.load(dir_path + "/saved_models/final_model_{}_diff".format(model_))
        else:
            net = torch.load(dir_path + "/saved_models/final_model_{}".format(model_))
        model = PyTorchModel(net=net, config=all_config,  device=device, strategy=SamplingStrategy.BOTTOM_UP)

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
        # set a random seed
        np.random.seed(76)

        # run for few iterations and plot the results
        save_folder = dir_path + "/traj_plots"
        error_list = []
        n_evals = 100
        for i in range(n_evals):
            # loss
            os.makedirs(save_folder+"/{}/eval_{}".format(diffID, i+1), exist_ok=True)
            error_run, data = objective(None, True, save_folder+"/{}/eval_{}/{}.pdf".format(diffID, i+1, model_))
            with open(save_folder+"/{}/eval_{}/{}_perf.txt".format(diffID, i+1, model_), "w") as f:
                f.write("Reward is: "+str(error_run))
            error_list.append(error_run)

            alpha, beta, gamma, reward = data

            # somehow, we need to store for each eval
            if model_ == "mmc":
                mmc_angles.append((alpha, beta, gamma))
                mmc_rewards.append(reward)
            elif model_ == "mlp":
                mlp_angles.append((alpha, beta, gamma))
                mlp_rewards.append(reward)
            elif model_ == "rnn":
                rnn_angles.append((alpha, beta, gamma))
                rnn_rewards.append(reward)

        # compute mean and standard deviation here
        with open(model_+"{}_perf.txt".format(diffID), "w") as f:
            mean, sd = np.mean(error_list), np.std(error_list)

            f.write("Mean and SD over {} runs is {}, {}".format(n_evals, mean, sd))
            print("Mean and SD over {} runs is {}, {}".format(n_evals, mean, sd))

    # we are here to plot the angles for each eval comparing across different models
    plot_y_axis = [r'$\alpha$', r'$\beta',r'$\gamma$']
    plot_ylabels = ["alpha", "beta", "gamma"]
    for i in range(len(rnn_angles)):
        flatui = ["#AFDDE8", "#E8C6AF", "#FFEDAA"]
        sns.set_palette(flatui)
        # for each joints
        for j in range(3):
            # we create one plot for each angle
            fig = plt.figure()

            # get for each strategy
            mmc_values = mmc_angles[i][j]
            plt.plot(mmc_values, label="MMC")

            mlp_values = mlp_angles[i][j]
            plt.plot(mlp_values, label="MLP")

            rnn_values = rnn_angles[i][j]
            plt.plot(rnn_values, label="RNN")

            plt.xlabel("time")
            plt.ylabel(plot_y_axis[j])

            plt.legend(loc="lower right", fontsize=10)

            plt.savefig(save_folder+"/{}/eval_{}/{}.pdf".format(diffID, i+1, plot_ylabels[j]))

    # now, let's plot the reward at each step
    final_df = pd.DataFrame()
    for i in range(len(rnn_angles)):

        # get for each strategy
        mmc_values = mmc_rewards[i]
        df = pd.DataFrame({"Time": np.arange(len(mmc_values)), "Reward": mmc_values})
        df["run"] = i
        df["strategy"] = "MMC"
        final_df = final_df.append(df)

        mlp_values = mlp_rewards[i]
        df = pd.DataFrame({"Time": np.arange(len(mlp_values)), "Reward": mlp_values})
        df["run"] = i
        df["strategy"] = "MLP"
        final_df = final_df.append(df)

        rnn_values = rnn_rewards[i]
        df = pd.DataFrame({"Time": np.arange(len(rnn_values)), "Reward": rnn_values})
        df["run"] = i
        df["strategy"] = "RNN"
        final_df = final_df.append(df)
    fig = plt.figure()
    sns.set(style="dark")
    sns.set_context("paper")
    g = sns.lineplot(data=final_df, x="Time", hue="strategy", y="Reward", ci="sd", err_style="band")
    plt.ylabel("Reward", fontsize=12)
    plt.xlabel("Time", fontsize=12)
    plt.legend(loc=4, fontsize=12)
    plt.savefig(save_folder+"/reward_comp.pdf")

