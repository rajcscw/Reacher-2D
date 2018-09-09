import torch
import torch.nn as nn
from components.models import SamplingStrategy
from components.estimators import SPSA
from components.models import PyTorchModel
from components.SGDOptimizers import RMSProp, SGD
from components.loss import InverseKinematic
from torch.autograd import Variable
import pandas as pd
import yaml
import os
import numpy as np
from components.utility import plot_learning_curve, rolling_mean, init_multiproc
from datetime import datetime
import torch


# set up the network architecture
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = input.reshape((1,1,self.input_size))
        output, hidden = self.rnn(input, hidden)
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

    # read the config
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path+"/config.yml", 'r') as ymlfile:
        config = yaml.load(ymlfile)
    print(config)

    # create folder name
    folder_name = os.path.dirname(os.path.realpath(__file__)) + "/outputs/Comparison-all-"+str(datetime.now())
    os.mkdir(folder_name)
    os.mkdir(folder_name+"/eval")

    episodic_total_rewards_strategy = pd.DataFrame()
    for k in range(int(config["log"]["runs"])):
        # check the device (CPU vs GPU)
        use_gpu = False
        use_multi_gpu = False
        device = "cuda:0" if torch.cuda.is_available() and use_gpu else "cpu"

        # input size
        # l1_x, l1_y, l2_x, l2_y, l3_x, l3_y, d1_x, d1_y, d2_y, d2_y, r_x, r_y

        # output size
        # l1_x, l1_y, l2_x, l2_y, l3_x, l3_y, d1_x, d1_y, d2_y, d2_y,

        net = Linear(input_size=12, output_size=10)

        # set up the model
        model = PyTorchModel(net=net, config=config,  device=device, strategy=SamplingStrategy.BOTTOM_UP)

        # the objective function
        objective = InverseKinematic(model=model,
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
        episodic_loss = []
        running_loss = 0
        solution_found = False
        max_iter = int(config["log"]["iterations"])
        for i in range(max_iter):

            # randomly generate a target
            l1 = np.random.randint(-3,3,2)
            l2 = np.random.randint(-3,3,2)
            target = np.random.randint(-5, 5, 2)

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
            running_loss += loss

            print("\rProcessed iteration {} of {}".format(i, max_iter), end="")
            if i % int(config["log"]["average_every"]) == 0 and i != 0:
                running_loss = running_loss / int(config["log"]["average_every"])
                print(",Evaluating at iteration: {}, Total Running Loss {}, gamma: {}, lr: {}".format(i, running_loss, estimator.get_perturb_scale(), optimizer.get_learning_rate()))
                running_loss = 0

            # book keeping stuff
            episodic_loss.append(loss)

            # step optimizer
            optimizer.step_t()
            estimator.step_t()

        # save the final model
        torch.save(model.net.cpu(), folder_name+"/final_model")

        # Compute the running mean
        N = int(config["log"]["average_every"])
        selected = rolling_mean(episodic_loss, N).tolist()

        # Combine all runs and add them to the strategy dataframe
        df = pd.DataFrame({"Iteration": np.arange(len(selected)), "Episodic Loss": selected})
        df["run"] = k
        df["strategy"] = "RNN"
        episodic_total_rewards_strategy = episodic_total_rewards_strategy.append(df)

    # Plot the learning curves here
    episodic_total_rewards_strategy.to_pickle(folder_name+"/learning_curve_df")
    plot_learning_curve(folder_name+"/_Episodic_total_loss.pdf", "Reacher", episodic_total_rewards_strategy, "Episodic Loss")

    # Log all the config parameters
    file = open(folder_name+"/model_config.txt", "w")
    file.write(str(config))
    file.close()