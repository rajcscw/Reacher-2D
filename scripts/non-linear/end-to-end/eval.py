import torch.nn as nn
from components.models import SamplingStrategy
from components.models import PyTorchModel
from components.loss2 import NonLinearFullKinematics
from torch.autograd import Variable
import yaml
import os
import torch
import os


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

# load the config
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path+"/config.yml", 'r') as ymlfile:
    config = yaml.load(ymlfile)
print(config)

# check the device (CPU vs GPU)
use_gpu = False
use_multi_gpu = False
device = "cuda:0" if torch.cuda.is_available() and use_gpu else "cpu"

# load the model
net = torch.load(dir_path + "/saved_models/final_model_with_target_position_run_4_last_reward_-7.072589753346181")
config["model"]["diff"] = False
model = PyTorchModel(net=net, config=config,  device=device, strategy=SamplingStrategy.BOTTOM_UP)

# objective
objective = NonLinearFullKinematics(model=model,
                                    config=config,
                                    device=device
                                   )

# run for few iterations and plot the results
save_folder = dir_path + "/eval_results"
total_error = 0
n_evals = 50
for i in range(n_evals):
    # loss
    total_error += objective(None, True, save_folder+"/eval_{}.pdf".format(i+1))

average_error = total_error / n_evals
print("Average Error over {} runs is {}".format(n_evals, average_error))
