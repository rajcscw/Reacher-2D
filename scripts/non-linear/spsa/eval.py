import torch.nn as nn
from components.models import SamplingStrategy
from components.models import PyTorchModel
from components.loss2 import NonLinearKinematics
from torch.autograd import Variable
import yaml
import os
import torch
import os


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

# set up the network architecture

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
net = torch.load(dir_path + "/saved_models/final_model")
model = PyTorchModel(net=net, config=config,  device=device, strategy=SamplingStrategy.BOTTOM_UP)

# objective
objective = NonLinearKinematics(model=model,
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
