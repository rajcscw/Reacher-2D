import torch.nn as nn
from components.models import SamplingStrategy
from components.models import PyTorchModel
from components.loss import InverseKinematic
from torch.autograd import Variable
import yaml
import os
import torch


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
net = torch.load("saved_models/final_model")
model = PyTorchModel(net=net, config=config,  device=device, strategy=SamplingStrategy.BOTTOM_UP)

# objective
objective = InverseKinematic(model=model,
                             config=config,
                             device=device
                             )

# run for few iterations and plot the results
save_folder = "eval_results"
for i in range(50):
    # loss
    loss = objective(None, True, save_folder+"/eval_{}.pdf".format(i+1))
