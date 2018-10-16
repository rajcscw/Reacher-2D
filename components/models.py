import numpy as np
import torch
from functools import reduce
from enum import Enum
from collections import OrderedDict


class SamplingStrategy(Enum):
    RANDOM = 1
    BOTTOM_UP = 2
    TOP_DOWN = 3
    ALL = 4


class PyTorchModel:
    def __init__(self, net, config, device, strategy=SamplingStrategy.RANDOM):
        """
        :param net: a pytorch net module
        """
        # net is the current optimized network
        self.net = net

        # send the model to that device
        self.device = device
        self.net = self.__to_device(self.net)

        print("The current device is now set to {}".format(self.device))

        # parameters
        self.parameters = list(torch.nn.Module.named_parameters(self.net))

        # strategy to sample layer
        self.sampling_strategy = strategy

        # get layer wise parameters
        self.layer_wise_parameters = self.__get_layer_params__()
        self.total_layers = len(self.layer_wise_parameters)
        self.all_layers = list(self.layer_wise_parameters.keys())
        self.layer_dimensions = {}

        # set counters
        self.__set__counters()

    def __to_device(self, net):
        if self.device == "cpu":
            return net
        else:
            return net.cuda(self.device)

    def __set__counters(self):
        if self.sampling_strategy == SamplingStrategy.BOTTOM_UP:
            self.last = -1
        else:
            self.last = len(self.layer_wise_parameters)

    @staticmethod
    # TBD: may have to take care of cuda tensors
    def __str__to__type(str):
        if str == "torch.float32":
            return torch.float32
        elif str == "torch.float64":
            return torch.float64

    def reset_cache(self, config):
        self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=float(config["optimizer"]["lr"]))

    def forward(self, input):

        # may be send it to GPU again (TBD)
        self.net = self.__to_device(self.net)

        output = self.net(input)
        return output

    def get_param_value(self, name):
        """
        recursively get the value of the parameter
        :param name: parameter name and its data type
        :return: return the parameter value as a numpy array
        """

        refs = name.split(".")
        last = self.net
        for ref in refs:
            last = getattr(last, ref)

        return torch.Tensor(last.cpu().data.numpy()), last.data.dtype

    def set_param_value(self, name, type, value):
        """
        :param name: parameter name
        :param value: value of the parameter
        """

        obj = self.net
        parts = name.split(".")
        for attr in parts[:-1]:
            obj = getattr(obj, attr)
        type = self.__str__to__type(str=type)
        setattr(obj, parts[-1], value.type(type))

    def __get_layer_params__(self):
        """ gets the parameters grouped by layers
        :return: returns a dict of dict containing shape and type of parameters
        """

        layer_params = OrderedDict()

        for param in self.parameters:

            # split into layer name and parameter name
            param_name = param[0]
            refs = param_name.split(".")
            layer_name = refs[0]

            # get parameter details
            data, type = self.get_param_value(param[0])

            type = str(type)
            shape = data.shape

            # store it in the layer information
            if layer_name in layer_params.keys():
                layer_params[layer_name][param_name] = (shape, type)
            else:
                layer_params[layer_name] = dict()
                layer_params[layer_name][param_name] = (shape, type)

        return layer_params

    def get_random_layer(self):
        """
        :return: returns a random layer name and its current value
        """
        # random parameter layer and its value
        choice = np.random.choice(np.arange(len(self.all_layers)))
        random_layer_name = self.all_layers[choice]
        random_layer_value = self.get_layer_value_by_name(random_layer_name)

        return random_layer_name, random_layer_value

    def sample_layer(self):
        """
        :return: returns a layer name and its current value based on the sampling strategy
        """
        if self.sampling_strategy == SamplingStrategy.RANDOM:
            layer_name, layer_value = self.get_random_layer()
        elif self.sampling_strategy == SamplingStrategy.BOTTOM_UP: # BOTTOM-UP approach
            self.last = self.last + 1
            self.last = self.last % (self.total_layers)
            layer_name = self.all_layers[self.last]
            layer_value = self.get_layer_value(layer_name)
        elif self.sampling_strategy == SamplingStrategy.TOP_DOWN: # TOP-DOWN approach
            self.last = self.last - 1
            if self.last < 0:
                self.last = self.total_layers - 1
            layer_name = self.all_layers[self.last]
            layer_value = self.get_layer_value(layer_name)
        else: # all layers at once
            layer_name = "all"
            layer_value = self.get_layer_value(layer_name)
        return layer_name, layer_value

    def get_layer_value(self, layer_name):
        """
        :param layer_name: name of the layer (can also be "all" which gives all layers in a big humongous vector)
        :return: returns the current value of the layer
        """
        layer_value = None
        if layer_name == "all":
            for layer in self.all_layers:
                layer_v = self.get_layer_value_by_name(layer)
                # set layer dimensions for later unpacking
                self.layer_dimensions[layer] = layer_v.shape[0]
                if layer_value is None:
                    layer_value = layer_v
                else:
                    layer_value = torch.cat((layer_value, layer_v))
        else:
            layer_value = self.get_layer_value_by_name(layer_name)

        return layer_value

    def get_layer_value_by_name(self, layer_name):
        """
        :param layer_name: name of the layer
        :return: returns the current value of all layer
        's parameters in a one big vector
        """

        layer_parameters = list(self.layer_wise_parameters[layer_name])

        # pack all the parameter values
        value = None
        for param in layer_parameters:
            param_value, _ = self.get_param_value(param)
            if value is None:
                value = param_value.reshape((-1,1))
            else:
                value = torch.cat((value, param_value.reshape((-1,1))))

        return value

    def set_layer_value(self, layer_name, layer_value):
        """
        :param layer_name: also can be "all" which unpacks the value to appropriate layers
        :param layer_value: value of the layer to be set
        """
        if layer_name == "all":
            # unpack into separate values
            last = 0
            for layer_name, layer_dim in self.layer_dimensions.items():
                value = layer_value[last:last+layer_dim]
                last += layer_dim
                self.set_layer_value_by_name(layer_name, value)
        else:
            # just call the underlying method
            self.set_layer_value_by_name(layer_name, layer_value)

    def set_layer_value_by_name(self, layer_name, layer_value, value=".data"):
        """
        sets the value of individual parameters in the layer by properly unpacking and setting them one at a time
        :param layer_name: name of the layer
        :param layer_value: value of the layer in a big vector form
        """
        layer_parameters = list(self.layer_wise_parameters[layer_name])
        last = 0

        # unpack individual parameters and set its value
        for param in layer_parameters:
            param_shape, param_type = self.layer_wise_parameters[layer_name][param]
            param_size = reduce(lambda x, y: x*y, list(param_shape))
            param_value = layer_value[last:last+param_size].reshape(param_shape)
            last += param_size

            self.set_param_value(param+value, param_type, param_value)

    def step_layer(self, layer_name, gradients):

        # reset all gradients
        self.optimizer.zero_grad()

        self.set_layer_value_by_name(layer_name, gradients, ".grad")

        # take a step
        self.optimizer.step()

