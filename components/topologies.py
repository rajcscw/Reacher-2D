
import numpy as np
import matplotlib.pyplot as plt


class ClassicReservoirTopology:
    def __init__(self, size):
        self.size = size

    def generateWeightMatrix(self, scaling=0.5):
        reservoirWeightRandom = np.random.uniform(-scaling, +scaling, self.size * self.size).reshape((self.size, self.size))
        return reservoirWeightRandom


class ClassicInputTopology:
    def __init__(self, inputSize, reservoirSize):
        self.inputSize = inputSize
        self.reservoirSize = reservoirSize

    def generateWeightMatrix(self, scaling = 0.5):
        inputWeightRandom = np.random.uniform(-scaling, +scaling, self.reservoirSize * self.inputSize).reshape((self.reservoirSize, self.inputSize))
        return inputWeightRandom


class RandomInputTopology:
    def __init__(self, inputSize, reservoirSize, inputConnectivity = 1.0):
        self.inputSize = inputSize
        self.reservoirSize = reservoirSize
        self.inputConnectivity = inputConnectivity
        self.connMatrix = None

    def generateConnectivityMatrix(self):
        connectivity = np.zeros((self.reservoirSize, self.inputSize))
        for i in range(self.reservoirSize):
            indices = np.random.choice(self.inputSize, size=int(np.ceil(self.inputConnectivity * self.inputSize)), replace=False)
            connectivity[i, indices] = 1.0
        return connectivity

    def generateWeightMatrix(self, scaling=0.5):
        # Multiply with randomly generated matrix with connected matrix
        random = np.random.uniform(-scaling, +scaling, self.reservoirSize * self.inputSize).reshape((self.reservoirSize, self.inputSize))
        self.connMatrix = self.generateConnectivityMatrix()
        weight = random * self.connMatrix
        return weight


class RandomReservoirTopology:
    def __init__(self, size, connectivity=1.0):
        self.size = size
        self.connectivity = connectivity
        self.connectivityMatrix = np.zeros((self.size, self.size))

        for i in range(self.size):
            indices = np.random.choice(self.size, size=int(np.ceil(self.connectivity * self.size)), replace=False)
            self.connectivityMatrix[i, indices] = 1.0

    def generateConnectivityMatrix(self):
        return self.connectivityMatrix

    def generateWeightMatrix(self, scaling=0.5):
        # Multiply with randomly generated matrix with connected matrix
        random = np.random.uniform(-scaling, +scaling, self.size * self.size).reshape((self.size, self.size))
        weight = random * self.connectivityMatrix
        return weight







