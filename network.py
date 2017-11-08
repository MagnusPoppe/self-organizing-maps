import random

from configuration import Configuration


class Network1D():

    def __init__(self, configuration: Configuration):
        self.config = configuration
        self.features = self.config.features
        self.remaining = []

        # Layers
        self.inputs = [(x, y) for i, x, y in self.config.casemanager.dataset]
        self.neurons = self.initialize()

    def random_input(self):
        if not self.remaining: self.remaining = list(range(0, self.config.nodes-1))
        index = random.choice(self.remaining)
        # self.remaining.remove(index)
        return index, self.inputs[index]

    def get_neighbourhood(self) -> list:
        return range( -(len(self.neurons)//2), (len(self.neurons)//2)+1  )

    def initialize(self):
        """
        Initializing the weight matrix. Weights are created using the following structure:
        self.weights contains nodes up to number of outputs
            features containing one node for each input
                feature containing a random value.
        :return: the weight matrix
        """
        import numpy as np
        weights = []
        for multiply in range(self.config.multiplier):
            for input in self.inputs:
                weights += [[np.random.uniform(*self.config.random_range) for feature in input]]
        return weights
