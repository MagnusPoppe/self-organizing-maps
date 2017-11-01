import random
import numpy as np

from configuration import Configuration
from decorators import timer
from network import Network1D




class Trainer():

    @timer("Initialization phase: ")
    def __init__(self, configuration: Configuration):
        def initialize():
            """
            Initializing the weight matrix. Weights are created using the following structure:
            self.weights contains nodes up to number of outputs
                features containing one node for each input
                    feature containing a random value.
            :return: the weight matrix
            """
            weights = []
            for input in self.network.inputs:
                weights += [np.random.uniform(self.config.random_range) for feature in input]
            return weights

        self.config = configuration
        self.network = Network1D(configuration)
        self.weights = initialize()



    @timer("Training phase: ")
    def train(self):
        def reduce_min(feature, feature_index):
            """ Finds the minimum distance using euclidian distance.
                Formula for euclidian distance in one dimension:
                \sqrt{(x-y)^2} = |x-y|.
            """
            distances = []
            for weights in self.weights:
                weight = weights[feature_index]
                distances += [np.math.sqrt(np.math.pow( (feature - weight) , 2))]
            minimum = min(distances)
            return minimum, distances.index(minimum)


        for epoch in range(self.config.epochs):
            # TODO: Maybe refactor. Network.random_case if the case variable is only used here (random.choice)
            case = random.randint(0, self.config.output_nodes-1)
            input = self.network.inputs[case]

            for i in range(len(input)):
                minimum, index = reduce_min(input[i], i)


