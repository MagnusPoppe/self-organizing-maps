import numpy as np

from configuration import Configuration
from decorators import timer
from network import Network1D
from calculations import *

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
                weights += [[np.random.uniform(*self.config.random_range) for feature in input]]
            return weights

        self.config = configuration
        self.network = Network1D(configuration)
        self.weights = initialize()

        if self.config.visuals:

            from live_graph import LiveGraph
            self.graph = LiveGraph("Travelling salesman problem", "x", "y")

    @timer("Training phase: ")
    def train(self):
        # Running random cases for a number of epochs:
        for epoch in range(1, self.config.epochs):

            # Adjusting parameters:
            learning_rate = linear_learning_rate_adjust(epoch/2)
            sigma = linear_decay(epoch, self.config.decay_sigma, self.config.decay_lambda)

            for i in range(len(self.network.inputs)):
                # Getting input for this run.
                case, input = self.network.random_input()

                # Running the self organizing map algorithm:
                self.organize_map(input, case, sigma, learning_rate)

            # Displaying visuals if system is configured to do so.
            if epoch % self.config.visuals_refresh_rate == 0:
                print("Current state: \n\tSigma: %f \n\tLearning rate: %f\n" %(sigma, learning_rate))
                if self.config.visuals:
                    self.graph.update(actuals=self.weights + [self.weights[0]], targets=self.network.inputs)

    def organize_map(self, input, case, sigma, learning_rate):
        # Finding the BMU
        distance, i = reduce_min(input, self.weights[case])

        # Finding neighbourhood size: TODO: Neighbourhood function is probably wrong.
        mod = self.config.nodes * (sigma / 2)

        # Adjusting the neighbourhood:
        for degree in range(int(case - mod), int(case + mod)+1):
            neighbour = case + degree if case + degree < self.config.nodes else abs(case - (case + degree)) - 1
            for x in range(len(input)):
                self.weights[neighbour][x] = update_weights(self.weights[neighbour][x], learning_rate, input[x], sigma)