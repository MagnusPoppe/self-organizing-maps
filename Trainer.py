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
            self.graph = LiveGraph("Travelling salesman problem", "x", "y", (0,1),(0,1))

    @timer("Training phase: ")
    def train(self):
        # Initial values. Will be adjusted during training.
        sigma = self.config.decay_sigma
        learning_rate = self.config.learning_rate

        # Running random cases for a number of epochs:
        for epoch in range(1, self.config.epochs):
            for case in range(len(self.network.inputs)):
                input = self.network.random_input()

                # Running the self organizing map algorithm:
                learning_rate, sigma = self.organize_map(epoch, input, sigma, learning_rate)

            # Displaying visuals if system is configured to do so.
            if self.config.visuals:
                self.graph.update(actuals=self.weights, targets=self.network.inputs)

        if self.config.visuals:
            import matplotlib.pyplot as plt
            plt.close("all")

    def organize_map(self, epoch, input, degree_of_neighbourhood, learning_rate):
        for i in range(len(input)):
            # Finding the BMU
            minimum, index = reduce_min(input[i], i, self.weights)

            # Adjusting the neighbourhood:
            neighbourhood = int(-(self.config.nodes * degree_of_neighbourhood/12)), int(self.config.nodes * int(degree_of_neighbourhood/12))
            weight = self.weights[index][i]
            for degree in range(*neighbourhood):
                neighbour = index + degree if index + degree < self.config.nodes else abs(index - (index + degree)) - 1
                self.weights[neighbour][i] = weight + (learning_rate * degree_of_neighbourhood * diff(weight, input))

            # Adjusting parameters:
            degree_of_neighbourhood = exponential_decay(epoch, self.config.decay_sigma, self.config.decay_lambda)
            learning_rate = linear_learning_rate_adjust(epoch)

        return learning_rate, degree_of_neighbourhood


