import numpy as np

from configuration import Configuration
from decorators import timer, average_runtime
from network import Network1D
import calculations as calc

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
            # TODO: FIX LEARNING RATE...
            learning_rate = calc.exponential_decay(epoch, self.config.learning_rate, self.config.learning_rate_decay)
            sigma = calc.exponential_decay(epoch, self.config.initial_neighbourhood, self.config.neighbourhood_decay)

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

    # @average_runtime(key="Organize map")
    def organize_map(self, input, case, sigma, learning_rate):
        # Finding the BMU
        distance, feature = calc.reduce_min(input, self.weights[case])

        # Finding neighbourhood size TODO: Kan denne flyttes inn i løkken under?:
        # Adjusting the neighbourhood:
        for wgt in range(len(self.weights)):
            x = abs(wgt-case)
            neighbourhood = calc.topological_neighbourhood(x, sigma)
            if neighbourhood > 0:
                delta = calc.weight_delta(self.weights[wgt][feature], learning_rate, input[feature], neighbourhood)
                self.weights[wgt][feature] = delta