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
            learning_rate = calc.power_series_learning_rate_adjust(epoch, self.config.epochs)
            sigma = calc.exponential_decay(epoch, self.config.initial_neighbourhood, self.config.neighbourhood_decay)

            # Getting input for this run.
            case, input = self.network.random_input()

            # Running the self organizing map algorithm:
            self.organize_map(input, case, sigma, learning_rate)

            # Displaying visuals if system is configured to do so.
            if epoch % self.config.visuals_refresh_rate == 0:
                print("Current state: \n\tSigma:         %f \n\tLearning rate: %f\n" %(sigma, learning_rate))
                if self.config.visuals:
                    self.graph.update(actuals=self.weights + [self.weights[0]], targets=self.network.inputs)

    @average_runtime(key="Organize map")
    def organize_map(self, input, case, sigma, learning_rate):
        # Finding the BMU
        feature = calc.reduce_min(input, self.weights[case])

        for offset in range( -(len(self.weights)//2), (len(self.weights)//2)+1  ):
            # Checking of the node is included in the neighbourhood:
            neighbourhood = calc.topological_neighbourhood(offset, sigma)
            wgt = (case + offset) % len(self.weights)
            if neighbourhood > 0:
                i = wgt
                delta = calc.weight_delta(self.weights[case][feature], learning_rate, input[feature], neighbourhood)
                self.weights[i][feature] = delta
