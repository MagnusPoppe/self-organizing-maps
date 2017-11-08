import numpy as np

from configuration import Configuration
from decorators import timer
from network import Network1D, Network2D
import calculations as calc

class Trainer():

    def __init__(self, configuration: Configuration):

        self.config = configuration
        if configuration.dataset == "mnist": self.network = Network2D(configuration)
        else:                                self.network = Network1D(configuration)
        self.weights = self.network.neurons

        if self.config.visuals:

            from live_graph import LiveGraph
            self.graph = LiveGraph(self.config.title, "x", "y")

    @timer("Training")
    def train(self):
        # Running random cases for a number of epochs:
        for epoch in range(1, self.config.epochs):

            # Adjusting parameters:
            learning_rate = calc.exponential_decay(epoch, self.config.learning_rate, self.config.learning_rate_decay)
            sigma = calc.exponential_decay(epoch, self.config.initial_neighbourhood, self.config.neighbourhood_decay)

            # Getting input for this run.
            case, input = self.network.random_input()

            # Running the self organizing map algorithm:
            self.organize_map(input, sigma, learning_rate)

            # Displaying visuals if system is configured to do so.
            if self.config.visuals and epoch % self.config.visuals_refresh_rate == 0:
                self.graph.update(actuals=self.weights + [self.weights[0]], targets=self.network.inputs)
            if epoch % self.config.printout_rate == 0:
                print("Current state: \n\tSigma:         %f \n\tLearning rate: %f\n" % (sigma, learning_rate))

    @timer("Organize map")
    def organize_map(self, input, sigma, learning_rate):
        # Finding the BMU (Best matching unit)
        # feature = calc.reduce_min(input, self.weights[case])
        bmu = calc.bmu(input, self.weights)
        for lattice_dist in self.network.get_neighbourhood():

            # Checking of the node is included in the neighbourhood:
            hood = calc.topological_neighbourhood(lattice_dist, sigma)

            if hood > 0: # matrix multiply
                wgt = (bmu + lattice_dist) % len(self.weights)
                for feature in range(len(input)):
                    delta = calc.weight_delta(self.weights[wgt][feature], learning_rate, input[feature], hood)
                    self.weights[wgt][feature] = delta
