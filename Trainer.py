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
            from live_graph import LiveGraph, LiveGrid
            gclass = LiveGraph if self.config.dataset != "mnist" else LiveGrid
            self.graph  = gclass(self.config.title, "x", "y")
        print("Setup complete!\n")

    @timer("Training")
    def train(self):
        print("Starting training")
        # Running random cases for a number of epochs:
        for epoch in range(1, self.config.epochs):

            # Adjusting parameters:
            learning_rate = calc.exponential_decay(epoch, self.config.learning_rate, self.config.learning_rate_decay)
            sigma = calc.exponential_decay(epoch, self.config.initial_neighbourhood, self.config.neighbourhood_decay)

            # Getting input for this run.
            case, input = self.network.random_input()

            # Running the self organizing map algorithm:
            self.organize_map(input, case, sigma, learning_rate)

            # Displaying visuals if system is configured to do so.
            if self.config.visuals and epoch % self.config.visuals_refresh_rate == 0:
                if self.config.dataset != "mnist":
                    self.graph.update(actuals=self.weights + [self.weights[0]], targets=self.network.inputs)
                else:
                    self.graph.update(self.network.neurons)

            if epoch % self.config.printout_rate == 0:
                print("Current state: \n\tSigma:         %f \n\tLearning rate: %f\n" % (sigma, learning_rate))

    @timer("Organize map")
    def organize_map(self, input, case, sigma, learning_rate):
        # Finding the BMU (Best matching unit)
        bmu = self.network.bmu(input, self.weights)

        # Tracking the winners
        if isinstance(self.network, Network2D):
            self.network.winnerlist[bmu] += [case]

        self.network.update_weights(self.weights, bmu, input, learning_rate, sigma)
