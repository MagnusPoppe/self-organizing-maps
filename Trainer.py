import numpy as np

from configuration import Configuration
from decorators import timer
from network import Network1D
import calculations as calc

class Trainer():

    @timer("Initialization")
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
            for multiply in range(self.config.multiplier):
                for input in self.network.inputs:
                    weights += [[np.random.uniform(*self.config.random_range) for feature in input]]
            return weights

        self.config = configuration
        self.network = Network1D(configuration)
        self.weights = initialize()

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
        for lattice_dist in range( -(len(self.weights)//2), (len(self.weights)//2)+1  ):

            # Checking of the node is included in the neighbourhood:
            hood = calc.topological_neighbourhood(lattice_dist, sigma)

            if hood > 0:
                wgt = (bmu + lattice_dist) % len(self.weights)
                for feature in range(len(input)):
                    delta = calc.weight_delta(self.weights[wgt][feature], learning_rate, input[feature], hood)
                    self.weights[wgt][feature] = delta
