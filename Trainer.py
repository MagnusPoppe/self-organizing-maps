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
        degree_of_neighbourhood = self.config.decay_sigma
        learning_rate = self.config.learning_rate
        for epoch in range(1, self.config.epochs):
            # TODO: Maybe refactor. Network.random_case if the case variable is only used here (random.choice)
            case = random.randint(0, self.config.output_nodes-1)
            input = self.network.inputs[case]
            for i in range(len(input)):
                minimum, index = reduce_min(input[i], i, self.weights)

                # Adjusting the neighbourhood:
                neighbourhood = (int(-degree_of_neighbourhood / 2), int(degree_of_neighbourhood / 2) + 1)
                weight = self.weights[index][i]
                for degree in range(*neighbourhood):
                    neighbour = index + degree
                    adjusted_weight = weight + ( learning_rate * degree * diff(weight, input) ) # TODO: Er dette vektor eller enkelt input?
                    self.weights[neighbour][i] = adjusted_weight

                # Adjusting parameters:
                degree_of_neighbourhood = linear_decay(epoch, degree_of_neighbourhood, self.config.decay_lambda)
                learning_rate = linear_learning_rate_adjust(epoch)

# Decay functions
linear_decay = lambda t, o, l: o + l * t
exponential_decay = lambda t, o, l: o * np.exp(-t / l)

# Learning rate adjustment functions:
linear_learning_rate_adjust = lambda t : 1/t
inverse_of_time_learning_rate_adjust = lambda t, T : 1-(t/T)
power_series_learning_rate_adjust = lambda t, T : np.math.pow(0.005, (t/T))

def diff(number, vector):
    return sum( v - number for v in vector)
    # return vector - number

def reduce_min(feature, feature_index, weights):
    """ Finds the minimum distance using euclidian distance.
        Formula for euclidian distance in one dimension:
        \sqrt{(x-y)^2} = |x-y|.
    """
    distances = []
    for features in weights:
        weight = features[feature_index]
        distances += [np.math.sqrt(np.math.pow((feature - weight), 2))]
    minimum = min(distances)
    return minimum, distances.index(minimum)

