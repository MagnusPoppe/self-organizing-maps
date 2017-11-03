import numpy as np

from decorators import infinity_handler

# Decay functions
linear_decay                            = lambda t, o, l: o + l * t
exponential_decay                       = lambda t, o, l: o * np.exp(- (t / l) ) + 0.1

# Learning rate adjustment functions:
linear_learning_rate_adjust             = lambda t : 1/t
inverse_of_time_learning_rate_adjust    = lambda t, T : 1-(t/T)
power_series_learning_rate_adjust       = lambda t, T : np.power(0.005, (t/T))

# @infinity_handler()
def euclidian_distance(x, y):
    return np.math.pow((x - y), 2)

# @infinity_handler()
def topological_neighbourhood(latteral_distance, sigma):
    return np.exp( -np.power(latteral_distance, 2) /  (2 * np.power(sigma, 2)) )

def weight_delta(weight, learning_rate, input, neighbourhood):
    return weight + ((learning_rate * neighbourhood) * (input - weight))

def reduce_min(input, weights):
    """ Finds the minimum distance using euclidian distance.
        Formula for euclidian distance in one dimension:
        \sqrt{(x-y)^2} = |x-y|.
    """
    distance = [euclidian_distance(input, weight) for weight, input in zip(weights, input)]
    minimum = min(distance)
    return minimum, distance.index(minimum)