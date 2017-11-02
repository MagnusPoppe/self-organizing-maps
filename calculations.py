import numpy as np

# Decay functions
linear_decay      = lambda t, o, l: o + l * t
exponential_decay = lambda t, o, l: o * np.exp(- (t / l))

# Learning rate adjustment functions:
linear_learning_rate_adjust = lambda t : 1/t
inverse_of_time_learning_rate_adjust = lambda t, T : 1-(t/T)
power_series_learning_rate_adjust = lambda t, T : np.math.pow(0.005, (t/T))

euclidian_distance = lambda x, y : np.power((x - y), 2)

mean = lambda x: sum(x)/len(x)

def topological_neighbourhood(winner, weight, sigma):
    return np.exp( -np.power(weight - winner, 2) / ( 2 * np.power(sigma, 2) ) )

def weight_delta(weight, learning_rate, input, neighbourhood):
    if neighbourhood > 0.0:   return weight + learning_rate * neighbourhood * (input - weight)
    else:                   return weight + learning_rate * (input - weight)

def diff(vector1, vector2):
    return sum([v1 - v2 for v1, v2 in zip(vector1, vector2)])

def reduce_min(input, weights):
    """ Finds the minimum distance using euclidian distance.
        Formula for euclidian distance in one dimension:
        \sqrt{(x-y)^2} = |x-y|.
    """
    distance = [euclidian_distance(input, weight) for weight, input in zip(weights, input)]
    minimum = min(distance)
    return minimum, distance.index(minimum)