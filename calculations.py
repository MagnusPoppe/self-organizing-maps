import numpy as np

# Decay functions
linear_decay      = lambda t, o, l: o + l * t
exponential_decay = lambda t, o, l: o * np.exp(- (t / l))

# Learning rate adjustment functions:
linear_learning_rate_adjust = lambda t : 1/t
inverse_of_time_learning_rate_adjust = lambda t, T : 1-(t/T)
power_series_learning_rate_adjust = lambda t, T : np.math.pow(0.005, (t/T))

def update_weights(weight, learning_rate, input, sigma):
    if sigma > 0:   return weight + learning_rate * sigma * (input - weight)
    else:           return weight + learning_rate * (input - weight)

def diff(vector1, vector2):
    return sum([v1 - v2 for v1, v2 in zip(vector1, vector2)])

def reduce_min(input, weights):
    """ Finds the minimum distance using euclidian distance.
        Formula for euclidian distance in one dimension:
        \sqrt{(x-y)^2} = |x-y|.
    """
    distance = [np.math.pow((input - weight), 2) for weight, input in zip(weights, input)]
    minimum = min(distance)
    return minimum, distance.index(minimum)