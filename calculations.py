
# Decay functions
import numpy as np

linear_decay = lambda t, o, l: o + l * t
exponential_decay = lambda t, o, l: o * np.exp(- (t / l))

# Learning rate adjustment functions:
linear_learning_rate_adjust = lambda t : 1/t
inverse_of_time_learning_rate_adjust = lambda t, T : 1-(t/T)
power_series_learning_rate_adjust = lambda t, T : np.math.pow(0.005, (t/T))

def diff(number, vector):
    # TODO: Er dette vektor eller enkelt input?
    return sum( v - number for v in vector )
    # ? return vector - number

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