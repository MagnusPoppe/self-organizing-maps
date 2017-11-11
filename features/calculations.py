import numpy as np

linear_decay                         = lambda t, o, l, T: o + l * t
exponential_decay                    = lambda t, o, l, T: o * np.exp(- (t / l) )
linear_learning_rate_adjust          = lambda t, o, l, T: 1/t
inverse_of_time_learning_rate_adjust = lambda t, o, l, T: 1-(t/T)
power_series_learning_rate_adjust    = lambda t, o, l, T: np.power(0.005, (t/l))
topological_neighbourhood            = lambda s, sigma: np.exp(-(s**2)/(2 * (sigma**2)))