import multiprocessing
from time import sleep

import numpy as np

# Decay functions
import sys

linear_decay                            = lambda t, o, l: o + l * t
exponential_decay                       = lambda t, o, l: o * np.exp(- (t / l) )

# Learning rate adjustment functions:
linear_learning_rate_adjust             = lambda t : 1/t
inverse_of_time_learning_rate_adjust    = lambda t, T : 1-(t/T)
power_series_learning_rate_adjust       = lambda t, T : np.power(0.005, (t/T))


def topological_neighbourhood(latteral_distance, sigma):
    return np.exp( -np.power(latteral_distance, 2) /  (2 * np.power(sigma, 2)) )

def weight_delta(weight, learning_rate, input, neighbourhood):
    return weight + (learning_rate * neighbourhood * (input - weight))

def bmu(input, nodes):
    i, v = shortest_distance((input, nodes))
    return i

def shortest_distance(io:tuple):
    input, nodes = io
    data = []
    for weights in nodes:
        data.append(0)
        # if not isinstance(weights[0], np.ndarray): weights = [weights]
        for wgt in weights:
            for feature, weight in zip(input, wgt):
                data[-1] = data[-1] + np.power(feature - weight, 2)
        data[-1] = np.sqrt(data[-1])
    minimum = min(data)
    return data.index(minimum), minimum

def parallel_bmu(input, nodes):
    # Setup:
    result = [None]
    processes = multiprocessing.cpu_count()
    tasks_per_proc = int(len(nodes) / processes)
    tasks = [(input, nodes[i*tasks_per_proc:(i+1)*tasks_per_proc]) for i in range(processes)]

    # Spawning processes:
    pool = multiprocessing.Pool(processes=processes-1)
    pool.map_async(
        func=shortest_distance,
        iterable=tasks[:-1],
        callback=callback(result, tasks_per_proc),
        error_callback=lambda x: print(x))

    i, v = shortest_distance(tasks[-1])

    # Waiting for sync
    while result[0] is None: pass
    pool.terminate()
    ii, vv = result[0]

    return i if v < vv else ii

def callback(result, no_tasks):
    def gather_bmu(results):
        best_value = sys.maxsize
        for i, res in enumerate(results):
            index, value = res
            if value < best_value: best_index, best_value = index + (no_tasks*i), value
        result[0] = best_index, best_value # THIS VALUE IS SENT TO THE MAIN PROCESS.
    return gather_bmu