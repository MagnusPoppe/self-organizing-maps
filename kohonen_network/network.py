import multiprocessing
import random
import sys

import numpy as np

from features import calculations as calc
from kohonen_network.configuration import Configuration


class Network():
    def __init__(self, configuration: Configuration):
        self.config = configuration
        self.features = self.config.features
        self.remaining = []

        # Layers
        self.inputs = self.generate_input_vectors(self.config.casemanager.dataset)
        self.neurons = self.initialize()

    def random_input(self):
        if not self.remaining: self.remaining = list(range(0, len(self.inputs)))
        index = random.choice(self.remaining)
        self.remaining.remove(index)
        return index, self.inputs[index]

    def generate_input_vectors(self, dataset): pass
    def initialize(self): pass

class Network1D(Network):

    def __init__(self, configuration: Configuration):
        super().__init__(configuration)

    def generate_input_vectors(self, dataset):
        return [(x, y) for i, x, y in dataset]

    def neighbourhood(self, sigma, bmu, size ):
        out = [None]*len(self.neurons[0])
        mod = 0 if len(self.neurons[0]) % 2 == 0 else 1
        for lattice_dist in range( -(len(self.neurons[0])//2), (len(self.neurons[0])//2) + mod):
            index = (bmu + lattice_dist) % len(self.neurons[0])
            out[index] =calc.topological_neighbourhood(lattice_dist, sigma)
        out = np.array(out, ndmin=2)
        out = np.reshape(out, (1, len(self.neurons[0])))
        return out

    def initialize(self):
        """
        Initializing the weight matrix. Weights are created using the following structure:
        self.weights contains nodes up to number of outputs
            features containing one node for each input
                feature containing a random value.
        :return: the weight matrix
        """
        import numpy as np
        output = []
        for i in range(self.config.features):
            output += [np.random.uniform(*self.config.random_range, size=len(self.inputs) * self.config.multiplier)]
        return output



class Network2D(Network):

    def __init__(self, configuration: Configuration):
        self.grid_size = configuration.grid
        self.nodes = self.grid_size[0]*self.grid_size[1]
        super().__init__(configuration)

        # The winner list is a counter for total cases won per neuron
        self.winnerlist = [[] for x in range(self.grid_size[0]*self.grid_size[1])]
        self.threshold = 1.0e+10

    def generate_input_vectors(self, dataset):
        return dataset # No preprocessing needed.

    def neighbourhood(self, sigma, bmu, grid):
        # MANHATTEN DISTANCE BETWEEN NODES
        bmux, bmuy = bmu % np.sqrt(grid), bmu // np.sqrt(grid)
        out = np.array([], ndmin=2)
        for i in range(grid):
            y, x = int(i // np.sqrt(grid)), int(i % np.sqrt(grid))
            out = np.append(out, calc.topological_neighbourhood(abs(bmux - x) + abs(bmuy - y), sigma))
            out = np.reshape(out, (len(out), 1))

        return out


    def initialize(self):
        import numpy as np
        output = []
        for i in range(self.grid_size[0]*self.grid_size[1]):
            output += [ np.random.uniform(*self.config.random_range, size=(len(self.inputs[0]) * self.config.multiplier))]
        return output

    def drawable(self):
        matrix = []
        for i in range(len(self.winnerlist)):

            # Creating the histogram:
            histogram = [0]*10
            for entry in self.winnerlist[i]:
                histogram[ self.config.casemanager.labels[entry] ] += 1

            # Finding the correct value for (x,y)
            if i % np.sqrt(len(self.winnerlist)) == 0: matrix.append([-1]*int(np.sqrt(len(self.winnerlist))))
            y, x = int(i // np.sqrt(len(self.winnerlist))), int(i % np.sqrt(len(self.winnerlist)))
            if any(e > 0 for e in histogram):
                matrix[y][x] = histogram.index(max(histogram))

        return matrix
    #
    # def parallel_bmu(self, input, nodes):
    #     # Setup:
    #     result = [None]
    #     processes = int(multiprocessing.cpu_count()/2)
    #     tasks_per_proc = int(len(nodes) / processes)
    #     tasks = [(input, nodes[i*tasks_per_proc:(i+1)*tasks_per_proc]) for i in range(processes)]
    #
    #     # Spawning processes:
    #     pool = multiprocessing.Pool(processes=processes-1)
    #     pool.map_async(
    #         func=self.shortest_distance,
    #         iterable=tasks[:-1],
    #         callback=self.callback(result, tasks_per_proc),
    #         error_callback=lambda x: print(x))
    #
    #     i, v = self.shortest_distance(tasks[-1])
    #
    #     # Waiting for sync
    #     while result[0] is None: pass
    #     pool.terminate()
    #
    #     ii, vv = result[0]
    #     return i if v < vv else ii
    #
    # def callback(self, result, no_tasks):
    #     def gather_bmu(results):
    #         best_value, best_index= sys.maxsize, -1
    #         for i, res in enumerate(results):
    #             index, value = res
    #             if value < best_value: best_index, best_value = index, value
    #         result[0] = best_index, best_value # THIS VALUE IS SENT TO THE MAIN PROCESS.
    #
    #     return gather_bmu
