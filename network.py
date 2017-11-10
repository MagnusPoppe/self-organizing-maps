import random

import multiprocessing
import numpy as np
import sys

import calculations as calc
from configuration import Configuration

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

    def get_neighbourhood(self):
        return range( -(len(self.neurons)//2), (len(self.neurons)//2)+1  )

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
        for i in range(len(self.inputs) * self.config.multiplier):
            output += [ [np.random.uniform(*self.config.random_range, size=len(self.inputs[0]))] ]
        return output

    def bmu(self, input, nodes):
        data = []
        for weights in nodes:
            data.append(0)
            for feature, weight in zip(input, weights[0]):
                data[-1] = data[-1] + np.power(feature - weight, 2)
            data[-1] = np.sqrt(data[-1])
        minimum = min(data)
        return data.index(minimum)#, minimum


    def update_weights(self, weights, bmu, input, learning_rate, sigma):
        for lattice_dist in self.get_neighbourhood():

            # Checking of the node is included in the neighbourhood:
            hood = calc.topological_neighbourhood(lattice_dist, sigma)

            if hood > 0:  # matrix multiply
                wgt = (bmu + lattice_dist) % len(weights)
                for y in range(len(weights[0])):
                    for feature in range(len(weights[0][0])):
                        delta = calc.weight_delta(weights[wgt][y][feature], learning_rate, input[feature], hood)
                        weights[wgt][y][feature] = delta

class Network2D(Network):

    def __init__(self, configuration: Configuration):
        self.grid_size = configuration.grid
        self.nodes = self.grid_size[0]*self.grid_size[1]
        super().__init__(configuration)

        # The winner list is a counter for total cases won per neuron
        self.winnerlist = [[] for x in range(self.grid_size[0]*self.grid_size[1])]

    def generate_input_vectors(self, dataset):
        return dataset # No preprocessing needed.

    def get_neighbourhood(self):
        return range( -(len(self.neurons)//2), (len(self.neurons)//2)+1  )

    def initialize(self):
        import numpy as np
        output = []
        for i in range(len(self.inputs[0]) * self.config.multiplier):
            output += [ np.random.uniform(*self.config.random_range, size=self.grid_size)]
        return output

    def bmu(self, input, nodes):
        i, v = self.shortest_distance((input, nodes))
        return i

    def shortest_distance(self, io: tuple):
        input, nodes = io
        data = [0] * (len(nodes[0]) * len(nodes[0][0]))
        for weights in nodes:
            for y, wgt in enumerate(weights):
                ymod = y * len(wgt)
                for x, tup in enumerate(list(zip(input, wgt))):
                    feature, weight = tup
                    data[x + ymod] += np.power(feature - weight, 2)
        for i in range(len(data)):
            data[i] = np.sqrt(data[i])
        minimum = min(data)
        return data.index(minimum), minimum

    def update_weights(self, weights, bmu, input, learning_rate, sigma):
        # Setup:
        grid = self.grid_size[1] * self.grid_size[0]
        neighbourhood = np.ndarray(shape=self.grid_size)
        bmux, bmuy = bmu % np.sqrt(grid), bmu // np.sqrt(grid)

        # MANHATTEN DISTANCE BETWEEN NODES
        for i in range(grid):
            y, x = int(i // np.sqrt(grid)), int(i % np.sqrt(grid))
            neighbourhood[y] += [calc.topological_neighbourhood(abs(bmux - x) + abs(bmuy - y), sigma)]

        # UPDATE THE WEIGHTS.
        for z in range(len(weights)):
            for y in range(len(neighbourhood)):
                for x in range(len(neighbourhood[y])):
                    delta = calc.weight_delta(weights[z][y][x], learning_rate, input[z], neighbourhood[y][x])
                    weights[z][y][x] = delta


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
