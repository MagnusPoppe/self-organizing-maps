import random
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
        output = []
        if self.config.placement_type == "random":
            for i in range(self.config.features):
                output += [np.random.uniform(*self.config.random_range, size=len(self.inputs)*self.config.multiplier)]
        elif self.config.placement_type == "circle":
            origoX = origoY = 0.5
            radius = 0.25
            output = [[],[]]
            for angle in range(0,360):
                output[0] += [ origoX + radius * np.cos(angle * ( np.pi / 180 )) ]
                output[1] += [ origoY + radius * np.sin(angle * ( np.pi / 180 )) ]
        elif self.config.placement_type == "line":
            half = (len(self.inputs) * self.config.multiplier)/2
            output = [
                [(1 / half) * x for x in list(range(int(half)+1)) + list(reversed(range(int(half))))],
                [0.5]*len(self.inputs) * self.config.multiplier
            ]
        elif self.config.placement_type == "box":
            sqrt = (len(self.inputs) * self.config.multiplier) / 4
            output = [[],[]]
            # top:
            output [0] += [(1 / sqrt) * x for x in list(range(int(sqrt)))]
            output [1] += [0] * int(sqrt)
            # right:
            output [0] += [1] * int(sqrt)
            output [1] += [(1 / sqrt) * x for x in list(range(int(sqrt)))]
            # bottom:
            output [0] += [(1 / sqrt) * x for x in list(reversed(range(int(sqrt))))]
            output [1] += [1] * int(sqrt)
            # left:
            output [0] += [0] * int(sqrt)
            output [1] += [(1 / sqrt) * x for x in list(reversed(range(int(sqrt))))]
        else: raise ValueError("Parameter placement type did not match any known placement type.")
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

    def toMatrix(self, vector):
        n = len(vector)
        sq = int(np.sqrt(n))
        matrix = [[-1]*sq for i in range(sq)]
        for i in range(n):
            # y, x = i // sq, int(i % sq)
            matrix[i // sq][i % sq] = vector[i]
        return matrix

    def get_value_mapping(self):
        # Creating the histogram:
        output = []
        for i in range(len(self.winnerlist)):
            histogram = [0] * 10
            for entry in self.winnerlist[i]:
                histogram[self.config.casemanager.labels[entry]] += 1
            output += [histogram.index(max(histogram)) if any(e > 0 for e in histogram) else -1]
        return output