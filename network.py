import random

import numpy as np

from configuration import Configuration
from node import TSPNode


class Network1D():

    def __init__(self, configuration: Configuration):
        self.config = configuration
        self.features = self.config.features
        self.remaining = []

        # Layers
        self.inputs = [(x, y) for i, x, y in self.config.casemanager.dataset]
        self.outputs = []

        # Building network:
        randoms = np.random.uniform(*self.config.random_range, size=self.config.nodes)
        for i, value in enumerate(randoms):
            # Setting up nodes
            prev = self.outputs[i - 1].edge_next if i != 0 else None
            self.outputs += [TSPNode(value, edge_prev=prev)]

        # Setting the last edge after loop though to connect the circle.
        self.outputs[0].set_egde_prev(self.outputs[-1].edge_next)

    def random_input(self):
        if not self.remaining: self.remaining = list(range(0, self.config.nodes-1))
        index = random.choice(self.remaining)
        self.remaining.remove(index)
        return self.inputs[index]