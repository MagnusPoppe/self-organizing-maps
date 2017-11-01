import numpy as np

from node import TSPNode


class Network1D():

    def __init__(self, configuration):
        self.config = configuration
        self.features = self.config.features

        # Layers
        self.inputs = [(x, y) for i, x, y in self.config.casemanager.dataset]
        self.outputs = []

        # Building network:
        randoms = np.random.uniform(*self.config.random_range, size=self.config.output_nodes)
        for i, value in enumerate(randoms):
            # Setting up nodes
            prev = self.outputs[i - 1].edge_next if i != 0 else None
            self.outputs += [TSPNode(value, edge_prev=prev)]

        # Setting the last edge after loop though to connect the circle.
        self.outputs[0].set_egde_prev(self.outputs[-1].edge_next)
