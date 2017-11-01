
import numpy as np

from node import TSPNode


class Network1D():

    def __init__(self, configuration):
        self.config = configuration
        self.nodes = []
        self.features = self.config.casemanager.features
        self.output_nodes = self.config.casemanager.output_nodes

        # Building network:
        for i, values in enumerate(self.config.casemanager.dataset):
            # Setting up nodes
            prev = self.nodes[i - 1].edge_next if i != 0 else None
            self.nodes += [TSPNode(*values, edge_prev=prev)]

        # Setting the last edge after loop though to connect the circle.
        self.nodes[0].set_egde_prev(self.nodes[-1].edge_next)

