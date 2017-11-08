import random

from configuration import Configuration


class Network1D():

    def __init__(self, configuration: Configuration):
        self.config = configuration
        self.features = self.config.features
        self.remaining = []

        # Layers
        self.inputs = [(x, y) for i, x, y in self.config.casemanager.dataset]

    def random_input(self):
        if not self.remaining: self.remaining = list(range(0, self.config.nodes-1))
        index = random.choice(self.remaining)
        # self.remaining.remove(index)
        return index, self.inputs[index]