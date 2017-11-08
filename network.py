import random

from configuration import Configuration

class Network():
    def __init__(self, configuration: Configuration):
        self.config = configuration
        self.features = self.config.features
        self.remaining = []

        # Layers
        self.inputs = self.generate_input_vectors(self.config.casemanager.dataset)
        self.neurons = self.initialize()
        pass

    def random_input(self):
        if not self.remaining: self.remaining = list(range(0, self.config.nodes-1))
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

        return [ [np.random.uniform(*self.config.random_range, size=len(self.inputs[0]))] for i in range(len(self.inputs)*self.config.multiplier)]

class Network2D(Network):

    def __init__(self, configuration: Configuration):
        self.grid_size = (5,5)
        super().__init__(configuration)

    def generate_input_vectors(self, dataset):
        return dataset # No preprocessing needed.

    def initialize(self):
        import numpy as np
        return [np.random.uniform(*self.config.random_range, size=self.grid_size)]