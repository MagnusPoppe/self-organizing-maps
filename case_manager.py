import sys


class CaseManager():

    def __init__(self, file, config):
        if "TSP" in file:     self.dataset = self.read_tsp_file(file, config)
        elif "mnist" in file: self.dataset = self.read_mnist(config)
        else: raise Exception("Unknown case...")

    def read_tsp_file(self, file, config):
        config.features = 2
        dataset = []
        with open(file, "r") as f:
            cities = f.readline()
            config.nodes = int(cities.split(":")[1])
            heading = f.readline().strip("\n")
            if heading == "NODE_COORD_SECTION":

                # Looping through all cities.
                for i in range(config.nodes):
                    # Reading line, and removing line ending. Each line is formatted like: 1 42.39 59.102
                    line = f.readline().strip("\n").split(" ")
                    dataset += [[int(line[0]), float(line[1]), float(line[2])]]


                if f.readline().strip("\n") != "EOF":
                    raise Exception("Failed to interpret dataset...")

        if config.normalize:
            dataset = self.normalize(dataset, config.normalization_mode)
        return dataset

    def read_mnist(self, config):
        from datasets.mnist import mnist_basics
        dataset, labels = mnist_basics.gen_flat_cases(fraction=config.fraction)
        self.labels = labels
        config.features = len(dataset[0])
        return dataset

    def normalize(self, dataset, feature_independant):
        highx = highy = -sys.maxsize
        for data in dataset:
            i, x, y = data
            highx = x if x > highx else highx
            highy = y if y > highy else highy

        if not feature_independant:
            highx = highy = max(highx, highy)

        for data in dataset:
            data[1] = data[1] / highx
            data[2] = data[2] / highy
        self.normalized_by = highx, highy
        return dataset