import sys
import numpy as np


class CaseManager():

    def __init__(self, file, config):
        if "TSP" in file:     self.dataset = self.read_tsp_file(file, config)
        elif "mnist" in file: self.dataset = self.read_mnist(config)
        else: raise Exception("Unknown case...")

        if config.accuracy_testing:
            # Partitioning testcases for accuracy testing:
            train, valid, test = int(len(self.dataset)*0.80), int(len(self.dataset)*0.90), len(self.dataset)

            # Dataset:
            self.training       = self.dataset[:train]
            self.vaidation      = self.dataset[train:valid]
            self.test           = self.dataset[valid:test]

            # Labels:
            self.lbl_training   = self.labels[:train]
            self.lbl_vaidation  = self.labels[train:valid]
            self.lbl_test       = self.labels[valid:test]
        else:
            self.training       = self.dataset

    def read_tsp_file(self, file, config):
        def normalize(dataset, feature_independant):
            highx = highy = -sys.maxsize
            for data in dataset:
                i, x, y = data
                highx = x if x > highx else highx
                highy = y if y > highy else highy

            if not feature_independant:
                highx = highy = max(highx, highy)

            for i in range(len(dataset)):
                dataset[i][1] = dataset[i][1] / highx
                dataset[i][2] = dataset[i][2] / highy

            config.normalized_by = (highx, highy)
            return dataset
        config.features = 2
        dataset = []
        with open(file, "r") as f:
            name = f.readline().split(":")[1]
            type = f.readline().split(":")[1]
            config.nodes = int(f.readline().split(":")[1])
            skip = f.readline().split(":")[1]
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
            self.original = np.copy(dataset)
            dataset = normalize(dataset, config.normalization_mode)
        return dataset

    def read_mnist(self, config):
        def normalize(dataset, high=255):
            for i in range(len(dataset)):
                dataset[i] = np.array(dataset[i])
                dataset[i] = dataset[i] / high
            return dataset
        from datasets.mnist import mnist_basics
        dataset, labels = mnist_basics.gen_flat_cases(fraction=config.fraction)
        dataset = normalize(dataset)
        self.labels = labels
        config.features = len(dataset[0])
        return dataset

