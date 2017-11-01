from node import TSPNode


class CaseManager():

    def __init__(self, file, config):
        if "TSP" in file: self.dataset = self.read_tsp_file(file, config)
        else: raise Exception("Unknown case...")

    def read_tsp_file(self, file, config):
        config.features = 2
        dataset = []
        with open(file, "r") as f:
            cities = f.readline()
            config.output_nodes = int(cities.split(":")[1])
            heading = f.readline().strip("\n")
            if heading == "NODE_COORD_SECTION":

                # Looping through all cities.
                for i in range(config.output_nodes):
                    # Reading line, and removing line ending. Each line is formatted like: 1 42.39 59.102
                    line = f.readline().strip("\n").split(" ")
                    dataset += [(int(line[0]), float(line[1]), float(line[2]))]


                if f.readline().strip("\n") != "EOF":
                    raise Exception("Failed to interpret dataset...")

        return dataset
