from node import TSPNode


class CaseManager():

    def __init__(self, file):
        self.output_nodes = 0
        self.features = 0

        if "TSP" in file: self.dataset = self.read_tsp_file(file)
        else: raise Exception("Unknown case...")

    def read_tsp_file(self, file):
        self.features = 2
        dataset = []
        with open(file, "r") as f:
            cities = f.readline()
            self.output_nodes = int(cities.split(":")[1])
            heading = f.readline().strip("\n")
            if heading == "NODE_COORD_SECTION":

                # Looping through all cities.
                for i in range(self.output_nodes):
                    # Reading line, and removing line ending. Each line is formatted like: 1 42.39 59.102
                    line = f.readline().strip("\n").split(" ")
                    dataset += [(int(line[0]), float(line[1]), float(line[2]))]


                if f.readline().strip("\n") != "EOF":
                    raise Exception("Failed to interpret dataset...")

        return dataset
