from case_manager import CaseManager
from network import Network1D


class Configuration():

    def __init__(self, file):
        self.casemanager = CaseManager(file) # type: CaseManager
        self.random_range = (-1.0, 1.0)

config = Configuration("datasets/TSP/1.txt")
network = Network1D(config)