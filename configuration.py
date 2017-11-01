
from case_manager import CaseManager


class Configuration():

    def __init__(self, file):
        self.random_range   = (-1.0, 1.0)
        self.epochs         = 100
        self.output_nodes   = 0
        self.features       = 0
        self.casemanager    = CaseManager(file, self) # type: CaseManager
