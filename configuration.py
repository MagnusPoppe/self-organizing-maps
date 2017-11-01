
from case_manager import CaseManager


class Configuration():

    def __init__(self, file):
        self.random_range                   = (-1.0, 1.0)
        self.decay_sigma                    = 5
        self.decay_lambda                   = -0.5
        self.epochs                         = 300000
        self.learning_rate                  = 0.25
        self.output_nodes                   = 0
        self.features                       = 0
        self.normalize                      = True
        self.normalize_feature_independant  = True
        self.casemanager                    = CaseManager(file, self) # type: CaseManager
