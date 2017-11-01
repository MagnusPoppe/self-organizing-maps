
from case_manager import CaseManager


class Configuration():

    def __init__(self, file):
        # Decay function parameters:
        self.decay_sigma                    = 5
        self.decay_lambda                   = -0.5

        # Network parameters:
        self.random_range                   = (-1.0, 1.0)
        self.epochs                         = 1000
        self.learning_rate                  = 0.25
        self.nodes                          = 0
        self.features                       = 0

        # Case setup:
        self.normalize                      = True
        self.normalize_feature_independant  = True
        self.casemanager                    = CaseManager(file, self) # type: CaseManager

        # User interface:
        self.visuals                        = False
        self.visuals_refresh_rate           = 100 # Number of epochs between refreshes.
