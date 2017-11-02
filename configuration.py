
from case_manager import CaseManager


class Configuration(): # TODO: Make this read and write to file.

    def __init__(self, file):
        # Decay function parameters:
        self.decay_sigma                    = 0.5     # Initial size of neighbourhood
        self.decay_lambda                   = -0.001  # shrink rate of neighbourhood in same unit as decay_sigma

        # Network parameters:
        self.random_range                   = (0, 1.0)
        self.epochs                         = 1000
        self.learning_rate                  = 0.5
        self.learning_rate_decay            = -0.0001
        self.nodes                          = 0
        self.features                       = 0

        # Case setup:
        self.normalize                      = True
        self.normalize_feature_independant  = False
        self.normalized_by                  = None
        self.casemanager                    = CaseManager(file, self) # type: CaseManager

        # User interface:
        self.visuals                        = False
        self.visuals_refresh_rate           = 100 # Number of epochs between refreshes.
