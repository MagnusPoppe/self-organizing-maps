
from case_manager import CaseManager


class Configuration(): # TODO: Make this read and write to file.

    def __init__(self, file):
        # Decay function parameters:
        self.initial_neighbourhood          = 0.45     # Initial size of neighbourhood
        self.neighbourhood_decay            = 450   # shrink rate of neighbourhood in same unit as decay_sigma

        # Network parameters:
        self.random_range                   = (0, 1.0)
        self.epochs                         = 1000
        self.learning_rate                  = 0.6
        self.learning_rate_decay            = 1000

        # These are set by the application:
        self.nodes                          = 0
        self.features                       = 0

        # Case setup:
        self.normalize                      = True
        self.normalize_feature_independant  = True
        self.normalized_by                  = None
        self.casemanager                    = CaseManager(file, self) # type: CaseManager

        # User interface:
        self.visuals                        = True
        self.visuals_refresh_rate           = 1 # Number of epochs between refreshes.
