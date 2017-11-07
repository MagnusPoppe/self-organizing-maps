
from case_manager import CaseManager


class Configuration(): # TODO: Make this read and write to file.

    def __init__(self, file):
        # Decay function parameters:
        self.initial_neighbourhood          = 2
        self.neighbourhood_decay            = 1000

        # Network parameters:
        self.random_range                   = (0, 1)
        self.epochs                         = 10000
        self.learning_rate                  = 0.40
        self.learning_rate_decay            = 655

        # These are set by the application:
        self.nodes                          = 0
        self.features                       = 0

        # Case setup:
        self.normalize                      = True
        self.normalize_feature_independant  = True
        self.normalized_by                  = None
        self.casemanager                    = CaseManager(file, self)

        # User interface:
        self.visuals                        = True
        self.visuals_refresh_rate           = 50
        self.printout_rate                  = 500
