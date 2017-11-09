import json

from case_manager import CaseManager


class Configuration():

    def __init__(self, file):
        with open(file, "r") as js:
            config = json.load(js)

        # Decay function parameters:
        self.initial_neighbourhood   = config["dataset"]["neighbourhood"]["initial"]
        self.neighbourhood_decay     = config["dataset"]["neighbourhood"]["decay"]
        self.neighbourhood_function  = config["dataset"]["neighbourhood"]["function"]
        self.learning_rate           = config["dataset"]["learning rate"]["initial"]
        self.learning_rate_decay     = config["dataset"]["learning rate"]["decay"]
        self.learning_rate_function  = config["dataset"]["learning rate"]["function"]

        # Network parameters:
        self.epochs                  = config["dataset"]["epochs"]
        self.random_range            = (config["random range"][0],config["random range"][1])
        self.multiplier              = 1
        # Case setup:
        self.dataset                 = config["dataset"]["file"]
        self.normalize               = config["normalize"]["use"]
        self.normalization_mode      = config["normalize"]["feature independant"]

        # User interface:
        self.title                   = config["visuals"]["title"]
        self.visuals                 = config["visuals"]["on"]
        self.visuals_refresh_rate    = config["visuals"]["refresh rate"]
        self.printout_rate           = config["visuals"]["print rate"]

        # These are set by the application:
        self.nodes                   = 0
        self.features                = 0
        self.normalized_by           = None
        self.casemanager             = CaseManager(self.dataset, self)
