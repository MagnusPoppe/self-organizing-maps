import json

import os

from kohonen_network.case_manager import CaseManager
from features import calculations as calc

def select_decay(function_name):
    if   function_name == "exponential decay":  return calc.exponential_decay
    elif function_name == "power series":       return calc.power_series_learning_rate_adjust
    elif function_name == "linear decay":       return calc.linear_decay
    elif function_name == "inverse of time":    return calc.inverse_of_time_learning_rate_adjust
    elif function_name == "linear time decay":  return calc.linear_learning_rate_adjust
    else: raise ValueError("Illegal decay function \"%s\"" % function_name)

class Configuration():

    def __init__(self, file):
        with open(file, "r") as js:
            config = json.load(js)

        # Decay function parameters:
        self.initial_neighbourhood   = config["dataset"]["neighbourhood"]["initial"]
        self.neighbourhood_decay     = config["dataset"]["neighbourhood"]["decay"]
        self.neighbourhood_function  = select_decay(config["dataset"]["neighbourhood"]["function"])
        self.learning_rate           = config["dataset"]["learning rate"]["initial"]
        self.learning_rate_decay     = config["dataset"]["learning rate"]["decay"]
        self.learning_rate_function  = select_decay(config["dataset"]["learning rate"]["function"])


        # Network parameters:
        self.epochs                  = config["dataset"]["epochs"]
        self.random_range            = (config["random range"][0], config["random range"][1])
        self.multiplier              = config["node multiplier"]
        try: self.grid               = (config["neurons"][0], config["neurons"][1])
        except KeyError:             pass

        # Case setup:
        self.dataset                 = config["dataset"]["file"]
        self.normalize               = config["normalize"]["use"]
        self.normalization_mode      = config["normalize"]["feature independant"]
        try: self.fraction           = config["dataset"]["fraction"]
        except KeyError:             pass
        try: self.accuracy_testing   = config["dataset"]["accuracy tests"]
        except KeyError:             self.accuracy_testing = False
        try: self.test_rate          = config["dataset"]["test rate (per epoch)"]
        except KeyError:             self.test_rate = 1

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

        if not "mnist" in self.dataset:
            def find_magic_number(digits=2) -> tuple:
                for i in range(digits, 0, -1):
                    if os.path.basename(self.dataset)[0:i].isdigit():
                        return int(os.path.basename(self.dataset)[0:i]), i
                else: raise ValueError("Unknown dataset...")

            number, digits = find_magic_number()
            with open("datasets/TSP/Optimal Value.txt", "r") as f:
                for line in f.readlines():
                    problem, distance = line.split(":")
                    if problem[-digits:].isdigit() and int(problem[-digits:]) == number:
                        self.optimal_distance = int(distance)
                        break
                else: raise EOFError("Optimal distance not found before end of file.")