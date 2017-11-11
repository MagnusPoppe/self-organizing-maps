import features.calculations as calc
import numexpr as ne
import numpy as np
from graphics.grid import Grid
from graphics.live_graph import LiveGrid
from kohonen_network.network import Network1D, Network2D

from features.decorators import timer
from kohonen_network.configuration import Configuration


class Trainer():

    def __init__(self, configuration: Configuration):

        self.config = configuration
        if configuration.dataset == "mnist": self.network = Network2D(configuration)
        else:                                self.network = Network1D(configuration)
        self.neurons = self.network.neurons

        if self.config.visuals and isinstance(self.network, Network1D):
            from graphics.live_graph import LiveGraph
            self.graph  = LiveGraph(self.config.title, "x", "y")

        elif self.config.visuals:
            self.graph = LiveGrid(self.config.title)

        print("Setup complete!")
        print("System info:\ncores=%d\nthreads=%d\nVML Version=%s" % (ne.ncores, ne.nthreads, ne.get_vml_version()))

    @timer("Training")
    def train(self):
        # Running random cases for a number of epochs:
        for epoch in range(1, self.config.epochs):
            for case in range(len(self.config.casemanager.training)):

                # Adjusting parameters:
                learning_rate = calc.exponential_decay(epoch, self.config.learning_rate, self.config.learning_rate_decay)
                sigma = calc.exponential_decay(epoch, self.config.initial_neighbourhood, self.config.neighbourhood_decay)

                # Getting input for this run.
                case, input = self.network.random_input()

                # Running the self organizing map algorithm:
                self.organize_map(input, case, sigma, learning_rate)


            # Displaying visuals if system is configured to do so.
            if self.config.visuals and epoch % self.config.visuals_refresh_rate == 0:
                if self.config.dataset != "mnist":
                    actual = list(zip(self.neurons[0], self.neurons[1]))
                    self.graph.update(actuals=actual + [actual[0]], targets=self.network.inputs)
                else:
                    drawn = self.network.toMatrix(self.network.get_value_mapping())
                    self.graph.update(self.network.grid_size, drawn)
            if epoch % self.config.printout_rate == 0:
                res = self.test(self.config.casemanager.vaidation, self.config.casemanager.lbl_vaidation, "Validation")
                print("Epoch %d: \n\tSigma:         %f \n\tLearning rate: %f\n%s" % (epoch,sigma,learning_rate,res))


    def test(self, cases, labels, test_type):
        trained_neurons = self.network.get_value_mapping()
        correct = 0
        for case, label in zip(cases, labels):
            bmu = self.bmu(case, self.neurons)
            if trained_neurons[bmu] == label:
                correct += 1
        return "\t%s accuracy: %f %s (%d/%d)\n" %(test_type, correct/len(cases)*100, "%", correct, len(cases))

    @timer("Organize map")
    def organize_map(self, input, case, sigma, learning_rate):
        inn = np.array(input, ndmin=2)

        # Finding the BMU (Best matching unit)
        if self.config.dataset != "mnist":
            inn = np.transpose(inn)
            bmu = self.bmu(input, list(zip(self.neurons[0], self.neurons[1])))
        else:
            bmu = self.bmu(input, self.neurons)
            self.network.winnerlist[bmu] += [case]

        neighbourhood = self.network.neighbourhood(sigma, bmu, len(self.neurons))

        # CALCULATE USING NumPY:
        self.neurons = self.neurons + (neighbourhood * learning_rate * (inn - self.neurons))

        # CALCULATE USING NumExpr
        # neurons = self.neurons
        # self.neurons = ne.evaluate('neurons + (neighbourhood * learning_rate * (inn - neurons))')

    def bmu(self, inputs, nodes):
        out = np.array([])
        for weights in nodes:
            distance = (inputs-weights)**2
            distance = np.sqrt(np.sum(distance))
            out = np.append(out, [distance])
        return np.argmin(out)