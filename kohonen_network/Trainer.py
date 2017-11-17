import features.calculations as calc
import numexpr as ne
import numpy as np
from graphics.live_graph import LiveGrid
from kohonen_network.network import Network1D, Network2D

from features.decorators import timer
from kohonen_network.configuration import Configuration


class Trainer():

    def __init__(self, configuration: Configuration):

        self.config = configuration
        if configuration.dataset == "mnist":
            self.network = Network2D(configuration)
        else:
            self.network = Network1D(configuration)
        self.neurons = self.network.neurons
        self.total_epochs = 0

        if self.config.visuals and isinstance(self.network, Network1D):
            from graphics.live_graph import LiveGraph
            self.graph  = LiveGraph(self.config.title, "x", "y")
        elif self.config.visuals:
            self.graph = LiveGrid(self.config.title)

    @timer("Training")
    def train(self, start=1, stop=100):
        # Running random cases for a number of epochs:
        bmu=0
        for epoch in range(start, stop):
            # Adjusting parameters:
            learning_rate = self.config.learning_rate_function(
                epoch,
                self.config.learning_rate,
                self.config.learning_rate_decay,
                self.config.epochs)
            sigma = self.config.neighbourhood_function(
                epoch,
                self.config.initial_neighbourhood,
                self.config.neighbourhood_decay,
                self.config.epochs)

            # Running the self organizing map algorithm:
            for case in range(len(self.config.casemanager.training)):
                case, input = self.network.random_input()
                bmu = self.organize_map(input, case, sigma, learning_rate)

            # Displaying visuals if system is configured to do so.
            self.report_to_user(epoch, sigma, learning_rate, bmu)
        self.total_epochs += stop - start

    def train_more(self, epochs):
        self.train(self.total_epochs+1, epochs + self.total_epochs+2)

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
        self.neurons = self.neurons + (neighbourhood * learning_rate * (inn - self.neurons))
        return bmu

    def bmu(self, inputs, nodes):
        out = np.array([])
        for weights in nodes:
            data = np.subtract(inputs, weights)
            data = np.power(data, 2)
            data = np.sqrt(np.sum(data))
            out = np.append(out, [data])
        return np.argmin(out)

    @timer("User feedback")
    def report_to_user(self, epoch, sigma, learning_rate, bmu):
        if self.config.visuals and epoch % self.config.visuals_refresh_rate == 0:
            if self.config.dataset != "mnist":
                actual = list(zip(self.neurons[0], self.neurons[1]))
                self.graph.update(actuals=actual + [actual[0]], targets=self.network.inputs)
            else:
                drawn = self.network.toMatrix(self.network.get_value_mapping())
                self.graph.update(self.network.grid_size, drawn, bmu)

        if epoch % self.config.printout_rate == 0:
            print("Epoch %d: \n\tSigma:         %f \n\tLearning rate: %f" % (epoch, sigma, learning_rate))
        if self.config.accuracy_testing and epoch % self.config.test_rate == 0:
            res = self.test_accuracy(self.config.casemanager.vaidation, self.config.casemanager.lbl_vaidation, "Validation")
            print("\t%s" % res)
            res = self.test_accuracy(self.config.casemanager.training, self.config.casemanager.lbl_training, "Training")
            print("\t%s" % res)
        print()

    def run_all_tests(self):
        if self.config.dataset == "mnist":
            t = self.test_accuracy(self.config.casemanager.test, self.config.casemanager.lbl_test, "Test")
            v = self.test_accuracy(self.config.casemanager.test, self.config.casemanager.lbl_test, "Validation")
            j = self.test_accuracy(self.config.casemanager.training, self.config.casemanager.lbl_training, "Training")
            print("\nTesting all: \n\t%s\n\t%s\n\t%s" %(t,v,j))
        else: print(self.test_distance(self.config.optimal_distance))

    def test_accuracy(self, cases, labels, test_type):
        trained_neurons = self.network.get_value_mapping()
        correct = 0
        for case, label in zip(cases, labels):
            bmu = self.bmu(case, self.neurons)
            if trained_neurons[bmu] == label:
                correct += 1
        return "%s accuracy: %f %s (%d/%d)" %(test_type, correct/len(cases)*100, "%", correct, len(cases))

    def test_distance(self, expected):
        def find_closest_cities():
            cities = []
            for cx, cy in self.network.inputs:
                distance = np.array([])
                for nx, ny in zip(self.neurons[0], self.neurons[1]):
                    distance = np.append(distance, np.sqrt(np.power((nx-cx), 2) + np.power((ny-cy), 2)))
                cities += [np.argmin(distance)]
            return cities

        def create_city_tour(cities):
            import sys
            priority = []
            for x in range(len(self.network.inputs)):
                city = np.argmin(cities)
                cities[city] = sys.maxsize
                priority.append(city)
            return priority

        def validate(city_tour) -> bool:
            all_cities = list(range(0, len(self.network.inputs)))
            perfect = True
            for city in city_tour:
                if city == -1: continue

                if all_cities[city] == -1:
                    print("DUPLICATE CITY. CITY %d VISITED TWICE!" % city)
                    perfect = False
                else:
                    all_cities[city] = -1
            if any(x > -1 for x in all_cities):
                print("SALESMAN DID NOT VISIT ALL CITIES.")
                perfect = False
            return perfect

        close = find_closest_cities()
        city_tour = create_city_tour(close)
        validate(city_tour)

        distance = 0
        previous = city_tour[0]
        for index in city_tour[1:]:
            prev_city, px, py = self.config.casemanager.original[previous]
            city, x, y = self.config.casemanager.original[index]
            p1 = np.power( y - py, 2)
            p2 = np.power( x - px, 2)
            distance += np.sqrt( p1 + p2 )
            previous = index
        print("Total distance: %f / %d)" % (round(distance,2), expected))
        print("Deviation from optimal solution: %f %s" % ( abs( float(distance)/float(expected)*100 - 100) , "%"))