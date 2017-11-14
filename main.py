import sys
from kohonen_network.Trainer import Trainer
from features.decorators import timer, instanciate_globals, print_time_averages, print_time_totals
from kohonen_network.configuration import Configuration


@timer("Total time: ")
def run(file):
    config = Configuration(file)
    trainer = Trainer( config )
    try:
        trainer.train()
    except KeyboardInterrupt as e:
        print("User cancelled the program.", end="\n")
    finally:
        if config.visuals:
            import matplotlib.pyplot as plt
            plt.close("all")

        # Print statistics
        print_time_totals()
        print_time_averages()
        if config.accuracy_testing:
            t = trainer.test_accuracy(config.casemanager.test, config.casemanager.lbl_test, "Test")
            j = trainer.test_accuracy(config.casemanager.training, config.casemanager.lbl_training, "Training")
            print("\nEnd of run testing:\n\t%s\n\t%s" %(t,j))
        else:
            # TESTING FOR THE TSP TOTAL DISTANCE
            trainer.test_distance(config.optimal_distance)

if __name__ == '__main__':

    # Setup:
    for arg in sys.argv:
        if "TSP" in arg.upper() or "MNIST" in arg.upper():
            file = arg
            break

    instanciate_globals()

    # Running the neural network.
    run(file)