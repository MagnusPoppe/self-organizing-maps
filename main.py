import sys

from Trainer import Trainer
from configuration import Configuration
from decorators import timer, instanciate_globals, print_time_averages, print_time_totals, table_print_time_dict


@timer("Total time: ")
def run(file):
    config = Configuration(file)
    trainer = Trainer( config )
    try:
        trainer.train()
    except KeyboardInterrupt as e:
        print("User cancelled the program.")
    finally:
        if config.visuals:
            import matplotlib.pyplot as plt
            plt.close("all")

if __name__ == '__main__':
    # Setup:
    file = sys.argv[1] if len(sys.argv) == 2 else  "configurations/mnist.json"
    instanciate_globals()

    # Running the neural network.
    run(file)

    # Print statistics
    print_time_totals()
    print_time_averages()
