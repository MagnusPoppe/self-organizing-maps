import sys
from kohonen_network.Trainer import Trainer
from features.decorators import timer, instanciate_globals, print_time_averages, print_time_totals
from kohonen_network.configuration import Configuration
from shell_commands.listener import listen


@timer("Total time: ")
def run(file):
    config = Configuration(file)
    trainer = Trainer( config )
    try:
        trainer.train( stop=config.epochs )
    except KeyboardInterrupt as e:
        print("User cancelled the program.", end="\n")
    finally:
        if config.visuals:
            import matplotlib.pyplot as plt
            plt.close("all")

        # Print statistics
        print_time_totals()
        print_time_averages()

        # Running tests:
        trainer.run_all_tests()
        return trainer

if __name__ == '__main__':

    # Setup:
    for arg in sys.argv:
        if "TSP" in arg.upper() or "MNIST" in arg.upper():
            file = arg
            break
    instanciate_globals()

    # Running the neural network.
    controller = run(file)

    listen(controller)