from Trainer import Trainer
from configuration import Configuration
from decorators import timer, instanciate_globals, print_time_averages, print_time_totals, table_print_time_dict


@timer("Total time: ")
def run():
    config = Configuration("configurations/mnist.json")
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
    instanciate_globals()
    run()
    print_time_totals()
    print_time_averages()
